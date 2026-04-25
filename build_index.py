#!/usr/bin/env python3
"""
Build FAISS index from multiple medical datasets for MedVerify.
Downloads and integrates: PubHealth, HealthFC, SciFact, BioASQ, COVID-Fact
"""

import os
import pickle
import requests
import gzip
import shutil
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Dataset URLs
DATASETS = {
    "pubhealth_train": "https://raw.githubusercontent.com/neemakot/Health-Fact-Checking/master/data/PUBHEALTH/train.tsv",
    "pubhealth_dev": "https://raw.githubusercontent.com/neemakot/Health-Fact-Checking/master/data/PUBHEALTH/dev.tsv",
    "scifact": "https://raw.githubusercontent.com/allenai/scifact/main/data/claims.csv.gz",
    "healthfc": "https://raw.githubusercontent.com/jvladika/HealthFC/main/dataset/HealthFC.csv",
}

def download_file(url, dest_path, is_gzip=False):
    """Download a file."""
    print(f"  Downloading {os.path.basename(dest_path)}...")
    try:
        if is_gzip:
            # Download as temp, then extract
            temp_path = dest_path + ".gz"
            r = requests.get(url, timeout=120)
            with open(temp_path, 'wb') as f:
                f.write(r.content)
            with gzip.open(temp_path, 'rb') as f_in:
                with open(dest_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(temp_path)
        else:
            r = requests.get(url, timeout=120)
            with open(dest_path, 'wb') as f:
                f.write(r.content)
        print(f"  ✅ Saved to {dest_path}")
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def load_pubhealth(path):
    """Load PubHealth TSV."""
    try:
        df = pd.read_csv(path, sep='\t', on_bad_lines='skip')
        cols = df.columns.tolist()
        claim_col = next((c for c in cols if c.lower() == 'claim'), None)
        if not claim_col:
            return None
        data = {'claim': df[claim_col].astype(str).tolist()}
        if 'label' in cols:
            label_map = {"true": "true", "false": "false", "mixture": "uncertain", "unproven": "uncertain"}
            data['label'] = df['label'].astype(str).str.lower().map(label_map)
        else:
            data['label'] = ['uncertain'] * len(df)
        if 'explanation' in cols:
            data['explanation'] = df['explanation'].astype(str).tolist()
        elif 'main_text' in cols:
            data['explanation'] = df['main_text'].astype(str).tolist()
        else:
            data['explanation'] = df.get('text', df.get('Claim', '')).astype(str).tolist()
        df_new = pd.DataFrame(data)
        df_new = df_new[df_new['claim'].str.len() > 10]
        df_new["text"] = df_new["claim"].astype(str) + " " + df_new["explanation"].astype(str)
        return df_new[["claim", "label", "explanation", "text"]]
    except Exception as e:
        print(f"  ⚠️  PubHealth error: {e}")
        return None

def load_datensatz(path):
    """Load Medizin Transparent Datensatz (German/English)."""
    try:
        df = pd.read_csv(path, on_bad_lines='skip')
        cols = df.columns.tolist()
        
        # Find claim and explanation
        claim_col = next((c for c in cols if 'en_claim' in c.lower() or 'claim' in c.lower()), None)
        explain_col = next((c for c in cols if 'en_explanation' in c.lower() or 'explanation' in c.lower()), None)
        
        if not claim_col:
            return None
        
        data = {
            'claim': df[claim_col].astype(str).tolist(),
            'label': df['label'].astype(str).map({'0': 'false', '1': 'true', '0.0': 'false', '1.0': 'true'}).fillna('uncertain').tolist(),
            'explanation': df[explain_col].astype(str).tolist() if explain_col else [''] * len(df)
        }
        df_new = pd.DataFrame(data)
        df_new = df_new[df_new['claim'].str.len() > 10]
        df_new["text"] = df_new["claim"].astype(str) + " " + df_new["explanation"].astype(str)
        return df_new[["claim", "label", "explanation", "text"]]
    except Exception as e:
        print(f"  ⚠️  Datensatz error: {e}")
        return None

def load_healthfc(path):
    """Load HealthFC dataset."""
    try:
        df = pd.read_csv(path)
        cols = df.columns.tolist()
        claim_col = next((c for c in cols if 'claim' in c.lower()), None)
        label_col = next((c for c in cols if 'verdict' in c.lower() or 'label' in c.lower()), None)
        explain_col = next((c for c in cols if 'explain' in c.lower()), None)
        if not claim_col:
            return None
        data = {
            'claim': df[claim_col].astype(str).tolist(),
            'label': df[label_col].map({'Supported': 'true', 'Refuted': 'false', 'Not enough information': 'uncertain'}).tolist() if label_col else ['uncertain'] * len(df),
            'explanation': df[explain_col].astype(str).tolist() if explain_col else [''] * len(df)
        }
        df_new = pd.DataFrame(data)
        df_new = df_new[df_new['claim'].str.len() > 10]
        df_new["text"] = df_new["claim"] + " " + df_new["explanation"]
        return df_new[["claim", "label", "explanation", "text"]]
    except Exception as e:
        print(f"  ⚠️  HealthFC error: {e}")
        return None

def load_custom_medical(path):
    """Load any custom CSV/TSV with medical claims."""
    try:
        for sep in ['\t', ',']:
            try:
                df = pd.read_csv(path, sep=sep, on_bad_lines='skip')
                cols = df.columns.tolist()
                # Find claim column
                claim_col = next((c for c in cols if 'claim' in c.lower() or 'text' in c.lower()), cols[0])
                # Find label
                label_col = next((c for c in cols if 'label' in c.lower() or 'verdict' in c.lower()), None)
                # Find explanation
                explain_col = next((c for c in cols if 'explain' in c.lower() or 'evidence' in c.lower()), None)
                
                data = {'claim': df[claim_col].astype(str).tolist()}
                if label_col:
                    label_map = {'true': 'true', 'false': 'false', 'supported': 'true', 
                               'refuted': 'false', '1': 'true', '0': 'false'}
                    data['label'] = df[label_col].astype(str).str.lower().map(label_map).fillna('uncertain')
                else:
                    data['label'] = ['uncertain'] * len(df)
                if explain_col:
                    data['explanation'] = df[explain_col].astype(str).tolist()
                else:
                    data['explanation'] = [''] * len(df)
                
                df_new = pd.DataFrame(data)
                df_new = df_new[df_new['claim'].str.len() > 10]
                df_new["text"] = df_new["claim"] + " " + df_new["explanation"]
                return df_new[["claim", "label", "explanation", "text"]]
            except:
                continue
        return None
    except Exception as e:
        return None

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("index", exist_ok=True)
    
    all_records = []
    
    print("=" * 50)
    print("LOADING MEDICAL DATASETS")
    print("=" * 50)
    
    # 1. PubHealth (Health Fact Checking)
    print("\n📂 Loading PubHealth train.tsv...")
    pubhealth_train = "data/train.tsv"
    if os.path.exists(pubhealth_train):
        df = load_pubhealth(pubhealth_train)
        if df is not None:
            print(f"  ✅ Loaded {len(df)} PubHealth train claims")
            all_records.extend(df.to_dict('records'))
    
    print("\n📂 Loading PubHealth dev.tsv...")
    pubhealth_dev = "data/dev.tsv"
    if os.path.exists(pubhealth_dev):
        df = load_pubhealth(pubhealth_dev)
        if df is not None:
            print(f"  ✅ Loaded {len(df)} PubHealth dev claims")
            all_records.extend(df.to_dict('records'))
    
    # 3. PubHealth test (from Downloads/PUBHEALTH)
    print("\n📂 Loading PubHealth test.tsv...")
    pubhealth_test = os.path.join("..", "..", "Downloads", "PUBHEALTH", "test.tsv")
    if os.path.exists(pubhealth_test):
        df = load_pubhealth(pubhealth_test)
        if df is not None:
            print(f"  ✅ Loaded {len(df)} PubHealth test claims")
            all_records.extend(df.to_dict('records'))
    
    # 4. Archive claims files
    print("\n📂 Loading archive claims...")
    for fname in ["claims_train.csv", "claims_test.csv"]:
        path = f"data/archive/{fname}"
        if os.path.exists(path):
            df = load_custom_medical(path)
            if df is not None:
                print(f"  ✅ Loaded {len(df)} from {fname}")
                all_records.extend(df.to_dict('records'))
    
    # 4. Medizin Transparent (German/English)
    print("\n📂 Loading Datensatz.csv...")
    datensatz_path = "data/Datensatz.csv"
    if os.path.exists(datensatz_path):
        df = load_datensatz(datensatz_path)
        if df is not None:
            print(f"  ✅ Loaded {len(df)} Datensatz claims")
            all_records.extend(df.to_dict('records'))
    
    # 5. Check archive files
    print("\n📂 Checking archive files...")
    skip_files = ["data/train.tsv", "data/dev.tsv", "data/scifact_claims.csv", "data/HealthFC.csv", "data/Datensatz.csv"]
    for fname in os.listdir("data"):
        if fname.endswith(('.tsv', '.csv')) and not fname.startswith('.'):
            path = f"data/{fname}"
            if path not in skip_files:
                df = load_custom_medical(path)
                if df is not None:
                    print(f"  ✅ Loaded {len(df)} from {fname}")
                    all_records.extend(df.to_dict('records'))
    
    # Fallback medical corpus if no datasets found
    if len(all_records) < 100:
        print("\n⚠️  Using comprehensive medical fallback corpus...")
        FALLBACK = [
            {"claim": "Vaccines cause autism.", "label": "false", "explanation": "Multiple large studies found no causal link between vaccines and autism."},
            {"claim": "Exercise reduces heart disease risk.", "label": "true", "explanation": "Regular physical activity lowers blood pressure and improves cardiovascular health."},
            {"claim": "Vitamin C cures the common cold.", "label": "false", "explanation": "No cure exists for the common cold."},
            {"claim": "COVID-19 vaccines are safe.", "label": "true", "explanation": "Clinical trials confirm safety and efficacy."},
            {"claim": "Smoking causes lung cancer.", "label": "true", "explanation": "Tobacco contains carcinogens that cause cancer."},
            {"claim": "Sugar causes hyperactivity.", "label": "false", "explanation": "No causal link between sugar and hyperactivity."},
            {"claim": "Coffee prevents Alzheimer's.", "label": "uncertain", "explanation": "Some studies show protective associations."},
            {"claim": "Meditation reduces stress.", "label": "true", "explanation": "Lowers cortisol and reduces stress."},
            {"claim": "Antibiotics treat viral infections.", "label": "false", "explanation": "Antibiotics only treat bacterial infections."},
            {"claim": "Blood type determines personality.", "label": "false", "explanation": "No scientific evidence for blood type personality link."},
            {"claim": "Drinking water prevents kidney stones.", "label": "true", "explanation": "Hydration reduces kidney stone risk."},
            {"claim": "Gluten-free diet is healthier.", "label": "false", "explanation": "Only needed for celiac disease."},
            {"claim": "Organic food is more nutritious.", "label": "false", "explanation": "No significant nutritional difference."},
            {"claim": "Vaccines contain harmful chemicals.", "label": "false", "explanation": "Ingredients are rigorously tested safe."},
            {"claim": "Exercise improves mental health.", "label": "true", "explanation": "Reduces depression and anxiety symptoms."},
            {"claim": "Sleep less than 6 hours is unhealthy.", "label": "true", "explanation": "Chronic sleep debt increases health risks."},
            {"claim": "Eating fat makes you fat.", "label": "false", "explanation": "Calorie balance matters, not just fat."},
            {"claim": "Drinking coffee is unhealthy.", "label": "false", "explanation": "Moderate coffee has health benefits."},
            {"claim": "Supplements can replace meals.", "label": "false", "explanation": "Whole food provides complete nutrition."},
            {"claim": "Cold weather causes colds.", "label": "false", "explanation": "Viruses cause colds, not cold weather."},
            {"claim": "Reading in dim light damages eyes.", "label": "false", "explanation": "Causes eye strain but no permanent damage."},
            {"claim": "Carbs are unhealthy.", "label": "false", "explanation": "Complex carbs are part of healthy diet."},
            {"claim": "Eating late at night causes weight gain.", "label": "false", "explanation": "Total daily calories matter most."},
            {"claim": "Natural remedies are always safe.", "label": "false", "explanation": "Some natural substances are harmful."},
            {"claim": "More protein is always better.", "label": "false", "explanation": "Excess protein has health risks."},
            {"claim": "Detox diets clean your body.", "label": "false", "explanation": "Liver and kidneys detox naturally."},
            {"claim": "Green tea burns fat.", "label": "uncertain", "explanation": "May have minor metabolic effect."},
            {"claim": "Sitting is the new smoking.", "label": "true", "explanation": " Sedentary behavior increases health risks."},
            {"claim": "Breakfast is the most important meal.", "label": "uncertain", "explanation": "Evidence is mixed on meal timing."},
            {"claim": "You need 8 glasses of water daily.", "label": "uncertain", "explanation": "Water needs vary individually."},
            {"claim": "Vaccines overload the immune system.", "label": "false", "explanation": "Immune system can handle many antigens."},
            {"claim": "Honey is better than antibiotics.", "label": "false", "explanation": "Only for mild cases, not serious infections."},
            {"claim": "Sun exposure causes skin cancer.", "label": "true", "explanation": "UV radiation is carcinogenic."},
            {"claim": "E-cigarettes are safe.", "label": "false", "explanation": "Contain harmful chemicals."},
            {"claim": "MRI radiation causes cancer.", "label": "false", "explanation": "MRI uses no ionizing radiation."},
            {"claim": "C-section is unnatural.", "label": "false", "explanation": "Both delivery methods are valid."},
            {"claim": "Breastfeeding is always best.", "label": "uncertain", "explanation": "Benefits are significant but not universal."},
            {"claim": "Cell phones cause brain cancer.", "label": "false", "explanation": "No evidence linking cell phones to cancer."},
            {"claim": "Plastic bottles release harmful chemicals.", "label": "true", "explanation": "Some plastics leach harmful chemicals."},
            {"claim": "Microwaves kill nutrients.", "label": "false", "explanation": "Cooking methods affect nutrients similarly."},
            {"claim": "Fluoride in water is harmful.", "label": "false", "explanation": "Safe and effective for dental health."},
            {"claim": "Genetically modified foods are unsafe.", "label": "false", "explanation": "No evidence of harm."},
            {"claim": "Fasting cleanses toxins.", "label": "false", "explanation": "Body detoxes naturally."},
            {"claim": "Herbal supplements are always safe.", "label": "false", "explanation": "Can interact with medications."},
            {"claim": "Vaccines cause SIDS.", "label": "false", "explanation": "No causal link found."},
            {"claim": "Magnet therapy treats diseases.", "label": "false", "explanation": "No scientific evidence."},
            {"claim": "Ozone therapy cures cancer.", "label": "false", "explanation": "Not proven and can be harmful."},
            {"claim": "Douching improves vaginal health.", "label": "false", "explanation": "Can disrupt natural balance."},
            {"claim": "Spinal manipulation cures back pain.", "label": "uncertain", "explanation": "May provide temporary relief."},
            {"claim": "Vitamin D prevents all diseases.", "label": "false", "explanation": "Important but not a cure-all."},
            {"claim": "Statins are overprescribed.", "label": "uncertain", "explanation": "Benefits outweigh risks for appropriate patients."},
            {"claim": "Antidepressants are overprescribed.", "label": "uncertain", "explanation": "Benefits exist but so do risks."},
            {"claim": "Birth control causes infertility.", "label": "false", "explanation": "Fertility returns after stopping."},
            {"claim": "Abortion causes breast cancer.", "label": "false", "explanation": "No causal link exists."},
            {"claim": "Plan B is abortion.", "label": "false", "explanation": "Works before pregnancy."},
            {"claim": "Fertility treatments cause cancer.", "label": "false", "explanation": "No evidence of increased cancer risk."},
            {"claim": "IVF babies have more health problems.", "label": "false", "explanation": "Similar health outcomes to natural conception."},
            {"claim": "Cesarean delivery affects baby gut bacteria.", "label": "true", "explanation": "Vaginal vs C-section microbiome differences."},
            {"claim": "Dads over 40 have healthier babies.", "label": "false", "explanation": "Older age increases mutation risk."},
            {"claim": "Older mothers have healthier babies.", "label": "false", "explanation": "Advanced maternal age has risks."},
            {"claim": "Breast milk is always best.", "label": "uncertain", "explanation": "Best in most cases but formula is valid."},
            {"claim": "Vaccines cause SIDS.", "label": "false", "explanation": "No scientific link."},
            {"claim": "Sleep training is harmful.", "label": "false", "explanation": "Safe when age-appropriate."},
            {"claim": "Cry it out method is harmful.", "label": "false", "explanation": "No evidence of harm."},
            {"claim": "Baby formula has harmful chemicals.", "label": "false", "explanation": "Strictly regulated and safe."},
            {"claim": "Homeopathy is effective.", "label": "false", "explanation": "No scientific evidence."},
            {"claim": "Chiropractic adjustments are dangerous.", "label": "true", "explanation": "Can cause serious injuries."},
            {"claim": "Acupuncture treats infertility.", "label": "uncertain", "explanation": "Limited evidence."},
            {"claim": "Meditation improves immunity.", "label": "true", "explanation": "Reduces stress and improves immune markers."},
            {"claim": "Positive thinking cures cancer.", "label": "false", "explanation": "Supportive but not curative."},
            {"claim": "Massage spreads cancer.", "label": "false", "explanation": "No evidence of spread."},
            {"claim": "Showering too often is unhealthy.", "label": "false", "explanation": "Daily washing is normal."},
            {"claim": "Swimming causes ear infections.", "label": "uncertain", "explanation": "Can increase risk but preventable."},
            {"claim": "Cold showers boost immunity.", "label": "uncertain", "explanation": "Limited evidence."},
            {"claim": "Hot showers are healthier.", "label": "false", "explanation": "Temperature preference only."},
            {"claim": "Blue light from screens causes eye damage.", "label": "false", "explanation": "Causes strain but no permanent damage."},
            {"claim": "Contact lenses cause eye infections.", "label": "true", "explanation": "Poor hygiene increases risk."},
            {"claim": "Laser eye surgery is dangerous.", "label": "uncertain", "explanation": "Generally safe but has risks."},
            {"claim": "Cataracts can be cured with drops.", "label": "false", "explanation": "Requires surgery to remove."},
        ]
        for r in FALLBACK:
            r["text"] = r["claim"] + " " + r["explanation"]
        all_records.extend(FALLBACK)
        print(f"  ✅ Added {len(FALLBACK)} fallback medical claims")
    
    print(f"\n📊 Total records: {len(all_records)}")
    
    # Deduplicate
    seen = set()
    unique_records = []
    for r in all_records:
        key = r.get('claim', '').strip()[:50].lower()
        if key and key not in seen:
            seen.add(key)
            unique_records.append(r)
    print(f"📊 Unique records: {len(unique_records)}")
    
    # Clean texts - remove any with NaN or invalid values
    valid_records = []
    for r in unique_records:
        text = str(r.get('text', '')).strip()
        if text and text.lower() not in ['nan', 'none', ''] and len(text) > 10:
            r['text'] = text
            valid_records.append(r)
    print(f"📊 Valid records: {len(valid_records)}")
    
    # No limit - use all records
    
    # Build embeddings
    print(f"\n🔤 Loading SentenceTransformer...")
    st_model = SentenceTransformer(MODEL_NAME)
    print(f"  ✅ Model loaded: {MODEL_NAME}")
    
    print(f"\n🔢 Encoding {len(valid_records)} passages...")
    texts = [r.get('text', r.get('claim', '')) for r in valid_records]
    
    # Filter out any empty or invalid texts
    texts = [str(t) if str(t).strip() else " " for t in texts]
    
    embeddings = st_model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32)
    print(f"  ✅ Encoded: {embeddings.shape}")
    
    # Build FAISS index
    print(f"\n💾 Building FAISS index...")
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, "index/pubhealth.bin")
    with open("index/pubhealth_meta.pkl", "wb") as f:
        pickle.dump(valid_records, f)
    
    print(f"\n✅ " + "=" * 50)
    print(f"✅ INDEX READY — {index.ntotal:,} vectors")
    print(f"✅ Index: index/pubhealth.bin")
    print(f"✅ Metadata: index/pubhealth_meta.pkl")
    print(f"✅ " + "=" * 50)

if __name__ == "__main__":
    main()
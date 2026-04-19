"""
Dataset Generator for Smart Health Monitoring System
=====================================================
Generates a synthetic symptoms-disease dataset for training.
Real-world alternatives (Kaggle):
  - https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
  - https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
  - https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset
"""

import pandas as pd
import numpy as np
import os

# Reproducibility
np.random.seed(42)

# ─── Disease definitions ──────────────────────────────────────────────────────
DISEASES = {
    "Common Cold": {
        "symptoms": ["runny_nose", "sore_throat", "sneezing", "mild_fever", "cough", "fatigue"],
        "temp_range": (37.0, 38.5),
        "hr_range": (65, 85),
        "risk": "Low",
        "weight": 0.15,
    },
    "Influenza": {
        "symptoms": ["high_fever", "body_aches", "fatigue", "cough", "headache", "chills", "sore_throat"],
        "temp_range": (38.5, 40.0),
        "hr_range": (80, 105),
        "risk": "Medium",
        "weight": 0.12,
    },
    "Pneumonia": {
        "symptoms": ["high_fever", "cough", "chest_pain", "shortness_of_breath", "fatigue", "chills"],
        "temp_range": (38.8, 40.5),
        "hr_range": (90, 120),
        "risk": "High",
        "weight": 0.08,
    },
    "Diabetes Type 2": {
        "symptoms": ["frequent_urination", "excessive_thirst", "fatigue", "blurred_vision", "slow_healing", "numbness"],
        "temp_range": (36.5, 37.5),
        "hr_range": (70, 95),
        "risk": "High",
        "weight": 0.10,
    },
    "Hypertension": {
        "symptoms": ["headache", "dizziness", "blurred_vision", "chest_pain", "shortness_of_breath", "nosebleed"],
        "temp_range": (36.5, 37.5),
        "hr_range": (85, 115),
        "risk": "High",
        "weight": 0.10,
    },
    "Gastroenteritis": {
        "symptoms": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "mild_fever", "fatigue"],
        "temp_range": (37.2, 38.8),
        "hr_range": (75, 100),
        "risk": "Medium",
        "weight": 0.10,
    },
    "Migraine": {
        "symptoms": ["severe_headache", "nausea", "sensitivity_to_light", "dizziness", "blurred_vision"],
        "temp_range": (36.5, 37.2),
        "hr_range": (60, 85),
        "risk": "Low",
        "weight": 0.08,
    },
    "Asthma": {
        "symptoms": ["shortness_of_breath", "wheezing", "cough", "chest_tightness", "fatigue"],
        "temp_range": (36.5, 37.5),
        "hr_range": (80, 110),
        "risk": "Medium",
        "weight": 0.08,
    },
    "Anemia": {
        "symptoms": ["fatigue", "pale_skin", "shortness_of_breath", "dizziness", "cold_hands", "headache"],
        "temp_range": (36.0, 37.0),
        "hr_range": (85, 110),
        "risk": "Medium",
        "weight": 0.07,
    },
    "COVID-19": {
        "symptoms": ["high_fever", "cough", "shortness_of_breath", "fatigue", "loss_of_taste", "body_aches", "headache"],
        "temp_range": (37.8, 40.0),
        "hr_range": (75, 110),
        "risk": "High",
        "weight": 0.12,
    },
}

# Full symptom vocabulary
ALL_SYMPTOMS = sorted({
    s for d in DISEASES.values() for s in d["symptoms"]
})

RISK_MAP = {"Low": 0, "Medium": 1, "High": 2}


def generate_record(disease_name: str, disease_info: dict) -> dict:
    """Generate one patient record for a given disease."""
    record = {s: 0 for s in ALL_SYMPTOMS}

    # Core symptoms always present
    for sym in disease_info["symptoms"][:3]:
        record[sym] = 1

    # Additional symptoms with 60-80% probability
    for sym in disease_info["symptoms"][3:]:
        if np.random.random() < 0.70:
            record[sym] = 1

    # Add 1-2 random noise symptoms occasionally
    noise_pool = [s for s in ALL_SYMPTOMS if s not in disease_info["symptoms"]]
    if np.random.random() < 0.25:
        noise_sym = np.random.choice(noise_pool, size=min(2, len(noise_pool)), replace=False)
        for ns in noise_sym:
            record[ns] = 1

    # Physiological readings
    lo, hi = disease_info["temp_range"]
    record["body_temperature"] = round(np.random.uniform(lo, hi) + np.random.normal(0, 0.15), 2)

    lo, hi = disease_info["hr_range"]
    record["heart_rate"] = int(np.random.uniform(lo, hi) + np.random.normal(0, 3))

    record["disease"] = disease_name
    record["risk_level"] = disease_info["risk"]
    record["risk_encoded"] = RISK_MAP[disease_info["risk"]]

    return record


def generate_dataset(n_samples: int = 3000) -> pd.DataFrame:
    """Generate the full dataset."""
    records = []
    names = list(DISEASES.keys())
    weights = [DISEASES[d]["weight"] for d in names]
    total = sum(weights)
    weights = [w / total for w in weights]

    for _ in range(n_samples):
        chosen = np.random.choice(names, p=weights)
        records.append(generate_record(chosen, DISEASES[chosen]))

    df = pd.DataFrame(records)

    # Ensure heart_rate is within physiological range
    df["heart_rate"] = df["heart_rate"].clip(40, 200)
    df["body_temperature"] = df["body_temperature"].clip(35.0, 42.0)

    return df


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = generate_dataset(3000)

    train_path = os.path.join(out_dir, "health_data.csv")
    df.to_csv(train_path, index=False)
    print(f"[OK] Dataset saved -> {train_path}")
    print(f"   Shape : {df.shape}")
    print(f"   Diseases:\n{df['disease'].value_counts().to_string()}")
    print(f"   Risk levels:\n{df['risk_level'].value_counts().to_string()}")
    print(f"   Symptoms ({len(ALL_SYMPTOMS)}): {ALL_SYMPTOMS}")

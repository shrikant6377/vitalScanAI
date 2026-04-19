"""
utils/preprocessor.py
=====================
Data cleaning, feature engineering, and encoding helpers.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import os

# ─── Constants ────────────────────────────────────────────────────────────────
SYMPTOM_COLS = [
    "abdominal_pain", "blurred_vision", "body_aches", "chest_pain",
    "chest_tightness", "chills", "cold_hands", "cough", "diarrhea",
    "dizziness", "excessive_thirst", "fatigue", "frequent_urination",
    "headache", "high_fever", "loss_of_taste", "mild_fever", "nausea",
    "nosebleed", "numbness", "pale_skin", "runny_nose",
    "sensitivity_to_light", "severe_headache", "shortness_of_breath",
    "slow_healing", "sneezing", "sore_throat", "vomiting", "wheezing",
]
VITAL_COLS = ["heart_rate", "body_temperature"]
FEATURE_COLS = SYMPTOM_COLS + VITAL_COLS
TARGET_DISEASE = "disease"
TARGET_RISK = "risk_level"

NORMAL_HR = (60, 100)          # bpm
NORMAL_TEMP = (36.1, 37.2)     # °C

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


# ─── Risk Score ───────────────────────────────────────────────────────────────
def compute_risk_score(row: dict) -> float:
    """
    Heuristic risk score [0.0 - 1.0].
    Combines symptom count, heart-rate deviation, and temperature deviation.
    """
    # Symptom severity (30 possible)
    sym_count = sum(row.get(s, 0) for s in SYMPTOM_COLS)
    sym_score = min(sym_count / 8.0, 1.0)          # saturate at 8+ symptoms

    # Heart-rate deviation
    hr = float(row.get("heart_rate", 75))
    if hr < NORMAL_HR[0]:
        hr_dev = (NORMAL_HR[0] - hr) / NORMAL_HR[0]
    elif hr > NORMAL_HR[1]:
        hr_dev = (hr - NORMAL_HR[1]) / NORMAL_HR[1]
    else:
        hr_dev = 0.0
    hr_score = min(hr_dev, 1.0)

    # Temperature deviation
    temp = float(row.get("body_temperature", 37.0))
    if temp < NORMAL_TEMP[0]:
        temp_dev = (NORMAL_TEMP[0] - temp) / NORMAL_TEMP[0]
    elif temp > NORMAL_TEMP[1]:
        temp_dev = (temp - NORMAL_TEMP[1]) / NORMAL_TEMP[1]
    else:
        temp_dev = 0.0
    temp_score = min(temp_dev * 10, 1.0)   # amplify - temperature changes are small

    # Weighted combination
    score = 0.50 * sym_score + 0.25 * hr_score + 0.25 * temp_score
    return round(float(score), 4)


def score_to_risk_label(score: float) -> str:
    if score < 0.30:
        return "Low"
    elif score < 0.60:
        return "Medium"
    return "High"


# ─── Preprocessing ────────────────────────────────────────────────────────────
class HealthPreprocessor:
    """
    Handles cleaning, encoding, and scaling for the health dataset.
    Fitted objects are persisted to disk for consistent inference.
    """

    def __init__(self):
        self.disease_encoder = LabelEncoder()
        self.risk_encoder = LabelEncoder()
        self.vital_scaler = MinMaxScaler()
        self._fitted = False

    # ── Fit ─────────────────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> "HealthPreprocessor":
        df = self._validate_and_clean(df)
        self.disease_encoder.fit(df[TARGET_DISEASE])
        self.risk_encoder.fit(df[TARGET_RISK])
        self.vital_scaler.fit(df[VITAL_COLS])
        self._fitted = True
        return self

    # ── Transform ───────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")

        df = self._validate_and_clean(df.copy())

        # Scale vitals
        df[VITAL_COLS] = self.vital_scaler.transform(df[VITAL_COLS])

        # Add computed risk score
        df["risk_score"] = df.apply(
            lambda r: compute_risk_score(
                {**{s: r[s] for s in SYMPTOM_COLS},
                 "heart_rate": r["heart_rate"],
                 "body_temperature": r["body_temperature"]}
            ), axis=1
        )
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ── Encode targets ──────────────────────────────────────────────────────
    def encode_targets(self, df: pd.DataFrame):
        y_disease = self.disease_encoder.transform(df[TARGET_DISEASE])
        y_risk = self.risk_encoder.transform(df[TARGET_RISK])
        return y_disease, y_risk

    # ── Single-row inference ─────────────────────────────────────────────────
    def transform_input(self, input_dict: dict) -> np.ndarray:
        """Convert a single prediction request dict -> feature array."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Load from disk first.")

        row = {s: 0 for s in SYMPTOM_COLS}
        row.update({s: 1 for s in input_dict.get("symptoms", []) if s in SYMPTOM_COLS})

        hr = float(input_dict.get("heart_rate", 75))
        temp = float(input_dict.get("body_temperature", 37.0))

        vital_scaled = self.vital_scaler.transform([[hr, temp]])[0]
        row["heart_rate"] = vital_scaled[0]
        row["body_temperature"] = vital_scaled[1]

        risk_score = compute_risk_score({
            **{s: input_dict.get(s, row.get(s, 0)) for s in SYMPTOM_COLS},
            "heart_rate": hr,
            "body_temperature": temp,
        })
        row["risk_score"] = risk_score

        feature_cols = FEATURE_COLS + ["risk_score"]
        return np.array([[row[c] for c in feature_cols]])

    # ── Persistence ─────────────────────────────────────────────────────────
    def save(self, directory: str = MODELS_DIR):
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self, os.path.join(directory, "preprocessor.pkl"))
        print(f"   [OK] Preprocessor saved -> {directory}/preprocessor.pkl")

    @classmethod
    def load(cls, directory: str = MODELS_DIR) -> "HealthPreprocessor":
        path = os.path.join(directory, "preprocessor.pkl")
        obj = joblib.load(path)
        return obj

    # ── Internal helpers ─────────────────────────────────────────────────────
    @staticmethod
    def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        # Symptom columns -> binary int
        for col in SYMPTOM_COLS:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0).astype(int).clip(0, 1)

        # Vitals -> float, fill with physiological defaults
        df["heart_rate"] = pd.to_numeric(df.get("heart_rate", 75), errors="coerce").fillna(75).clip(30, 220)
        df["body_temperature"] = pd.to_numeric(df.get("body_temperature", 37.0), errors="coerce").fillna(37.0).clip(34.0, 43.0)

        return df


# ─── CSV Upload Processor ─────────────────────────────────────────────────────
def process_uploaded_csv(filepath: str) -> dict:
    """
    Read a user-uploaded CSV of historical health records.
    Returns summary stats and trend-ready data.
    """
    df = pd.read_csv(filepath)

    required = ["heart_rate", "body_temperature"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"error": f"Missing columns: {missing}"}

    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
    df["body_temperature"] = pd.to_numeric(df["body_temperature"], errors="coerce")
    df = df.dropna(subset=required)

    stats = {
        "rows": int(len(df)),
        "avg_heart_rate": round(float(df["heart_rate"].mean()), 1),
        "avg_temperature": round(float(df["body_temperature"].mean()), 2),
        "max_heart_rate": int(df["heart_rate"].max()),
        "min_heart_rate": int(df["heart_rate"].min()),
        "hr_trend": df["heart_rate"].tolist()[-50:],     # last 50 readings
        "temp_trend": df["body_temperature"].tolist()[-50:],
    }
    return stats

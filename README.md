# 🫀 VitalScan AI — Smart Health Monitoring & Diagnosis System

> **⚠️ DISCLAIMER:** This is a research/educational project. It is NOT a medical diagnosis tool.  
> Always consult a qualified healthcare professional for medical decisions.

---

## 📌 Overview

An end-to-end AI-powered health monitoring system that simulates a body-scanning device,
collects vital signs + symptoms, and predicts possible diseases using ML/DL models.

## 🗂️ Folder Structure


##  Quick Start

### 1. Install dependencies
```bash
pip install flask flask-cors scikit-learn pandas numpy matplotlib seaborn tensorflow joblib
```

### 2. Generate dataset
```bash
python data/generate_dataset.py
```

### 3. Train models
```bash
python models/train.py
```

### 4. Start the API server
```bash
python backend/app.py
# → Runs on http://localhost:5001
```

### 5. Open the frontend
Open `frontend/index.html` in your browser.

---

##  ML Models & Performance

| Model               | Accuracy | CV Accuracy |
|---------------------|----------|-------------|
| Random Forest       | ~99.7%   | ~99.6%      |
| Logistic Regression | ~99.5%   | ~99.6%      |
| Naive Bayes         | ~99.0%   | ~99.0%      |
| Gradient Boosting   | ~99.5%   | ~99.4%      |
| Neural Network (TF) | ~99.5%   | —           |

> Accuracy is high because dataset is synthetic and controlled.
> Real-world medical data will show lower, more realistic performance.

##  IoT Sensor Simulation

The `IoTHealthSensor` class in `utils/iot_simulator.py` simulates:
- **Heart Rate** (bpm)
- **Body Temperature** (°C)
- **SpO₂** (oxygen saturation %)
- **Blood Pressure** (systolic/diastolic mmHg)
- **Signal Quality** (0–1)

Four simulation modes: `healthy`, `fever`, `cardiac`, `hypotension`

## 🌐 API Endpoints

| Method | Endpoint              | Description                    |
|--------|-----------------------|--------------------------------|
| GET    | `/api/health`         | Liveness check                 |
| POST   | `/api/predict`        | Disease prediction             |
| GET    | `/api/iot-reading`    | Single IoT sensor snapshot     |
| GET    | `/api/iot-stream`     | SSE stream of N readings       |
| POST   | `/api/upload-csv`     | Upload historical health CSV   |
| GET    | `/api/models/info`    | Loaded model metadata          |
| GET    | `/api/symptoms`       | List of all symptom keys       |

### Predict Example
```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symptoms": ["high_fever", "cough", "shortness_of_breath", "fatigue"],
    "heart_rate": 98,
    "body_temperature": 38.9
  }'
```

### Sample Response
```json
{
  "predicted_disease": "COVID-19",
  "risk_level": "High",
  "risk_score": 0.72,
  "top_5_probabilities": [...],
  "individual_models": {...},
  "recommendations": [...],
  "disclaimer": "⚠️ This is NOT a medical diagnosis."
}
```

## 📊 Risk Calculation Logic

```
risk_score = 0.50 × symptom_score
           + 0.25 × heart_rate_deviation
           + 0.25 × temperature_deviation

Low    : score < 0.30
Medium : 0.30 ≤ score < 0.60
High   : score ≥ 0.60
```

## 🗃️ Real-World Dataset Suggestions (Kaggle)

| Dataset | Link |
|---------|------|
| Disease Symptom Description | [itachi9604/disease-symptom-description-dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) |
| Disease Prediction ML | [kaushil268/disease-prediction-using-machine-learning](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning) |
| Disease Symptoms & Patient Profile | [uom190346a/disease-symptoms-and-patient-profile-dataset](https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset) |

## ⚙️ Configuration

- **Port**: Default 5001 (change in `backend/app.py`)
- **Models Dir**: `health_monitor/models/`
- **Data Dir**: `health_monitor/data/`

---

*Built with: Python · Flask · Scikit-learn · TensorFlow · Pandas · NumPy · Matplotlib · Seaborn*

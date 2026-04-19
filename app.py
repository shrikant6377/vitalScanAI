"""
backend/app.py
==============
Flask REST API for the Smart Health Monitoring System.

Endpoints:
  POST /api/predict          - disease prediction from symptoms + vitals
  GET  /api/iot-reading      - simulated IoT sensor snapshot
  GET  /api/iot-stream       - stream of N IoT readings (SSE)
  POST /api/upload-csv       - upload historical health CSV
  GET  /api/models/info      - model metadata
  GET  /api/health           - liveness check

Run (dev):
    python backend/app.py

DISCLAIMER: This system is NOT a medical diagnosis tool.
            Always consult a qualified healthcare professional.
"""

import os
import sys
import json
import uuid
import tempfile
import traceback
from datetime import datetime

from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np

# ─── Path setup ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from preprocessor import (
    HealthPreprocessor, SYMPTOM_COLS,
    compute_risk_score, score_to_risk_label, process_uploaded_csv
)
from iot_simulator import IoTHealthSensor

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")

MODELS_DIR = os.path.join(BASE_DIR, "models")

# ─── Load artefacts at startup ────────────────────────────────────────────────
def load_models():
    models = {}
    prep = HealthPreprocessor.load(MODELS_DIR)

    model_files = {
        "Random Forest":      "random_forest.pkl",
        "Logistic Regression":"logistic_regression.pkl",
        "Naive Bayes":        "naive_bayes.pkl",
        "Gradient Boosting":  "gradient_boosting.pkl",
    }
    for name, fname in model_files.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    # Optional: TensorFlow NN
    nn_path = os.path.join(MODELS_DIR, "neural_network.keras")
    nn_model = None
    if os.path.exists(nn_path):
        try:
            import tensorflow as tf
            nn_model = tf.keras.models.load_model(nn_path)
        except Exception:
            pass

    meta = {}
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    return prep, models, nn_model, meta

try:
    PREP, SK_MODELS, NN_MODEL, META = load_models()
    CLASSES = META.get("classes", [])
    print(f"[OK] Loaded {len(SK_MODELS)} SK models + {'NN' if NN_MODEL else 'no NN'}")
    print(f"   Classes: {CLASSES}")
except Exception as e:
    print(f"[ERROR] Model load error: {e}")
    traceback.print_exc()
    PREP, SK_MODELS, NN_MODEL, META, CLASSES = None, {}, None, {}, []


# ─── Disease Recommendations ─────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Common Cold":       ["Rest and stay hydrated", "Use saline nasal spray", "OTC decongestants may help", "See a doctor if symptoms last >10 days"],
    "Influenza":         ["Rest at home", "Drink plenty of fluids", "Antiviral medication within 48 h onset", "Isolate to prevent spread"],
    "Pneumonia":         ["Seek medical attention immediately", "Antibiotic treatment as prescribed", "Rest and monitor oxygen levels", "Hospital admission may be required"],
    "Diabetes Type 2":   ["Monitor blood glucose regularly", "Follow a low-glycaemic diet", "Exercise for 30 min daily", "Consult an endocrinologist"],
    "Hypertension":      ["Reduce sodium intake (<1500 mg/day)", "Exercise regularly", "Limit alcohol and caffeine", "Take prescribed antihypertensives consistently"],
    "Gastroenteritis":   ["Stay hydrated with ORS", "Eat bland foods (BRAT diet)", "Avoid dairy and fatty foods", "See doctor if symptoms persist >3 days"],
    "Migraine":          ["Rest in a quiet dark room", "Apply cold compress to head", "OTC pain relief (ibuprofen/paracetamol)", "Track triggers in a diary"],
    "Asthma":            ["Use rescue inhaler as prescribed", "Avoid known triggers", "Monitor peak flow readings", "Have an action plan ready"],
    "Anemia":            ["Eat iron-rich foods (leafy greens, lentils)", "Vitamin C improves iron absorption", "Iron supplements if prescribed", "Check for underlying causes"],
    "COVID-19":          ["Isolate immediately", "Monitor oxygen levels (target >94%)", "Seek emergency care if breathing worsens", "Follow national health guidelines"],
}


# ─── Prediction Logic ─────────────────────────────────────────────────────────
def run_prediction(input_data: dict) -> dict:
    if PREP is None:
        return {"error": "Models not loaded"}

    X = PREP.transform_input(input_data)

    predictions = {}
    probabilities = {}

    for name, model in SK_MODELS.items():
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        predictions[name] = CLASSES[pred]
        probabilities[name] = {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(CLASSES, proba)
        }

    if NN_MODEL is not None:
        nn_proba = NN_MODEL.predict(X, verbose=0)[0]
        nn_pred = int(np.argmax(nn_proba))
        predictions["Neural Network"] = CLASSES[nn_pred]
        probabilities["Neural Network"] = {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(CLASSES, nn_proba)
        }

    # Majority vote across all models
    from collections import Counter
    vote_counts = Counter(predictions.values())
    top_disease, votes = vote_counts.most_common(1)[0]

    # Risk score
    risk_raw = compute_risk_score({
        **{s: input_data.get("symptoms", {}).get(s, 0)
            if isinstance(input_data.get("symptoms"), dict)
            else (1 if s in input_data.get("symptoms", []) else 0)
           for s in SYMPTOM_COLS},
        "heart_rate": input_data.get("heart_rate", 75),
        "body_temperature": input_data.get("body_temperature", 37.0),
    })
    risk_label = score_to_risk_label(risk_raw)

    # Top-5 probabilities (average across models)
    all_probs = list(probabilities.values())
    avg_probs = {}
    for cls in CLASSES:
        avg_probs[cls] = round(
            sum(m.get(cls, 0) for m in all_probs) / len(all_probs), 2
        )
    top5 = sorted(avg_probs.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "predicted_disease":  top_disease,
        "model_votes":        dict(vote_counts),
        "individual_models":  predictions,
        "top_5_probabilities": [{"disease": d, "probability": p} for d, p in top5],
        "risk_score":         risk_raw,
        "risk_level":         risk_label,
        "recommendations":    RECOMMENDATIONS.get(top_disease, ["Consult a healthcare professional"]),
        "disclaimer":         "⚠️ This is NOT a medical diagnosis. Always consult a qualified healthcare professional.",
        "timestamp":          datetime.utcnow().isoformat() + "Z",
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "models_loaded": len(SK_MODELS),
        "nn_loaded": NN_MODEL is not None,
        "classes": CLASSES,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Body (JSON):
    {
      "symptoms": ["fatigue", "cough", "high_fever"],
      "heart_rate": 98,
      "body_temperature": 38.9
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body"}), 400

        symptoms = data.get("symptoms", [])
        heart_rate = float(data.get("heart_rate", 75))
        body_temp  = float(data.get("body_temperature", 37.0))

        # Basic validation
        if heart_rate < 20 or heart_rate > 300:
            return jsonify({"error": "heart_rate must be between 20-300 bpm"}), 400
        if body_temp < 30 or body_temp > 45:
            return jsonify({"error": "body_temperature must be between 30-45 °C"}), 400

        result = run_prediction({
            "symptoms": symptoms,
            "heart_rate": heart_rate,
            "body_temperature": body_temp,
        })

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/iot-reading", methods=["GET"])
def iot_reading():
    """Return a single simulated sensor snapshot."""
    mode = request.args.get("mode", "random")
    sensor = IoTHealthSensor(mode=mode)
    reading = sensor.read()
    return jsonify(reading.to_dict())


@app.route("/api/iot-stream", methods=["GET"])
def iot_stream():
    """Server-Sent Events stream of IoT readings."""
    mode = request.args.get("mode", "healthy")
    n    = min(int(request.args.get("n", 20)), 100)

    def generate():
        sensor = IoTHealthSensor(mode=mode)
        for i, reading in enumerate(sensor.stream(n)):
            data = json.dumps(reading.to_dict())
            yield f"data: {data}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/upload-csv", methods=["POST"])
def upload_csv():
    """Accept a CSV upload and return trend statistics."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Only .csv files accepted"}), 400

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    f.save(tmp.name)
    try:
        stats = process_uploaded_csv(tmp.name)
    finally:
        os.unlink(tmp.name)

    if "error" in stats:
        return jsonify(stats), 400
    return jsonify(stats)


@app.route("/api/models/info", methods=["GET"])
def models_info():
    return jsonify({
        "sklearn_models": list(SK_MODELS.keys()),
        "neural_network": NN_MODEL is not None,
        "results": META.get("sklearn_results", {}),
        "nn_result": META.get("neural_network"),
        "classes": CLASSES,
        "symptoms": SYMPTOM_COLS,
    })


@app.route("/api/symptoms", methods=["GET"])
def list_symptoms():
    return jsonify({"symptoms": SYMPTOM_COLS})


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Smart Health Monitoring API")
    print("  [DISCLAIMER] NOT a medical diagnosis tool")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5001, debug=False)

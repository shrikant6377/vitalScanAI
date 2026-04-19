"""
models/train.py
===============
Trains three scikit-learn models (Random Forest, Logistic Regression,
Naive Bayes) plus a TensorFlow Neural Network on the health dataset.
Saves all models and a comparison report.

Run:
    python models/train.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

DATA_PATH   = os.path.join(BASE_DIR, "health_data.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
PLOTS_DIR   = os.path.join(BASE_DIR, "models", "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

from preprocessor import HealthPreprocessor, FEATURE_COLS, TARGET_DISEASE


# ─── Load & Preprocess ────────────────────────────────────────────────────────
def load_data():
    print("[INFO] Loading dataset ...")
    df = pd.read_csv(DATA_PATH)
    print(f"   Shape: {df.shape}  |  Diseases: {df[TARGET_DISEASE].nunique()}")

    prep = HealthPreprocessor()
    df_proc = prep.fit_transform(df)
    prep.save(MODELS_DIR)

    feature_cols = FEATURE_COLS + ["risk_score"]
    X = df_proc[feature_cols].values
    y_disease, y_risk = prep.encode_targets(df_proc)

    classes = list(prep.disease_encoder.classes_)
    print(f"   Classes: {classes}")
    return X, y_disease, y_risk, classes, prep


# ─── Scikit-learn Models ──────────────────────────────────────────────────────
def train_sklearn(X_train, X_test, y_train, y_test, classes):
    print("\n[INFO] Training scikit-learn models ...")

    sk_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, min_samples_split=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced",
            random_state=42, solver="lbfgs"
        ),
        "Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42
        ),
    }

    results = {}
    for name, model in sk_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

        results[name] = {
            "model": model,
            "accuracy": round(acc * 100, 2),
            "cv_mean":  round(cv_scores.mean() * 100, 2),
            "cv_std":   round(cv_scores.std()  * 100, 2),
            "report":   classification_report(y_test, y_pred, target_names=classes, output_dict=True),
        }

        print(f"   [OK] {name:<25s}  Acc={acc*100:.1f}%  CV={cv_scores.mean()*100:.1f}±{cv_scores.std()*100:.1f}%")

        # Save model
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{safe_name}.pkl"))

    return results


# ─── Deep Learning (TensorFlow) ───────────────────────────────────────────────
def train_neural_network(X_train, X_test, y_train, y_test, n_classes):
    print("\n[INFO] Training Neural Network (TensorFlow) ...")
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        tf.random.set_seed(42)

        # One-hot encode targets for NN
        y_train_oh = tf.keras.utils.to_categorical(y_train, n_classes)
        y_test_oh  = tf.keras.utils.to_categorical(y_test,  n_classes)

        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.35),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(64,  activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(n_classes, activation="softmax"),
        ], name="HealthNet")

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True, monitor="val_loss"),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=6, monitor="val_loss"),
        ]

        history = model.fit(
            X_train, y_train_oh,
            validation_split=0.15,
            epochs=80,
            batch_size=64,
            callbacks=callbacks,
            verbose=0,
        )

        _, acc = model.evaluate(X_test, y_test_oh, verbose=0)
        print(f"   [OK] Neural Network              Acc={acc*100:.1f}%")

        model.save(os.path.join(MODELS_DIR, "neural_network.keras"))

        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history["accuracy"],     label="Train Accuracy", color="#4CAF50")
        ax1.plot(history.history["val_accuracy"], label="Val Accuracy",   color="#FF5722")
        ax1.set_title("Model Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend()
        ax2.plot(history.history["loss"],     label="Train Loss", color="#2196F3")
        ax2.plot(history.history["val_loss"], label="Val Loss",   color="#FF9800")
        ax2.set_title("Model Loss"); ax2.set_xlabel("Epoch"); ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "nn_training_history.png"), dpi=120, bbox_inches="tight")
        plt.close()

        return {"accuracy": round(acc * 100, 2), "epochs": len(history.history["loss"])}

    except Exception as e:
        print(f"   [WARN]  Neural Network skipped: {e}")
        return None


# ─── Visualisations ───────────────────────────────────────────────────────────
def plot_results(results, nn_result, X_test, y_test, classes):
    print("\n[INFO] Generating plots ...")
    sns.set_theme(style="whitegrid", palette="muted")

    # ── 1. Model Accuracy Comparison ────────────────────────────────────────
    names = list(results.keys())
    accs  = [v["accuracy"] for v in results.values()]
    if nn_result:
        names.append("Neural Network"); accs.append(nn_result["accuracy"])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, accs, color=sns.color_palette("viridis", len(names)), edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() - 1.5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", ha="right", color="white", fontweight="bold")
    ax.set_xlabel("Accuracy (%)"); ax.set_title("Model Performance Comparison")
    ax.set_xlim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # ── 2. Confusion Matrix - best SK model ─────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # ── 3. Feature Importance (Random Forest) ───────────────────────────────
    rf = results["Random Forest"]["model"]
    feat_names = FEATURE_COLS + ["risk_score"]
    importances = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)[:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="rocket_r", ax=ax)
    ax.set_title("Top-20 Feature Importances - Random Forest")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=120, bbox_inches="tight")
    plt.close()

    print(f"   Plots saved to {PLOTS_DIR}")


# ─── Save Metadata ────────────────────────────────────────────────────────────
def save_metadata(results, nn_result, classes):
    meta = {
        "classes": classes,
        "sklearn_results": {
            k: {"accuracy": v["accuracy"], "cv_mean": v["cv_mean"], "cv_std": v["cv_std"]}
            for k, v in results.items()
        },
        "neural_network": nn_result,
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[INFO] Metadata saved -> {MODELS_DIR}/metadata.json")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Smart Health Monitoring - Model Training Pipeline")
    print("=" * 60)

    X, y_disease, y_risk, classes, prep = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_disease, test_size=0.20, stratify=y_disease, random_state=42
    )

    sk_results = train_sklearn(X_train, X_test, y_train, y_test, classes)
    nn_result  = train_neural_network(X_train, X_test, y_train, y_test, len(classes))
    plot_results(sk_results, nn_result, X_test, y_test, classes)
    save_metadata(sk_results, nn_result, classes)

    print("\n[OK] Training complete!")
    print(f"   Models saved in: {MODELS_DIR}")

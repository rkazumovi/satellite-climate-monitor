"""
classifier.py — Land use classification with scikit-learn

Loads the CNN feature vectors saved by features.py, trains several
classical scikit-learn classifiers (Random Forest, SVM, Gradient Boosting)
on the 512-dim ResNet18 embeddings, picks the best performer on the
validation set, then reports final accuracy/metrics on the held-out test
set. Saves the best model plus a confusion matrix plot to outputs/.

Run directly:
    python src/classifier.py
"""

import os
import time
import json

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

CLASS_NAMES = [
    "Annual Crop", "Forest", "Herbaceous Vegetation", "Highway",
    "Industrial Buildings", "Pasture", "Permanent Crop",
    "Residential Buildings", "River", "SeaLake",
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")


def load_split(split_name):
    features = np.load(os.path.join(FEATURES_DIR, f"{split_name}_features.npy"))
    labels = np.load(os.path.join(FEATURES_DIR, f"{split_name}_labels.npy"))
    return features, labels


def build_candidate_models():
    """
    Three classical classifiers, each with reasonable default-ish
    hyperparameters. We're comparing model families on the same fixed
    CNN features rather than doing a full hyperparameter search.
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
        ),
        "SVM_RBF": SVC(
            kernel="rbf", C=10, gamma="scale", random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, learning_rate=0.1, random_state=42
        ),
    }


def evaluate_on_validation(models, X_train, y_train, X_val, y_val, scaler):
    """Train each candidate on train set, score on validation set, return the best."""
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} on {X_train.shape[0]} training examples...")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)

        print(f"  {name} trained in {elapsed:.1f}s, validation accuracy: {val_accuracy:.4f}")
        results[name] = {"model": model, "val_accuracy": val_accuracy}

    best_name = max(results, key=lambda k: results[k]["val_accuracy"])
    print(f"\nBest model on validation set: {best_name} ({results[best_name]['val_accuracy']:.4f})")
    return best_name, results[best_name]["model"]


def evaluate_on_test(model, X_test, y_test):
    predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)

    print(f"\n--- Final test set evaluation ---")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES))

    return predictions, test_accuracy


def save_confusion_matrix(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)
    plt.title("EuroSAT Land Use Classification — Confusion Matrix (Test Set)")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved confusion matrix plot: {path}")


def save_metrics_json(best_name, test_accuracy, y_test, predictions):
    """
    Export structured classification results as JSON so reporter.py can
    generate natural-language reports from real numbers instead of guessing.
    """
    report_dict = classification_report(
        y_test, predictions, target_names=CLASS_NAMES, output_dict=True
    )

    summary = {
        "best_model": best_name,
        "test_accuracy": float(test_accuracy),
        "num_test_samples": int(len(y_test)),
        "per_class_metrics": {
            class_name: {
                "precision": report_dict[class_name]["precision"],
                "recall": report_dict[class_name]["recall"],
                "f1_score": report_dict[class_name]["f1-score"],
                "support": report_dict[class_name]["support"],
            }
            for class_name in CLASS_NAMES
        },
        "macro_avg_f1": report_dict["macro avg"]["f1-score"],
        "weighted_avg_f1": report_dict["weighted avg"]["f1-score"],
    }

    path = os.path.join(OUTPUT_DIR, "classification_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved classification summary JSON: {path}")


def save_model(model, scaler, best_name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_classifier.joblib")
    scaler_path = os.path.join(MODELS_DIR, "feature_scaler.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Saved best model ({best_name}): {model_path}")
    print(f"Saved feature scaler: {scaler_path}")


def main():
    print("Loading CNN features from features.py output...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("validation")
    X_test, y_test = load_split("test")

    print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # Scale features: SVM in particular is sensitive to feature scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    models = build_candidate_models()
    best_name, best_model = evaluate_on_validation(
        models, X_train_scaled, y_train, X_val_scaled, y_val, scaler
    )

    predictions, test_accuracy = evaluate_on_test(best_model, X_test_scaled, y_test)
    save_confusion_matrix(y_test, predictions)
    save_metrics_json(best_name, test_accuracy, y_test, predictions)
    save_model(best_model, scaler, best_name)

    print(f"\nStep 3 (classifier.py) complete. Best model: {best_name}, test accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
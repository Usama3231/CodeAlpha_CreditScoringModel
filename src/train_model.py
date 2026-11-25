from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from safe_datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 42


def load_credit_data():
    dataset = fetch_openml(name="credit-g", version=1, as_frame=True)
    features = dataset.data.copy()
    target = (dataset.target == "good").astype(int)
    return features, target, dataset


def get_feature_types(features):
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    return numeric, categorical


def make_preprocessor(features):
    numeric, categorical = get_feature_types(features)
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )


def build_models(preprocessor):
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
    }


def save_confusion_matrix(y_true, y_pred, model_name):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Bad", "Good"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Bad", "Good"])
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    fig.savefig(RESULTS_DIR / f"confusion_matrix_{slug}.png", dpi=150)
    plt.close(fig)


def save_sample_input(features):
    value = features.iloc[0]
    sample = {
        key: item.item() if hasattr(item, "item") else item
        for key, item in value.items()
    }
    with open(DATA_DIR / "sample_input.json", "w", encoding="utf-8") as file:
        json.dump(sample, file, indent=2)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    features, target, dataset = load_credit_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    preprocessor = make_preprocessor(x_train)
    models = build_models(preprocessor)
    fitted_models = {}
    metrics = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        scores = model.predict_proba(x_test)[:, 1]
        metrics[name] = calculate_metrics(y_test, predictions, scores)
        fitted_models[name] = model
        save_confusion_matrix(y_test, predictions, name)
        print(
            f"{name:20s} accuracy={metrics[name]['accuracy']:.4f} "
            f"precision={metrics[name]['precision']:.4f} "
            f"recall={metrics[name]['recall']:.4f} "
            f"f1={metrics[name]['f1_score']:.4f} "
            f"roc_auc={metrics[name]['roc_auc']:.4f}"
        )

    best_name = max(metrics, key=lambda name: metrics[name]["roc_auc"])
    joblib.dump(fitted_models[best_name], MODELS_DIR / "credit_scoring_model.joblib")
    save_sample_input(x_test)

    numeric, categorical = get_feature_types(features)
    summary = {
        "dataset": {
            "name": "Statlog German Credit (OpenML credit-g)",
            "shape": list(features.shape),
            "target": "1 = good credit, 0 = bad credit",
            "class_counts": {
                str(key): int(value)
                for key, value in dataset.target.value_counts().to_dict().items()
            },
        },
        "features": {
            "numeric": numeric,
            "categorical": categorical,
            "total": len(features.columns),
        },
        "best_model": best_name,
        "metrics": metrics,
    }

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Best model: {best_name}")
    print(f"Saved model: {MODELS_DIR / 'credit_scoring_model.joblib'}")
    print(f"Saved metrics: {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()

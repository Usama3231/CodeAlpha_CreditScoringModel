from pathlib import Path
import argparse
import json

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "credit_scoring_model.joblib"


def read_input(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return pd.DataFrame([data])


def main():
    parser = argparse.ArgumentParser(description="Predict creditworthiness from one record.")
    parser.add_argument("json_file", help="Path to a JSON file containing one credit record")
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    record = read_input(args.json_file)
    probability = float(model.predict_proba(record)[0, 1])
    prediction = int(model.predict(record)[0])
    label = "Good credit" if prediction == 1 else "Bad credit"

    print(f"Prediction: {label}")
    print(f"Good credit probability: {probability:.4f}")
    print(f"Bad credit probability: {1 - probability:.4f}")


if __name__ == "__main__":
    main()

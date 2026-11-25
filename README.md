# CodeAlpha_CreditScoringModel

Machine learning project for predicting whether an applicant is creditworthy using past financial data.

## Objective

Build a classification model that predicts creditworthiness and compare multiple algorithms using accuracy, precision, recall, F1-score, and ROC-AUC.

## Dataset

- Source: Statlog German Credit dataset from OpenML (`credit-g`)
- Rows: 1000
- Features: 20
- Target: `1 = good credit`, `0 = bad credit`
- Class balance: 700 good, 300 bad

The dataset loads automatically through `src/safe_datasets.py`, so no manual download is required.

## Project Structure

```text
CodeAlpha_CreditScoringModel/
├── src/
│   ├── safe_datasets.py
│   ├── train_model.py
│   └── predict.py
├── data/
│   └── sample_input.json
├── models/
│   └── credit_scoring_model.joblib
├── results/
│   ├── metrics.json
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_decision_tree.png
│   └── confusion_matrix_random_forest.png
├── requirements.txt
└── README.md
```

## Setup

```bat
cd CodeAlpha_CreditScoringModel
python -m pip install -r requirements.txt
```

## Train the Models

```bat
python src/train_model.py
```

This trains Logistic Regression, Decision Tree, and Random Forest, then saves the best model based on ROC-AUC.

## Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7050 | 0.7755 | 0.8143 | 0.7944 | 0.7594 |
| Decision Tree | 0.6850 | 0.7251 | 0.8857 | 0.7974 | 0.6799 |
| Random Forest | 0.7500 | 0.7848 | 0.8857 | 0.8322 | 0.7827 |

Best model: **Random Forest** with ROC-AUC **0.7827**.

## Predict One Applicant

A sample record is provided at `data/sample_input.json`.

```bat
python src/predict.py data/sample_input.json
```

Example output:

```text
Prediction: Good credit
Good credit probability: 0.8500
Bad credit probability: 0.1500
```

## Notes

Numeric columns are standardized and categorical columns are one-hot encoded inside a scikit-learn pipeline. This prevents data leakage and makes the saved model usable directly for new records.

`src/safe_datasets.py` is included for Windows systems where Smart App Control blocks one unused scikit-learn dataset helper file.

## GitHub Submission

Upload this folder as a repository named:

```text
CodeAlpha_CreditScoringModel
```

"""
train_model.py
---------------
Trains and compares churn models, picks the best by ROC-AUC / recall
trade-off, and saves the fitted pipeline + evaluation artifacts.

Model choice note:
  Tech stack originally specified XGBoost/LightGBM. This sandbox has no
  internet access to install them, so we use sklearn's
  HistGradientBoostingClassifier -- a histogram-binned gradient boosting
  implementation that is architecturally the same family as XGBoost/LightGBM
  (both are histogram-based GBDTs). We also train Logistic Regression and
  Random Forest as baselines for comparison. The explainability layer
  (Kernel SHAP, implemented in explain.py) is model-agnostic and works
  identically regardless of which of these wins.

Run: python3 train_model.py
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = "../data/telecom_churn.csv"
ARTIFACT_DIR = "../artifacts"

CATEGORICAL = ["contract_type", "payment_method", "internet_service"]
NUMERIC = [
    "tenure_months", "senior_citizen", "partner", "dependents",
    "paperless_billing", "multiple_lines", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    "monthly_charges", "total_charges", "num_support_calls_6m",
    "late_payments_last_year", "avg_monthly_usage_gb", "days_since_last_login",
    "satisfaction_score",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["customer_id", "churn"])
    y = df["churn"]
    ids = df["customer_id"]
    return X, y, ids, df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
        ]
    )


def evaluate(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    metrics = {
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }
    return metrics, proba, pred


def main():
    X, y, ids, df = load_data()
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=8,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, max_depth=6, learning_rate=0.06,
            l2_regularization=0.5, class_weight="balanced", random_state=42
        ),
    }

    results = []
    fitted = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        metrics, proba, pred = evaluate(name, pipe, X_test, y_test)
        results.append(metrics)
        fitted[name] = pipe
        print(f"\n=== {name} ===")
        print(json.dumps(metrics, indent=2))
        print(classification_report(y_test, pred, target_names=["stay", "churn"]))

    results_df = pd.DataFrame(results)
    # Business framing: missing a churner (false negative) is usually costlier
    # than one extra retention offer to a customer who would've stayed, so we
    # select on a recall-weighted composite rather than raw accuracy.
    results_df["selection_score"] = 0.5 * results_df["roc_auc"] + 0.5 * results_df["recall"]
    results_df = results_df.sort_values("selection_score", ascending=False)
    print("\n=== Model comparison (sorted by 0.5*ROC-AUC + 0.5*Recall) ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["model"]
    best_pipe = fitted[best_name]
    print(f"\nSelected best model: {best_name}")

    # Save comparison table
    results_df.to_csv(f"{ARTIFACT_DIR}/model_comparison.csv", index=False)

    # Save confusion matrix for the winner
    proba = best_pipe.predict_proba(X_test)[:, 1]
    pred = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    np.save(f"{ARTIFACT_DIR}/confusion_matrix.npy", cm)

    # Persist model + test split (needed by explain.py and the dashboard)
    with open(f"{ARTIFACT_DIR}/best_model.pkl", "wb") as f:
        pickle.dump({"pipeline": best_pipe, "model_name": best_name,
                     "numeric": NUMERIC, "categorical": CATEGORICAL}, f)

    test_export = X_test.copy()
    test_export.insert(0, "customer_id", id_test.values)
    test_export["churn_actual"] = y_test.values
    test_export["churn_proba"] = proba
    test_export["churn_pred"] = pred
    test_export.to_csv(f"{ARTIFACT_DIR}/test_predictions.csv", index=False)

    with open(f"{ARTIFACT_DIR}/best_model_metrics.json", "w") as f:
        json.dump(results_df.iloc[0].to_dict(), f, indent=2)

    print("\nSaved artifacts to", ARTIFACT_DIR)


if __name__ == "__main__":
    main()

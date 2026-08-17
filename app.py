"""
app.py
------
Interactive churn explainability dashboard.

Tech-stack note: the project spec called for Streamlit. This sandbox has
no internet access to `pip install streamlit`, so this is a Flask app
instead (Flask was already available). Functionally it covers the same
ground Streamlit would: browse customers, drill into a per-customer SHAP
explanation, and see the resulting retention action plan. If you run this
locally with internet, `pip install streamlit` and the explain.py /
recommend.py modules can be reused as-is behind a `streamlit run` UI --
only this file would need rewriting.

Run: python3 app.py
Then open http://127.0.0.1:5050
"""

import base64
import io
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, abort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from explain import KernelSHAPExplainer, make_predict_fn  # noqa: E402
from recommend import build_action_plan  # noqa: E402

BASE_DIR = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(BASE_DIR, "..", "artifacts")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "telecom_churn.csv")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model + data once at startup
# ---------------------------------------------------------------------------
with open(os.path.join(ARTIFACT_DIR, "best_model.pkl"), "rb") as f:
    ART = pickle.load(f)
PIPELINE = ART["pipeline"]
FEATURE_COLS = ART["numeric"] + ART["categorical"]
MODEL_NAME = ART["model_name"]

with open(os.path.join(ARTIFACT_DIR, "best_model_metrics.json")) as f:
    METRICS = json.load(f)

DF = pd.read_csv(DATA_PATH)
PREDICT_FN = make_predict_fn(PIPELINE)
DF["churn_proba"] = PREDICT_FN(DF[FEATURE_COLS])

RNG = np.random.default_rng(7)
BACKGROUND = DF[FEATURE_COLS].iloc[RNG.choice(len(DF), 80, replace=False)]
EXPLAINER = KernelSHAPExplainer(PREDICT_FN, BACKGROUND, n_background=40, n_samples=150)

GLOBAL_IMPORTANCE = pd.read_csv(os.path.join(ARTIFACT_DIR, "global_feature_importance.csv"))

_EXPLANATION_CACHE = {}

RISK_THRESHOLD_HIGH = 0.6
RISK_THRESHOLD_MED = 0.35


def risk_bucket(p):
    if p >= RISK_THRESHOLD_HIGH:
        return "High"
    if p >= RISK_THRESHOLD_MED:
        return "Medium"
    return "Low"


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_waterfall_b64(feature_names, phis, base_value, f_instance, customer_id, top_n=10):
    order = np.argsort(-np.abs(phis))[:top_n]
    names = [feature_names[i] for i in order][::-1]
    vals = [phis[i] for i in order][::-1]
    colors = ["#e5484d" if v > 0 else "#30a46c" for v in vals]

    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on churn probability)")
    ax.set_title(f"base rate {base_value:.2f}  ->  predicted {f_instance:.2f}")
    fig.tight_layout()
    return fig_to_base64(fig)


def make_global_importance_b64(top_n=12):
    top = GLOBAL_IMPORTANCE.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#4C6EF5")
    ax.set_xlabel("Mean |SHAP value|")
    fig.tight_layout()
    return fig_to_base64(fig)


def get_explanation(customer_id):
    if customer_id in _EXPLANATION_CACHE:
        return _EXPLANATION_CACHE[customer_id]
    row = DF[DF["customer_id"] == customer_id].iloc[0]
    instance = row[FEATURE_COLS]
    phis, f_instance = EXPLAINER.shap_values(instance)
    shap_dict = {FEATURE_COLS[i]: float(phis[i]) for i in range(len(FEATURE_COLS))}
    plan = build_action_plan(row, shap_dict, top_k=4)
    waterfall_b64 = make_waterfall_b64(FEATURE_COLS, phis, EXPLAINER.expected_value,
                                        f_instance, customer_id)
    result = {
        "row": row,
        "shap_values": shap_dict,
        "f_instance": f_instance,
        "base_value": EXPLAINER.expected_value,
        "plan": plan,
        "waterfall_b64": waterfall_b64,
    }
    _EXPLANATION_CACHE[customer_id] = result
    return result


@app.route("/")
def index():
    q = request.args.get("q", "").strip()
    risk_filter = request.args.get("risk", "")
    sort = request.args.get("sort", "risk_desc")

    view = DF.copy()
    if q:
        view = view[view["customer_id"].str.contains(q, case=False)]
    view["risk_bucket"] = view["churn_proba"].apply(risk_bucket)
    if risk_filter in ("High", "Medium", "Low"):
        view = view[view["risk_bucket"] == risk_filter]

    if sort == "risk_desc":
        view = view.sort_values("churn_proba", ascending=False)
    elif sort == "risk_asc":
        view = view.sort_values("churn_proba", ascending=True)
    elif sort == "charges_desc":
        view = view.sort_values("monthly_charges", ascending=False)

    view = view.head(200)

    kpi = {
        "total_customers": len(DF),
        "predicted_churners": int((DF["churn_proba"] >= 0.5).sum()),
        "high_risk": int((DF["churn_proba"] >= RISK_THRESHOLD_HIGH).sum()),
        "revenue_at_risk": round(
            DF.loc[DF["churn_proba"] >= RISK_THRESHOLD_HIGH, "monthly_charges"].sum(), 2
        ),
        "model_name": MODEL_NAME,
        "roc_auc": METRICS["roc_auc"],
        "recall": METRICS["recall"],
        "precision": METRICS["precision"],
    }

    customers = view[["customer_id", "churn_proba", "risk_bucket", "contract_type",
                       "tenure_months", "monthly_charges", "satisfaction_score"]].to_dict("records")

    return render_template("index.html", customers=customers, kpi=kpi,
                            q=q, risk_filter=risk_filter, sort=sort)


@app.route("/customer/<customer_id>")
def customer_detail(customer_id):
    if customer_id not in set(DF["customer_id"]):
        abort(404)
    exp = get_explanation(customer_id)
    row = exp["row"]

    drivers = sorted(exp["shap_values"].items(), key=lambda kv: -abs(kv[1]))[:8]
    driver_rows = [
        {"feature": f, "value": row[f], "shap": round(v, 4),
         "direction": "increases risk" if v > 0 else "decreases risk"}
        for f, v in drivers
    ]

    return render_template(
        "customer.html",
        customer_id=customer_id,
        row=row.to_dict(),
        proba=exp["f_instance"],
        risk=risk_bucket(exp["f_instance"]),
        base_value=exp["base_value"],
        drivers=driver_rows,
        plan=exp["plan"],
        waterfall_b64=exp["waterfall_b64"],
    )


@app.route("/overview")
def overview():
    global_b64 = make_global_importance_b64()
    comparison = pd.read_csv(os.path.join(ARTIFACT_DIR, "model_comparison.csv")).to_dict("records")
    return render_template("overview.html", metrics=METRICS, model_name=MODEL_NAME,
                            global_b64=global_b64, comparison=comparison,
                            n_features=len(FEATURE_COLS), n_rows=len(DF))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)

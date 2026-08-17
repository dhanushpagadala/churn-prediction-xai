"""
recommend.py
------------
Translates a customer's top SHAP drivers into concrete retention actions.

This is the layer most student churn projects skip entirely -- a model that
outputs "73% churn risk" is not actionable on its own. A retention team
needs to know *why*, and what lever to pull. This module maps each
risk-increasing feature to a playbook action, ranks actions by estimated
impact (SHAP magnitude x a rough cost-adjusted priority), and produces a
per-customer action plan.

The mapping below is a starting business ruleset -- in a real deployment
this table would be built with the retention/CS team and backed by A/B
test results, not just intuition. That caveat is worth saying out loud in
an interview.
"""

import json

import pandas as pd

ARTIFACT_DIR = "../artifacts"

# feature -> (condition on raw value, action, rationale, est. monthly cost to business)
PLAYBOOK = {
    "contract_type": {
        "trigger": lambda v: v == "Month-to-month",
        "action": "Offer a discounted 1-year contract upgrade (e.g. 10% off for switching)",
        "rationale": "Month-to-month customers churn at far higher rates than contracted customers.",
        "est_cost": "$",
    },
    "tenure_months": {
        "trigger": lambda v: v < 6,
        "action": "Enroll in a new-customer success program (onboarding call + 90-day check-in)",
        "rationale": "Customers in their first 6 months are in the highest-risk 'trial' window.",
        "est_cost": "$",
    },
    "monthly_charges": {
        "trigger": lambda v: v > 85,
        "action": "Proactively offer a right-sized plan review or loyalty discount",
        "rationale": "High bill relative to usage is a common, fixable churn trigger.",
        "est_cost": "$$",
    },
    "satisfaction_score": {
        "trigger": lambda v: v < 5,
        "action": "Route to a retention specialist for a personal outreach call within 48h",
        "rationale": "Low self-reported satisfaction is one of the strongest leading indicators.",
        "est_cost": "$$",
    },
    "tech_support": {
        "trigger": lambda v: v == 0,
        "action": "Offer a free 3-month trial of the Tech Support add-on",
        "rationale": "Tech support subscribers churn noticeably less -- likely a switching-cost effect.",
        "est_cost": "$",
    },
    "online_security": {
        "trigger": lambda v: v == 0,
        "action": "Bundle Online Security add-on into next renewal at no extra cost",
        "rationale": "Security add-on adoption correlates with lower churn (stickiness + trust).",
        "est_cost": "$",
    },
    "num_support_calls_6m": {
        "trigger": lambda v: v >= 3,
        "action": "Escalate to a senior support agent to resolve the recurring issue directly",
        "rationale": "Repeated support contact usually signals an unresolved pain point, not just noise.",
        "est_cost": "$$",
    },
    "late_payments_last_year": {
        "trigger": lambda v: v >= 2,
        "action": "Offer flexible billing date or autopay incentive to reduce payment friction",
        "rationale": "Payment friction is often operational, not a satisfaction problem -- cheap to fix.",
        "est_cost": "$",
    },
    "days_since_last_login": {
        "trigger": lambda v: v > 30,
        "action": "Trigger a re-engagement email/push campaign highlighting unused features",
        "rationale": "Disengaged usage is an early warning sign that often precedes cancellation.",
        "est_cost": "$",
    },
    "payment_method": {
        "trigger": lambda v: v == "Electronic check",
        "action": "Incentivize switching to autopay (bank transfer/credit card) with a small credit",
        "rationale": "Manual electronic-check payers show the highest payment-related churn risk.",
        "est_cost": "$",
    },
}


def build_action_plan(customer_row: pd.Series, shap_values: dict, top_k=3):
    """Given a customer's raw feature values and their SHAP values, return a
    ranked list of retention actions for the risk-increasing drivers only."""
    # Only consider features that push risk UP (positive SHAP) and that we
    # have a playbook entry for.
    risk_features = {f: v for f, v in shap_values.items() if v > 0 and f in PLAYBOOK}
    ranked = sorted(risk_features.items(), key=lambda kv: -kv[1])

    plan = []
    for feature, impact in ranked:
        rule = PLAYBOOK[feature]
        raw_value = customer_row[feature]
        if rule["trigger"](raw_value):
            plan.append({
                "driver": feature,
                "customer_value": raw_value if not hasattr(raw_value, "item") else raw_value.item(),
                "shap_impact": round(float(impact), 4),
                "recommended_action": rule["action"],
                "rationale": rule["rationale"],
                "est_cost": rule["est_cost"],
            })
        if len(plan) >= top_k:
            break
    return plan


def main():
    df = pd.read_csv("../data/telecom_churn.csv")
    with open(f"{ARTIFACT_DIR}/sample_explanations.json") as f:
        explanations = json.load(f)

    all_plans = {}
    for cid, exp in explanations.items():
        row = df[df["customer_id"] == cid].iloc[0]
        plan = build_action_plan(row, exp["shap_values"])
        all_plans[cid] = {
            "churn_probability": exp["churn_probability"],
            "action_plan": plan,
        }
        print(f"\n=== {cid} (churn risk: {exp['churn_probability']:.1%}) ===")
        for step in plan:
            print(f"  - [{step['est_cost']}] {step['recommended_action']}")
            print(f"    driver: {step['driver']} = {step['customer_value']} "
                  f"(SHAP impact +{step['shap_impact']})")
            print(f"    why: {step['rationale']}")

    with open(f"{ARTIFACT_DIR}/retention_action_plans.json", "w") as f:
        json.dump(all_plans, f, indent=2)
    print(f"\nSaved action plans to {ARTIFACT_DIR}/retention_action_plans.json")


if __name__ == "__main__":
    main()

"""
generate_data.py
-----------------
Generates a realistic synthetic telecom/subscription churn dataset.

Why synthetic instead of downloading the classic Kaggle Telco Churn CSV?
This environment has no internet access. Instead of faking that, we build
a generator with an explicit, documented causal structure (tenure, contract
type, support friction, pricing, engagement -> churn probability). This has
two advantages over grabbing a real CSV blind:
  1. We *know* the ground-truth drivers, so we can sanity-check that SHAP
     recovers them (a validity check most student projects skip entirely).
  2. It's trivial to swap in the real Telco churn CSV later -- just point
     train_model.py at a different file with the same column names.

Run: python3 generate_data.py
Output: ../data/telecom_churn.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000


def generate():
    customer_id = [f"CUST-{100000+i}" for i in range(N)]

    # --- Demographics / account basics ---
    tenure_months = np.clip(RNG.exponential(scale=24, size=N), 0, 72).round().astype(int)
    senior_citizen = RNG.binomial(1, 0.16, size=N)
    partner = RNG.binomial(1, 0.48, size=N)
    dependents = RNG.binomial(1, 0.30, size=N)

    # --- Plan / contract ---
    contract_type = RNG.choice(
        ["Month-to-month", "One year", "Two year"], size=N, p=[0.55, 0.25, 0.20]
    )
    payment_method = RNG.choice(
        ["Electronic check", "Mailed check", "Bank transfer (auto)", "Credit card (auto)"],
        size=N, p=[0.34, 0.19, 0.24, 0.23]
    )
    paperless_billing = RNG.binomial(1, 0.59, size=N)

    internet_service = RNG.choice(["DSL", "Fiber optic", "No"], size=N, p=[0.34, 0.44, 0.22])
    has_internet = internet_service != "No"

    online_security = np.where(has_internet, RNG.binomial(1, 0.29, size=N), 0)
    online_backup = np.where(has_internet, RNG.binomial(1, 0.34, size=N), 0)
    device_protection = np.where(has_internet, RNG.binomial(1, 0.34, size=N), 0)
    tech_support = np.where(has_internet, RNG.binomial(1, 0.29, size=N), 0)
    streaming_tv = np.where(has_internet, RNG.binomial(1, 0.38, size=N), 0)
    streaming_movies = np.where(has_internet, RNG.binomial(1, 0.39, size=N), 0)
    multiple_lines = RNG.binomial(1, 0.42, size=N)

    # --- Pricing (fiber costs more; add-ons cost more; small noise) ---
    base = np.where(internet_service == "Fiber optic", 70,
           np.where(internet_service == "DSL", 45, 20))
    addon_cost = (online_security + online_backup + device_protection +
                  tech_support + streaming_tv + streaming_movies) * 4.5
    monthly_charges = np.round(
        base + addon_cost + multiple_lines * 8 + RNG.normal(0, 6, size=N), 2
    )
    monthly_charges = np.clip(monthly_charges, 18, 130)
    total_charges = np.round(monthly_charges * np.maximum(tenure_months, 1) *
                              RNG.uniform(0.92, 1.0, size=N), 2)

    # --- Engagement / friction signals (these matter a lot for retention actions) ---
    num_support_calls = RNG.poisson(
        lam=np.clip(1.2 + (contract_type == "Month-to-month") * 1.0 +
                    (tech_support == 0) * 0.8, 0.3, None)
    )
    late_payments_last_year = RNG.poisson(lam=0.6 + (payment_method == "Electronic check") * 0.9)
    avg_monthly_usage_gb = np.round(np.clip(RNG.normal(180, 90, size=N), 5, 600), 1)
    satisfaction_score = np.clip(
        RNG.normal(
            7.2 - num_support_calls * 0.35 - late_payments_last_year * 0.25 +
            tech_support * 0.4 + online_security * 0.2,
            1.3, size=N
        ), 1, 10
    ).round(1)
    days_since_last_login = RNG.integers(0, 90, size=N)  # SaaS-style engagement proxy

    contract_risk = np.select(
        [contract_type == "Month-to-month", contract_type == "One year", contract_type == "Two year"],
        [1.0, 0.35, 0.05]
    )
    payment_risk = np.select(
        [payment_method == "Electronic check", payment_method == "Mailed check",
         payment_method == "Bank transfer (auto)", payment_method == "Credit card (auto)"],
        [0.55, 0.30, 0.05, 0.05]
    )

    # --- True churn generating process (logit) ---
    # New-customer risk: month-to-month contracts are only really risky in the
    # first year (an early-tenure x contract-type interaction) -- a pattern a
    # linear model can't represent but a tree ensemble can.
    new_customer_mtm_spike = (contract_type == "Month-to-month") * (tenure_months < 12) * 1.1
    # High support-call volume only tips people over the edge once they're
    # already unhappy (support_calls x low satisfaction interaction).
    frustration_interaction = num_support_calls * (satisfaction_score < 5) * 0.35
    # Fiber is only a churn driver when the customer has none of the "sticky"
    # add-ons (fiber x no tech-support/security interaction).
    unsupported_fiber = (internet_service == "Fiber optic") * (tech_support == 0) * (online_security == 0) * 0.6

    logit = (
        -2.75
        + 1.55 * contract_risk
        + 0.95 * payment_risk
        - 0.045 * tenure_months
        + 0.017 * monthly_charges
        + 0.10 * num_support_calls
        + 0.30 * late_payments_last_year
        - 0.16 * satisfaction_score
        - 0.55 * tech_support
        - 0.35 * online_security
        + 0.010 * days_since_last_login
        + 0.55 * senior_citizen
        - 0.30 * partner
        - 0.25 * dependents
        + 0.10 * (internet_service == "Fiber optic")
        + new_customer_mtm_spike
        + frustration_interaction
        + unsupported_fiber
        + RNG.normal(0, 0.55, size=N)  # irreducible noise
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = RNG.binomial(1, prob_churn)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "tenure_months": tenure_months,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "paperless_billing": paperless_billing,
        "internet_service": internet_service,
        "multiple_lines": multiple_lines,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_support_calls_6m": num_support_calls,
        "late_payments_last_year": late_payments_last_year,
        "avg_monthly_usage_gb": avg_monthly_usage_gb,
        "days_since_last_login": days_since_last_login,
        "satisfaction_score": satisfaction_score,
        "churn": churn,
    })
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "../data/telecom_churn.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Churn rate: {df['churn'].mean():.3%}")
    print(df.head())

"""
explain.py
----------
Explainability layer for the churn model.

No internet access in this environment means the `shap` package can't be
installed. Rather than fake it, this module implements **Kernel SHAP** from
scratch in NumPy/Pandas -- the same model-agnostic algorithm the `shap`
library's `KernelExplainer` uses (Lundberg & Lee, 2017):

  1. Take a background dataset (reference distribution for "feature absent").
  2. For a given instance, sample feature coalitions (subsets of features
     "present" vs "replaced by background").
  3. For each coalition, get the model's prediction with those features
     swapped to background values.
  4. Fit a weighted linear regression where the weights come from the
     Shapley kernel -- the regression coefficients ARE the approximate
     Shapley values (this is the key insight of Kernel SHAP: Shapley values
     are the unique solution to a specific weighted least-squares problem).

The public interface (`ChurnExplainer.shap_values(row)`,
`.expected_value`) mirrors the real `shap` library closely on purpose --
if you run this project somewhere with internet, you can swap in
`shap.KernelExplainer` or `shap.TreeExplainer` with almost no changes
to explain.py's callers.

Run standalone: python3 explain.py
  -> writes global_feature_importance.csv, sample_explanations.json,
     and a couple of matplotlib PNGs to ../artifacts/
"""

import itertools
import json
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARTIFACT_DIR = "../artifacts"
DATA_PATH = "../data/telecom_churn.csv"


class KernelSHAPExplainer:
    """Model-agnostic Kernel SHAP, implemented from scratch.

    Works on any object with a `.predict_proba(DataFrame) -> ndarray[:,1]`
    method, so it plugs directly into our sklearn Pipeline.
    """

    def __init__(self, predict_fn, background_df: pd.DataFrame, n_background=50,
                 n_samples=256, random_state=42):
        self.predict_fn = predict_fn
        self.rng = np.random.default_rng(random_state)
        if len(background_df) > n_background:
            bg_idx = self.rng.choice(len(background_df), n_background, replace=False)
            self.background = background_df.iloc[bg_idx].reset_index(drop=True)
        else:
            self.background = background_df.reset_index(drop=True)
        self.feature_names = list(background_df.columns)
        self.M = len(self.feature_names)
        self.n_samples = n_samples
        # E[f(X)] over the background -- SHAP's "base value"
        self.expected_value = float(np.mean(self.predict_fn(self.background)))

    def _shapley_kernel_weights(self, coalition_sizes):
        M = self.M
        weights = np.zeros(len(coalition_sizes))
        for i, s in enumerate(coalition_sizes):
            if s == 0 or s == M:
                weights[i] = 1e6  # enforce these via huge weight (edge constraints)
            else:
                from math import comb
                weights[i] = (M - 1) / (comb(M, s) * s * (M - s))
        return weights

    def shap_values(self, instance: pd.Series) -> np.ndarray:
        """Return approximate Shapley values, one per feature, for `instance`."""
        M = self.M
        instance_df = pd.DataFrame([instance.values], columns=self.feature_names)

        # Sample binary coalition masks (1 = keep instance's value, 0 = use background)
        max_full = 2 ** M
        masks = []
        seen = set()
        # Always include the empty and full coalitions (anchors expected value & prediction)
        masks.append(tuple([0] * M))
        masks.append(tuple([1] * M))
        seen.update(masks)

        n_target = min(self.n_samples, max(max_full - 2, 0))
        attempts = 0
        while len(masks) < n_target + 2 and attempts < n_target * 20 + 200:
            attempts += 1
            size = self.rng.integers(1, M)  # exclude 0 and M, already added
            idx = self.rng.choice(M, size=size, replace=False)
            m = [0] * M
            for j in idx:
                m[j] = 1
            t = tuple(m)
            if t not in seen:
                seen.add(t)
                masks.append(t)

        masks = np.array(masks)
        coalition_sizes = masks.sum(axis=1)
        weights = self._shapley_kernel_weights(coalition_sizes)

        # For each mask, build a synthetic dataset: background rows with the
        # "kept" features overwritten by the instance's values, then average
        # the model's predicted probability over all background rows.
        y = np.zeros(len(masks))
        for i, mask in enumerate(masks):
            synth = self.background.copy()
            keep_cols = [self.feature_names[j] for j in range(M) if mask[j] == 1]
            for col in keep_cols:
                synth[col] = instance[col]
            y[i] = np.mean(self.predict_fn(synth))

        # Weighted least squares: y ~ phi0 + sum(mask_j * phi_j)
        X = masks.astype(float)
        W = np.diag(weights)
        X_design = np.hstack([np.ones((len(X), 1)), X])
        WX = W @ X_design
        try:
            beta = np.linalg.solve(X_design.T @ WX, X_design.T @ W @ y)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

        phi0 = beta[0]
        phis = beta[1:]
        # Efficiency correction: force sum(phi) + phi0 == f(instance) exactly
        f_instance = float(np.mean(self.predict_fn(
            pd.DataFrame([instance.values] * len(self.background), columns=self.feature_names)
        )))
        gap = (f_instance - phi0) - phis.sum()
        phis = phis + gap / M
        return phis, f_instance


def make_predict_fn(pipeline):
    def predict_fn(df):
        return pipeline.predict_proba(df)[:, 1]
    return predict_fn


def load_artifacts():
    with open(f"{ARTIFACT_DIR}/best_model.pkl", "rb") as f:
        art = pickle.load(f)
    pipeline = art["pipeline"]
    numeric, categorical = art["numeric"], art["categorical"]
    feature_cols = numeric + categorical
    df = pd.read_csv(DATA_PATH)
    return pipeline, feature_cols, df


def build_global_importance(explainer, X_sample, feature_names, out_path):
    """Average |SHAP value| per feature across a sample of customers."""
    all_phis = []
    for _, row in X_sample.iterrows():
        phis, _ = explainer.shap_values(row)
        all_phis.append(phis)
    all_phis = np.array(all_phis)
    mean_abs = np.abs(all_phis).mean(axis=0)
    imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.to_csv(out_path, index=False)
    return imp, all_phis


def plot_global_importance(imp_df, out_path, top_n=15):
    top = imp_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["mean_abs_shap"], color="#4C6EF5")
    plt.xlabel("Mean |SHAP value| (avg impact on churn probability)")
    plt.title("Global Feature Importance (Kernel SHAP)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_waterfall(feature_names, phis, base_value, f_instance, customer_id, out_path, top_n=10):
    order = np.argsort(-np.abs(phis))[:top_n]
    names = [feature_names[i] for i in order][::-1]
    vals = [phis[i] for i in order][::-1]
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in vals]

    plt.figure(figsize=(8, 5.5))
    plt.barh(names, vals, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("SHAP value (impact on churn probability)")
    plt.title(f"Why {customer_id} is predicted to churn\n"
              f"base rate={base_value:.2f} -> predicted={f_instance:.2f}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    pipeline, feature_cols, df = load_artifacts()
    predict_fn = make_predict_fn(pipeline)

    X = df[feature_cols]
    rng = np.random.default_rng(7)
    background = X.iloc[rng.choice(len(X), 80, replace=False)]

    explainer = KernelSHAPExplainer(predict_fn, background, n_background=50, n_samples=200)
    print(f"Base rate (expected value): {explainer.expected_value:.4f}")

    # Global importance over a sample of 60 customers (Kernel SHAP is
    # sampling-based and O(n_samples) predict calls per customer, so we
    # keep this modest for a from-scratch NumPy implementation).
    sample_idx = rng.choice(len(X), 60, replace=False)
    X_sample = X.iloc[sample_idx].reset_index(drop=True)
    imp_df, all_phis = build_global_importance(
        explainer, X_sample, feature_cols, f"{ARTIFACT_DIR}/global_feature_importance.csv"
    )
    plot_global_importance(imp_df, f"{ARTIFACT_DIR}/global_feature_importance.png")
    print("\nTop 10 global drivers of churn:")
    print(imp_df.head(10).to_string(index=False))

    # Per-customer explanations for a handful of highest-risk customers
    proba = predict_fn(X)
    df_scored = df.copy()
    df_scored["churn_proba"] = proba
    top_risk = df_scored.sort_values("churn_proba", ascending=False).head(5)

    explanations = {}
    for _, row in top_risk.iterrows():
        cid = row["customer_id"]
        instance = row[feature_cols]
        phis, f_instance = explainer.shap_values(instance)
        plot_waterfall(
            feature_cols, phis, explainer.expected_value, f_instance, cid,
            f"{ARTIFACT_DIR}/waterfall_{cid}.png"
        )
        explanations[cid] = {
            "churn_probability": round(float(f_instance), 4),
            "base_rate": round(explainer.expected_value, 4),
            "shap_values": {feature_cols[i]: round(float(phis[i]), 4) for i in range(len(feature_cols))},
        }
        print(f"Saved waterfall for {cid} (p={f_instance:.2%})")

    with open(f"{ARTIFACT_DIR}/sample_explanations.json", "w") as f:
        json.dump(explanations, f, indent=2)

    print("\nDone. Artifacts written to", ARTIFACT_DIR)


if __name__ == "__main__":
    main()

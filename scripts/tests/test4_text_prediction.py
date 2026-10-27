
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

EXPLORATION_PATH = Path("data/splits/exploration.csv.gz")
CONFIRMATION_PATH = Path("data/splits/confirmation.csv.gz")
REPORT_PATH = Path("reports/test4_text_prediction.json")

SEED = 42
ALPHA = 0.05
PRACTICAL_MAE_REDUCTION = 0.01
N_BOOTSTRAPS = 2000

STRUCTURED_FEATURES = [
    "Score",
    "ReviewLengthChars",
    "SummaryLengthChars",
    "ReviewYear",
]


def load_eligible(path):
    df = pd.read_csv(path, compression="gzip")

    for col in [
        "Score",
        "ReviewLengthChars",
        "SummaryLengthChars",
        "HelpfulnessNumerator",
        "HelpfulnessDenominator",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ReviewDateUTC"] = pd.to_datetime(
        df["ReviewDateUTC"], errors="coerce", utc=True
    )
    df["ReviewYear"] = df["ReviewDateUTC"].dt.year

    num = df["HelpfulnessNumerator"]
    den = df["HelpfulnessDenominator"]

    valid = (
        num.notna()
        & den.notna()
        & num.ge(0)
        & den.gt(0)
        & num.le(den)
    )

    required = STRUCTURED_FEATURES + [
        "ProductId",
        "Summary",
        "Text",
    ]
    df = df.loc[valid].dropna(subset=required).copy()

    df["TargetHelpfulnessRate"] = (
        df["HelpfulnessNumerator"]
        / df["HelpfulnessDenominator"]
    )

    df["ReviewText"] = (
        df["Summary"].fillna("").astype(str)
        + " "
        + df["Text"].fillna("").astype(str)
    )

    return df.reset_index(drop=True)


def structured_matrix(train_df, test_df):
    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(
        train_df[STRUCTURED_FEATURES]
    )
    X_test = imputer.transform(
        test_df[STRUCTURED_FEATURES]
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return csr_matrix(X_train), csr_matrix(X_test)


def product_bootstrap(delta, product_ids, n_bootstraps, seed):
    """
    Resample products, retaining all evaluated reviews belonging
    to each sampled product. The estimand is the review-weighted
    mean paired MAE reduction.
    """
    rng = np.random.default_rng(seed)

    product_ids = np.asarray(product_ids)
    unique_products = np.unique(product_ids)

    positions = {
        product: np.flatnonzero(product_ids == product)
        for product in unique_products
    }

    observed = float(np.mean(delta))
    bootstrap_estimates = np.empty(n_bootstraps)

    for b in range(n_bootstraps):
        sampled_products = rng.choice(
            unique_products,
            size=len(unique_products),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [positions[p] for p in sampled_products]
        )
        bootstrap_estimates[b] = np.mean(delta[sampled_indices])

    ci_low, ci_high = np.quantile(
        bootstrap_estimates, [0.025, 0.975]
    )

    # One-sided bootstrap-tail estimate for H1: mean improvement > 0.
    p_value = (
        1 + np.sum(bootstrap_estimates <= 0)
    ) / (n_bootstraps + 1)

    return {
        "mean_mae_reduction": observed,
        "ci_95": [float(ci_low), float(ci_high)],
        "one_sided_bootstrap_p_value": float(p_value),
        "bootstrap_replicates": int(n_bootstraps),
        "unique_products": int(len(unique_products)),
        "seed": int(seed),
        "method": (
            "Product-cluster bootstrap; resample products with "
            "replacement and retain all evaluated reviews per draw."
        ),
    }


def main():
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_df = load_eligible(EXPLORATION_PATH)
    test_df = load_eligible(CONFIRMATION_PATH)

    if train_df.empty or test_df.empty:
        raise ValueError("No eligible training or confirmation rows.")

    # Fit TF-IDF on exploration data only.
    vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=5,
        ngram_range=(1, 2),
        stop_words="english",
        dtype=np.float32,
    )

    X_text_train = vectorizer.fit_transform(
        train_df["ReviewText"]
    )
    X_text_test = vectorizer.transform(
        test_df["ReviewText"]
    )

    X_struct_train, X_struct_test = structured_matrix(
        train_df, test_df
    )

    X_combined_train = hstack(
        [X_struct_train, X_text_train], format="csr"
    )
    X_combined_test = hstack(
        [X_struct_test, X_text_test], format="csr"
    )

    y_train = train_df["TargetHelpfulnessRate"].to_numpy()
    y_test = test_df["TargetHelpfulnessRate"].to_numpy()

    # Identical structured model in both comparisons.
    baseline = Ridge(alpha=1.0)
    baseline.fit(X_struct_train, y_train)

    text_model = Ridge(alpha=1.0)
    text_model.fit(X_combined_train, y_train)

    pred_baseline = np.clip(
        baseline.predict(X_struct_test), 0, 1
    )
    pred_text = np.clip(
        text_model.predict(X_combined_test), 0, 1
    )

    abs_error_baseline = np.abs(y_test - pred_baseline)
    abs_error_text = np.abs(y_test - pred_text)

    # Positive means adding text reduced absolute prediction error.
    paired_reduction = abs_error_baseline - abs_error_text

    bootstrap = product_bootstrap(
        paired_reduction,
        test_df["ProductId"].to_numpy(),
        N_BOOTSTRAPS,
        SEED,
    )

    mae_baseline = float(np.mean(abs_error_baseline))
    mae_text = float(np.mean(abs_error_text))
    reduction = mae_baseline - mae_text

    ci_low, ci_high = bootstrap["ci_95"]

    report = {
        "analysis": "Test 4: TF-IDF text prediction comparison",
        "exploration_path": str(EXPLORATION_PATH),
        "confirmation_path": str(CONFIRMATION_PATH),
        "target": (
            "HelpfulnessNumerator / HelpfulnessDenominator "
            "for valid positive-denominator vote records"
        ),
        "structured_features": STRUCTURED_FEATURES,
        "baseline_model": "Ridge regression, alpha=1.0",
        "text_model": (
            "Ridge regression, alpha=1.0, structured features "
            "plus TF-IDF unigrams and bigrams"
        ),
        "tfidf_fit_on_exploration_only": True,
        "max_tfidf_features": int(X_text_train.shape[1]),
        "alpha": ALPHA,
        "practical_mae_reduction_threshold": (
            PRACTICAL_MAE_REDUCTION
        ),
        "n_exploration_reviews": int(len(train_df)),
        "n_confirmation_reviews": int(len(test_df)),
        "n_confirmation_products": int(
            test_df["ProductId"].nunique()
        ),
        "baseline_mae": mae_baseline,
        "text_model_mae": mae_text,
        "mae_reduction_baseline_minus_text": reduction,
        "relative_mae_reduction": (
            reduction / mae_baseline if mae_baseline else None
        ),
        "bootstrap": bootstrap,
        "reject_h0_at_0_05": bool(
            bootstrap["one_sided_bootstrap_p_value"] < ALPHA
        ),
        "practical_threshold_reached_by_point_estimate": bool(
            reduction >= PRACTICAL_MAE_REDUCTION
        ),
        "ci_supports_positive_improvement": bool(ci_low > 0),
        "ci_supports_improvement_of_at_least_0_01": bool(
            ci_low >= PRACTICAL_MAE_REDUCTION
        ),
        "interpretation_note": (
            "Positive MAE reduction favors the text-enhanced model. "
            "The bootstrap p-value is an empirical one-sided tail "
            "estimate, not an exact test. The confidence interval "
            "describes uncertainty in the review-weighted mean paired "
            "MAE reduction under product-level resampling. The target "
            "is a noisy review-level helpfulness proportion, and the "
            "analysis is predictive rather than causal."
        ),
    }

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print("=" * 68)
    print("TEST 4: TEXT-ENHANCED PREDICTION")
    print("=" * 68)
    print(json.dumps(report, indent=2))
    print(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

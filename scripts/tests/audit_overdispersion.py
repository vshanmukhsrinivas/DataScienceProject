
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

TESTS = {
    "test1_rating_extremity": {
        "term": "RatingExtremity",
        "label": "Test 1: rating extremity",
    },
    "test2_review_length": {
        "term": "LogReviewLength_c",
        "label": "Test 2: review length",
    },
}

results = []

for filename, config in TESTS.items():
    path = REPORTS / f"{filename}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Required report not found: {path}"
        )

    with path.open(encoding="utf-8") as f:
        report = json.load(f)

    beta = float(report["coefficient"])
    se = float(report["cluster_robust_standard_error"])
    z_stat = beta / se

    # Recover the Pearson dispersion estimate from Test 1,
    # where it was explicitly calculated.
    if filename == "test1_rating_extremity":
        dispersion = float(report["pearson_dispersion_ratio"])
    else:
        dispersion = None

    results.append({
        "test": config["label"],
        "coefficient": beta,
        "cluster_robust_se": se,
        "z_statistic": z_stat,
        "one_sided_p": float(norm.sf(z_stat)),
        "dispersion_ratio_from_test1": dispersion,
    })

# This is a diagnostic only. It applies the same estimated
# dispersion factor to the two reported standard errors.
# It is NOT a substitute for a correctly specified variance model.
dispersion = results[0]["dispersion_ratio_from_test1"]

if dispersion is None or dispersion <= 0:
    raise ValueError("A positive dispersion estimate is required.")

inflation_factor = float(np.sqrt(dispersion))

for item in results:
    adjusted_se = item["cluster_robust_se"] * inflation_factor
    adjusted_z = item["coefficient"] / adjusted_se

    item["diagnostic_inflation_factor"] = inflation_factor
    item["diagnostic_inflated_se"] = adjusted_se
    item["diagnostic_z_statistic"] = adjusted_z
    item["diagnostic_one_sided_p"] = float(norm.sf(adjusted_z))

output = {
    "purpose": (
        "Diagnostic sensitivity check only; not a replacement "
        "for product-clustered primary inference."
    ),
    "dispersion_ratio": dispersion,
    "standard_error_inflation_factor": inflation_factor,
    "results": results,
    "warning": (
        "Multiplying already cluster-robust standard errors by "
        "sqrt(Pearson dispersion) may double-count variability. "
        "Interpret these values as a stress test, not corrected "
        "p-values or definitive inference."
    ),
}

out_path = REPORTS / "overdispersion_diagnostic.json"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, allow_nan=False)

print(json.dumps(output, indent=2))
print(f"\nSaved diagnostic to: {out_path}")

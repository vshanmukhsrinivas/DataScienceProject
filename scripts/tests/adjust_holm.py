
import json
from pathlib import Path

from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"

# Use product-clustered inference for all three tests.
specs = [
    {
        "id": "test1",
        "label": "Test 1: rating extremity",
        "file": "test1_rating_extremity.json",
        "p_path": ("p_value_one_sided",),
    },
    {
        "id": "test2",
        "label": "Test 2: review length",
        "file": "test2_review_length.json",
        "p_path": ("p_value_one_sided",),
    },
    {
        "id": "test3",
        "label": "Test 3: summary length",
        "file": "test3_summary_length.json",
        "p_path": ("robust_sensitivity", "p_value"),
    },
]

results = []

for spec in specs:
    path = REPORTS / spec["file"]

    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")

    with path.open(encoding="utf-8") as f:
        report = json.load(f)

    value = report
    for key in spec["p_path"]:
        value = value[key]

    p_value = float(value)

    if not 0 <= p_value <= 1:
        raise ValueError(
            f"Invalid p-value for {spec['label']}: {p_value}"
        )

    results.append({
        "id": spec["id"],
        "label": spec["label"],
        "raw_p_value": p_value,
        "source": spec["file"],
        "inference": (
            "Product-clustered asymptotic z-test, one-sided"
            if spec["id"] in ("test1", "test2")
            else "Product-clustered asymptotic Wald chi-square test, two-sided"
        ),
    })

raw_p = [item["raw_p_value"] for item in results]

reject, adjusted_p, _, _ = multipletests(
    raw_p,
    alpha=0.05,
    method="holm",
)

for item, adjusted, rejected in zip(
    results, adjusted_p, reject
):
    item["holm_adjusted_p_value"] = float(adjusted)
    item["familywise_alpha"] = 0.05
    item["decision_after_holm"] = (
        "Reject H0" if rejected else "Fail to reject H0"
    )

output = {
    "analysis": "Holm correction for explanatory Tests 1-3",
    "method": "Holm step-down procedure",
    "familywise_alpha": 0.05,
    "number_of_tests": len(results),
    "p_value_policy": (
        "Product-clustered inference p-values: one-sided for Tests 1-2 "
        "and two-sided for Test 3."
    ),
    "results": results,
    "caution": (
        "The adjustment controls family-wise error under arbitrary "
        "dependence when the input p-values are valid. Product clustering "
        "and asymptotic inference assumptions still require consideration."
    ),
}

out_path = REPORTS / "holm_adjustment.json"
with out_path.open("w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, allow_nan=False)

print(json.dumps(output, indent=2))
print(f"\nSaved report: {out_path}")

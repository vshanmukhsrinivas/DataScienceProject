
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

DATA_PATH = "data/splits/exploration.csv.gz"
OUTPUT_DIR = Path("data/processed")
REPORT_PATH = Path("reports/feature_scale.json")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 1. Load exploration data only.
df = pd.read_csv(DATA_PATH, compression="gzip")

# 2. Combine review summary and review body.
texts = (
    df["Summary"].fillna("").astype(str)
    + " "
    + df["Text"].fillna("").astype(str)
)

# 3. Learn TF-IDF vocabulary and IDF weights from exploration data.
vectorizer = TfidfVectorizer(
    max_features=20000,
    min_df=5,
    ngram_range=(1, 2),
    stop_words="english",
    dtype=np.float32,
)

X_text = vectorizer.fit_transform(texts)

# 4. Build structured features.
# Rating extremity is included for the feature-count demonstration.
dates = pd.to_datetime(df["ReviewDateUTC"], errors="coerce")

structured = pd.DataFrame({
    "Score": df["Score"],
    "RatingExtremity": (df["Score"] - 3).abs(),
    "ReviewLengthChars": df["ReviewLengthChars"],
    "SummaryLengthChars": df["SummaryLengthChars"],
    "ReviewYear": dates.dt.year,
})

structured = structured.fillna(structured.median())
X_structured = csr_matrix(
    structured.to_numpy(dtype=np.float32)
)

# 5. Combine structured and TF-IDF features into a sparse matrix.
X = hstack([X_structured, X_text], format="csr")

# 6. Save the matrix and fitted vectorizer for reuse.
save_npz(OUTPUT_DIR / "exploration_features.npz", X)
joblib.dump(vectorizer, OUTPUT_DIR / "tfidf_vectorizer.joblib")

# Save the feature names so we can interpret columns later.
feature_names = (
    list(structured.columns)
    + [f"tfidf:{term}" for term in vectorizer.get_feature_names_out()]
)

with (OUTPUT_DIR / "feature_names.json").open("w", encoding="utf-8") as f:
    json.dump(feature_names, f, ensure_ascii=False)

# 7. Report the feature-count arithmetic.
n_samples, n_features = X.shape
total_entries = int(n_samples) * int(n_features)

report = {
    "dataset": "exploration",
    "n_samples": int(n_samples),
    "structured_feature_count": int(X_structured.shape[1]),
    "tfidf_feature_count": int(X_text.shape[1]),
    "total_feature_count": int(n_features),
    "instances_times_attributes": total_entries,
    "three_million_requirement_met": total_entries >= 3_000_000,
    "stored_nonzero_entries": int(X.nnz),
    "matrix_density": float(X.nnz / total_entries),
    "vectorizer_settings": {
        "max_features": 20000,
        "min_df": 5,
        "ngram_range": [1, 2],
        "stop_words": "english",
    },
}

with REPORT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("FEATURE SCALE REPORT")
print(json.dumps(report, indent=2))
print(f"\nSaved matrix: {OUTPUT_DIR / 'exploration_features.npz'}")
print(f"Saved vectorizer: {OUTPUT_DIR / 'tfidf_vectorizer.joblib'}")
print(f"Saved feature names: {OUTPUT_DIR / 'feature_names.json'}")
print(f"Saved report: {REPORT_PATH}")

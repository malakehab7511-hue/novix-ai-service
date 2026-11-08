import numpy as np
import pandas as pd
from .config import RAW_DATA, PROCESSED_DATA, FEATURES, TARGET, GROUP_ENV, GROUP_SESSION, ALL_LABELS

REQUIRED = FEATURES + [TARGET, GROUP_ENV, GROUP_SESSION, "elapsed_s"]

def load_and_clean(input_path=RAW_DATA, output_path=PROCESSED_DATA):
    df = pd.read_csv(input_path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.drop_duplicates(subset=["row_id"] if "row_id" in df.columns else None).copy()
    for col in FEATURES + ["elapsed_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=REQUIRED).copy()
    df[TARGET] = df[TARGET].astype(str).str.strip().str.upper()

    unknown = sorted(set(df[TARGET]) - set(ALL_LABELS))
    if unknown:
        raise ValueError(f"Unknown labels found: {unknown}")

    # Preserve temporal order; elapsed_s is the reliable ordering field in this CSV.
    df = df.sort_values([GROUP_ENV, GROUP_SESSION, "elapsed_s"]).reset_index(drop=True)

    # A baseline should be constant within a session in this dataset.
    for c in ["baseline_mq2", "baseline_mq6"]:
        if c in df.columns:
            n_unique = df.groupby(GROUP_SESSION)[c].nunique(dropna=True)
            if (n_unique > 1).any():
                print(f"WARNING: {c} changes within sessions: {n_unique[n_unique > 1].to_dict()}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    gaps = df.groupby(GROUP_SESSION)["elapsed_s"].diff().dropna()
    report = {
        "rows_before": int(before),
        "rows_after": int(len(df)),
        "removed_rows": int(before - len(df)),
        "feature_count": len(FEATURES),
        "labels": df[TARGET].value_counts().to_dict(),
        "environments": df[GROUP_ENV].value_counts().to_dict(),
        "sessions": df.groupby(GROUP_SESSION).size().to_dict(),
        "median_interval_seconds": float(gaps.median()) if len(gaps) else None,
        "max_interval_seconds": float(gaps.max()) if len(gaps) else None,
        "missing_feature_values_after_cleaning": int(df[FEATURES].isna().sum().sum()),
    }
    return df, report

if __name__ == "__main__":
    data, audit = load_and_clean()
    print(audit)

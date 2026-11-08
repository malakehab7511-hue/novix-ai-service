import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, recall_score,
    balanced_accuracy_score, precision_score
)
from .config import FEATURES, TARGET, GROUP_ENV, GROUP_SESSION, SAFE_LABELS, ALL_LABELS, N_CONSECUTIVE, MAX_GAP_SECONDS
from .models import make_binary_model, make_multiclass_model, LIGHTGBM_CANDIDATES


def _positive_probability(model, X):
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return model.predict_proba(X)[:, classes.index(1)]


def _session_groups(df):
    # Session names alone are not guaranteed globally unique.
    return (df[GROUP_ENV].astype(str) + "::" + df[GROUP_SESSION].astype(str)).to_numpy()


def consecutive_alarm(probabilities, groups, elapsed, threshold, n=N_CONSECUTIVE, max_gap=MAX_GAP_SECONDS):
    """Alarm becomes active on the nth consecutive qualifying observation."""
    out = np.zeros(len(probabilities), dtype=bool)
    temp = pd.DataFrame({"p": probabilities, "g": np.asarray(groups), "t": np.asarray(elapsed)})
    for _, idx in temp.groupby("g", sort=False).groups.items():
        idx = list(idx)
        run = 0
        previous_t = None
        for i in idx:
            now = float(temp.at[i, "t"])
            if previous_t is not None and (now - previous_t > max_gap or now <= previous_t):
                run = 0
            run = run + 1 if float(temp.at[i, "p"]) >= threshold else 0
            out[i] = run >= n
            previous_t = now
    return out


def _threshold_metrics(y_true, probabilities, groups, elapsed, threshold):
    alarm = consecutive_alarm(probabilities, groups, elapsed, float(threshold))
    y_true = np.asarray(y_true, dtype=int)
    recall = recall_score(y_true, alarm, zero_division=0)
    specificity = float(np.mean(~alarm[y_true == 0])) if np.any(y_true == 0) else 0.0
    balanced = 0.5 * (recall + specificity)
    precision = precision_score(y_true, alarm, zero_division=0)
    f1 = f1_score(y_true, alarm, zero_division=0)
    far = float(np.mean(alarm[y_true == 0])) if np.any(y_true == 0) else np.nan
    return {
        "threshold": float(threshold), "balanced_accuracy": float(balanced),
        "hazard_recall": float(recall), "specificity": float(specificity),
        "precision": float(precision), "f1": float(f1),
        "safe_false_alarm_fraction": far,
    }


def choose_balanced_threshold(y_true, probabilities, groups, elapsed):
    """Choose threshold by balanced accuracy after the 10-reading confirmation rule."""
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.r_[0.01, np.quantile(probabilities, np.linspace(0.02, 0.98, 121)), 0.99])
    rows = [_threshold_metrics(y_true, probabilities, groups, elapsed, th) for th in candidates]
    # Balanced accuracy is the primary objective; F1 then specificity break ties.
    best = max(rows, key=lambda r: (r["balanced_accuracy"], r["f1"], r["specificity"], r["threshold"]))
    return best


def _make_temporal_block_ids(groups, elapsed, n_blocks=6):
    """Create contiguous validation blocks within each session for purged OOF."""
    groups = np.asarray(groups)
    elapsed = np.asarray(elapsed, dtype=float)
    block_ids = np.empty(len(groups), dtype=object)
    for group in pd.unique(groups):
        idx = np.where(groups == group)[0]
        idx = idx[np.argsort(elapsed[idx])]
        chunks = [chunk for chunk in np.array_split(idx, min(n_blocks, len(idx))) if len(chunk)]
        for block_no, chunk in enumerate(chunks):
            block_ids[chunk] = f"{group}::BLOCK_{block_no}"
    return block_ids


def _purged_block_splits(block_ids, groups, elapsed, n_splits=5, purge_seconds=30.0):
    """Group folds over contiguous blocks; purge nearby same-session rows from training."""
    block_ids = np.asarray(block_ids)
    groups = np.asarray(groups)
    elapsed = np.asarray(elapsed, dtype=float)
    unique_blocks = pd.Series(block_ids).nunique()
    splitter = GroupKFold(n_splits=min(n_splits, unique_blocks))
    dummy_x = np.zeros((len(block_ids), 1))
    dummy_y = np.zeros(len(block_ids))
    for train_idx, valid_idx in splitter.split(dummy_x, dummy_y, block_ids):
        keep = np.ones(len(train_idx), dtype=bool)
        train_groups = groups[train_idx]
        train_times = elapsed[train_idx]
        for block in pd.unique(block_ids[valid_idx]):
            v_idx = valid_idx[block_ids[valid_idx] == block]
            session = groups[v_idx[0]]
            lo = float(np.min(elapsed[v_idx])) - purge_seconds
            hi = float(np.max(elapsed[v_idx])) + purge_seconds
            same_session = train_groups == session
            keep &= ~(same_session & (train_times >= lo) & (train_times <= hi))
        yield train_idx[keep], valid_idx


def _oof_probabilities(X, y_binary, groups, elapsed, model_name, params=None):
    """Purged temporal-block OOF predictions; avoids random-row leakage."""
    block_ids = _make_temporal_block_ids(groups, elapsed)
    oof = np.full(len(y_binary), np.nan, dtype=float)
    for tr, va in _purged_block_splits(block_ids, groups, elapsed):
        if len(tr) == 0 or len(np.unique(y_binary.iloc[tr])) < 2:
            continue
        model = make_binary_model(model_name, params=params)
        model.fit(X.iloc[tr], y_binary.iloc[tr])
        oof[va] = _positive_probability(model, X.iloc[va])
    return oof, block_ids

def tune_threshold_oof(X, y_binary, groups, elapsed, model_name):
    """Session-grouped OOF threshold selection using training data only."""
    oof, block_ids = _oof_probabilities(X, y_binary, groups, elapsed, model_name)
    valid = np.isfinite(oof)
    if valid.sum() == 0 or y_binary.iloc[np.where(valid)[0]].nunique() < 2:
        return 0.5, {"note": "Insufficient grouped OOF class coverage; fallback threshold 0.5"}
    best = choose_balanced_threshold(
        y_binary.iloc[np.where(valid)[0]].to_numpy(), oof[valid],
        block_ids[valid], np.asarray(elapsed)[valid]
    )
    best["note"] = "Selected by maximum balanced accuracy on purged temporal-block OOF predictions."
    return best["threshold"], best


def select_lightgbm_params(X, y_binary, groups, elapsed):
    """Select a modest LightGBM configuration by grouped OOF balanced accuracy."""
    records = []
    for candidate_id, params in enumerate(LIGHTGBM_CANDIDATES, start=1):
        oof, block_ids = _oof_probabilities(X, y_binary, groups, elapsed, "LightGBM", params=params)
        valid = np.isfinite(oof)
        if valid.sum() == 0 or y_binary.iloc[np.where(valid)[0]].nunique() < 2:
            continue
        score = choose_balanced_threshold(
            y_binary.iloc[np.where(valid)[0]].to_numpy(), oof[valid],
            block_ids[valid], np.asarray(elapsed)[valid]
        )
        records.append({"candidate_id": candidate_id, "params": params, **score})
    if not records:
        return {}, 0.5, {"note": "No valid grouped OOF result; using default LightGBM parameters and threshold 0.5"}
    best = max(records, key=lambda r: (r["balanced_accuracy"], r["f1"], r["specificity"], r["threshold"]))
    info = {k: v for k, v in best.items() if k != "params"}
    info["note"] = "Selected LightGBM configuration and threshold by purged temporal-block OOF balanced accuracy."
    return best["params"], best["threshold"], info


def false_alarms_per_hour(alarm, y_true, elapsed, groups, max_gap=MAX_GAP_SECONDS):
    """Count false-alarm rising edges per observed safe hour, using actual safe intervals."""
    frame = pd.DataFrame({"alarm": np.asarray(alarm, dtype=bool), "y": np.asarray(y_true, dtype=int),
                          "t": np.asarray(elapsed, dtype=float), "g": np.asarray(groups)})
    safe_alarm_count = 0
    safe_seconds = 0.0
    for _, part in frame.groupby("g", sort=False):
        part = part.sort_values("t")
        a = part["alarm"].to_numpy()
        y = part["y"].to_numpy()
        t = part["t"].to_numpy()
        if len(part) > 1:
            dt = np.diff(t)
            valid_interval = (dt > 0) & (dt <= max_gap)
            safe_seconds += float(np.sum(dt[valid_interval] * (y[:-1][valid_interval] == 0)))
        prev = np.r_[False, a[:-1]]
        safe_alarm_count += int(np.sum(a & ~prev & (y == 0)))
    return safe_alarm_count / (safe_seconds / 3600.0) if safe_seconds > 0 else np.nan


def evaluate_fold(model_name, train_df, test_df, fold_name, output_dir):
    Xtr = train_df[FEATURES].astype(float)
    Xte = test_df[FEATURES].astype(float)
    ytr_multi = train_df[TARGET].astype(str)
    yte = test_df[TARGET].astype(str)
    ytr_bin = (~ytr_multi.isin(SAFE_LABELS)).astype(int)
    train_groups = _session_groups(train_df)
    test_groups = _session_groups(test_df)

    if model_name == "LightGBM":
        best_params, threshold, threshold_info = select_lightgbm_params(
            Xtr.reset_index(drop=True), ytr_bin.reset_index(drop=True), train_groups,
            train_df["elapsed_s"].to_numpy()
        )
    else:
        threshold, threshold_info = tune_threshold_oof(
            Xtr.reset_index(drop=True), ytr_bin.reset_index(drop=True), train_groups,
            train_df["elapsed_s"].to_numpy(), model_name
        )
        best_params = {}

    stage1 = make_binary_model(model_name, params=best_params)
    stage1.fit(Xtr, ytr_bin)
    p_hazard = _positive_probability(stage1, Xte)

    hazard_train = ~ytr_multi.isin(SAFE_LABELS)
    stage2_hazard = make_multiclass_model(model_name, params=best_params)
    stage2_hazard.fit(Xtr.loc[hazard_train], ytr_multi.loc[hazard_train])
    safe_train = ytr_multi.isin(SAFE_LABELS)
    safe_classes = ytr_multi.loc[safe_train].nunique()
    stage2_safe = make_multiclass_model(model_name, params=best_params) if safe_classes > 1 else DummyClassifier(strategy="most_frequent")
    stage2_safe.fit(Xtr.loc[safe_train], ytr_multi.loc[safe_train])

    alarm = consecutive_alarm(p_hazard, test_groups, test_df["elapsed_s"].to_numpy(), threshold)
    y_pred = np.asarray(stage2_safe.predict(Xte)).reshape(-1).astype(object)
    if alarm.any():
        y_pred[alarm] = np.asarray(stage2_hazard.predict(Xte.loc[alarm])).reshape(-1)

    y_true_bin = (~yte.isin(SAFE_LABELS)).astype(int).to_numpy()
    hazard_recall = recall_score(y_true_bin, alarm, zero_division=0) if np.any(y_true_bin == 1) else np.nan
    safe_mask = y_true_bin == 0
    specificity = float(np.mean(~alarm[safe_mask])) if safe_mask.any() else np.nan
    balanced = 0.5 * (hazard_recall + specificity) if np.isfinite(hazard_recall) and np.isfinite(specificity) else np.nan
    safe_far = float(alarm[safe_mask].mean()) if safe_mask.any() else np.nan
    per_class = {}
    for cls in ALL_LABELS:
        mask = yte.to_numpy() == cls
        per_class[f"recall_{cls}"] = float(np.mean(y_pred[mask] == cls)) if mask.any() else np.nan

    metrics = {
        "model": model_name, "held_out_environment": fold_name,
        "rows_test": len(test_df), "threshold": threshold,
        "balanced_accuracy_after_10": balanced,
        "hazard_recall_after_10": hazard_recall,
        "hazard_specificity_after_10": specificity,
        "safe_false_alarm_fraction_after_10": safe_far,
        "false_alarms_per_safe_hour": false_alarms_per_hour(alarm, y_true_bin, test_df["elapsed_s"].to_numpy(), test_groups),
        "macro_f1_present_classes": f1_score(yte, y_pred, labels=sorted(yte.unique()), average="macro", zero_division=0),
        "weighted_f1_present_classes": f1_score(yte, y_pred, labels=sorted(yte.unique()), average="weighted", zero_division=0),
        **per_class, "threshold_note": threshold_info.get("note", ""),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(confusion_matrix(yte, y_pred, labels=ALL_LABELS), index=ALL_LABELS, columns=ALL_LABELS).to_csv(
        output_dir / f"confusion_{model_name}_{fold_name}.csv")
    pd.DataFrame(classification_report(yte, y_pred, labels=ALL_LABELS, output_dict=True, zero_division=0)).T.to_csv(
        output_dir / f"classification_{model_name}_{fold_name}.csv")
    return metrics, stage1, stage2_hazard, threshold, threshold_info, stage2_safe, best_params

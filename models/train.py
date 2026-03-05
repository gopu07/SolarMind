"""
Layer 3 — Model Training.

Walk-forward time-series cross-validation with XGBoost, Platt scaling,
and comprehensive evaluation. Produces model artifacts in
``models/artifacts/``.

Usage::

    python -m models.train
    python -m models.train --include-inferred
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import structlog
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

# ── Ensure project root on sys.path ────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config  # noqa: E402
from features.pipeline import FEATURE_COLUMNS  # noqa: E402

log = structlog.get_logger(__name__)


# =====================================================================
# Walk-forward time-series splits
# =====================================================================
def walk_forward_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    gap_days: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward time-series CV indices with a gap.

    Enforces a ``gap_days`` gap between the end of each training fold
    and the start of its validation fold to prevent label leakage.

    Args:
        df: Feature DataFrame with a ``timestamp`` column.
        n_folds: Number of CV folds.
        gap_days: Days between train end and val start.

    Returns:
        List of ``(train_indices, val_indices)`` tuples.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"]

    total_days = (timestamps.max() - timestamps.min()).days
    if total_days < 1:
        log.warning("dataset_too_short_for_cv", total_days=total_days)
        return []

    # Reserve last N days for test — adaptive for small datasets
    test_reserve = min(90, max(1, total_days // 5))
    test_cutoff = timestamps.max() - pd.Timedelta(days=test_reserve)
    cv_df = df[timestamps < test_cutoff].copy()
    cv_ts = pd.to_datetime(cv_df["timestamp"])

    if len(cv_df) < 50:
        log.warning("cv_data_too_small", rows=len(cv_df))
        return []

    cv_days = (cv_ts.max() - cv_ts.min()).days
    if cv_days < gap_days * 2:
        log.warning("cv_days_too_short", cv_days=cv_days, gap_days=gap_days)
        return []

    fold_size = max(1, cv_days // (n_folds + 1))

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_folds):
        train_end = cv_ts.min() + pd.Timedelta(days=fold_size * (i + 1))
        val_start = train_end + pd.Timedelta(days=gap_days)
        val_end = val_start + pd.Timedelta(days=fold_size)

        train_mask = cv_ts <= train_end
        val_mask = (cv_ts >= val_start) & (cv_ts <= val_end)

        train_idx = cv_df.index[train_mask].values
        val_idx = cv_df.index[val_mask].values

        if len(train_idx) > 10 and len(val_idx) > 5:
            splits.append((train_idx, val_idx))
            log.info(
                "cv_fold_created",
                fold=i + 1,
                train_end=str(train_end.date()),
                val_start=str(val_start.date()),
                val_end=str(val_end.date()),
                train_rows=len(train_idx),
                val_rows=len(val_idx),
            )

    return splits


# =====================================================================
# Find optimal business threshold (maximises F0.5)
# =====================================================================
def _find_business_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the probability threshold that maximises F0.5.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.

    Returns:
        Optimal threshold value.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # F0.5 = (1 + 0.25) * precision * recall / (0.25 * precision + recall)
    f05 = np.where(
        (0.25 * precisions[:-1] + recalls[:-1]) > 0,
        1.25 * precisions[:-1] * recalls[:-1] / (0.25 * precisions[:-1] + recalls[:-1]),
        0.0,
    )
    best_idx = np.argmax(f05)
    return float(thresholds[best_idx])


# =====================================================================
# Calibration curve
# =====================================================================
def _calibration_curve_decile(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> List[Dict[str, float]]:
    """Compute calibration curve in decile bins.

    Args:
        y_true: True labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        List of dicts with ``bin_start``, ``bin_end``, ``expected``,
        ``actual``, ``count``.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        result.append(
            {
                "bin_start": round(float(lo), 2),
                "bin_end": round(float(hi), 2),
                "expected": round(float(y_prob[mask].mean()), 4),
                "actual": round(float(y_true[mask].mean()), 4),
                "count": int(mask.sum()),
            }
        )
    return result


# =====================================================================
# Main training routine
# =====================================================================
def train(include_inferred: bool = False) -> Dict[str, Any]:
    """Train and evaluate the XGBoost failure prediction model.

    Args:
        include_inferred: If True, include rows with ``label_source='inferred'``
                          in training data.

    Returns:
        Training report dict (also saved to disk).
    """
    # ── Load features ───────────────────────────────────────────────
    features_path = config.PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        log.error("features_parquet_missing", path=str(features_path))
        sys.exit(1)

    df = pd.read_parquet(features_path)
    log.info("features_loaded", rows=len(df), columns=len(df.columns))

    # ── Filter by label source ──────────────────────────────────────
    if not include_inferred and "label_source" in df.columns:
        before = len(df)
        df = df[df["label_source"] != "inferred"].copy()
        log.info(
            "inferred_labels_excluded",
            dropped=before - len(df),
            remaining=len(df),
        )
    else:
        log.info("inferred_labels_included")

    # ── Prepare X / y ───────────────────────────────────────────────
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    if len(available_features) < len(FEATURE_COLUMNS):
        missing = set(FEATURE_COLUMNS) - set(available_features)
        log.warning("missing_feature_columns", missing=list(missing))

    df = df.sort_values("timestamp").reset_index(drop=True)
    X = df[available_features].copy()
    y = df["label"].astype(int).values

    # Fill any remaining NaNs for XGBoost
    X = X.fillna(0)

    log.info(
        "training_data_prepared",
        rows=len(X),
        features=len(available_features),
        positive_rate=f"{y.mean():.4%}",
    )

    # ── Walk-forward CV ─────────────────────────────────────────────
    splits = walk_forward_splits(df, n_folds=5, gap_days=10)
    if not splits:
        log.warning("no_cv_splits_generated_using_simple_split")
        # Fallback: simple 80/20 temporal split
        split_idx = int(len(X) * 0.8)
        splits = [(np.arange(split_idx), np.arange(split_idx, len(X)))]

    best_iterations: List[int] = []
    fold_metrics: List[Dict[str, float]] = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pos_count = int(y_train.sum())
        neg_count = int((y_train == 0).sum())
        spw = neg_count / max(pos_count, 1)

        model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric=["auc", "aucpr"],
            early_stopping_rounds=50,
            random_state=42,
            use_label_encoder=False,
            verbosity=0,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        best_iter = model.best_iteration
        best_iterations.append(best_iter)

        y_prob = model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, y_prob) if y_val.sum() > 0 else 0.0
        roc = roc_auc_score(y_val, y_prob) if y_val.sum() > 0 else 0.0

        fold_metrics.append({"fold": fold_num, "pr_auc": pr_auc, "roc_auc": roc, "best_iteration": best_iter})
        log.info(
            "cv_fold_complete",
            fold=fold_num,
            pr_auc=f"{pr_auc:.4f}",
            roc_auc=f"{roc:.4f}",
            best_iteration=best_iter,
        )

    # ── Retrain on all CV data ──────────────────────────────────────
    mean_best_iter = max(int(np.mean(best_iterations)), 10)
    log.info("retraining_on_all_data", n_estimators=mean_best_iter)

    # Test set = last N days (adaptive for small datasets)
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[timestamps.notna()].copy()
    timestamps = timestamps.dropna()
    total_days = (timestamps.max() - timestamps.min()).days
    test_days = min(90, max(1, total_days // 5))
    test_cutoff = timestamps.max() - pd.Timedelta(days=test_days)
    train_mask = timestamps < test_cutoff
    test_mask = timestamps >= test_cutoff

    X = df[available_features].fillna(0)
    y = df["label"].astype(int).values

    X_train_full, y_train_full = X[train_mask], y[train_mask.values]
    X_test, y_test = X[test_mask], y[test_mask.values]

    pos_count_full = int(y_train_full.sum())
    neg_count_full = int((y_train_full == 0).sum())
    spw_full = neg_count_full / max(pos_count_full, 1)

    final_model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=mean_best_iter,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw_full,
        eval_metric=["auc", "aucpr"],
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )
    final_model.fit(X_train_full, y_train_full, verbose=False)

    # ── Platt scaling (calibration) ─────────────────────────────────
    log.info("applying_platt_calibration")
    calibrated_model = CalibratedClassifierCV(
        final_model,
        method="sigmoid",
        cv="prefit",
    )
    # Use test set for calibration fitting (or a subset of train)
    # Ensure calibration data has both classes
    cal_split = int(len(X_train_full) * 0.85)
    X_cal = X_train_full.iloc[cal_split:]
    y_cal = y_train_full[cal_split:]
    if len(X_cal) > 50 and len(np.unique(y_cal)) > 1:
        calibrated_model.fit(X_cal, y_cal)
    elif len(np.unique(y_train_full)) > 1:
        calibrated_model.fit(X_train_full, y_train_full)
    else:
        # Only one class present — skip calibration, wrap model anyway
        log.warning("single_class_in_calibration_data")
        calibrated_model.fit(X_train_full, y_train_full)

    # ── Evaluate on test set ────────────────────────────────────────
    if len(X_test) == 0 or y_test.sum() == 0:
        log.warning("test_set_empty_or_no_positives", test_rows=len(X_test))
        # Generate synthetic metrics for the report
        report = _generate_report_no_test(fold_metrics, mean_best_iter, df)
    else:
        y_prob_test = calibrated_model.predict_proba(X_test)[:, 1]

        pr_auc_test = average_precision_score(y_test, y_prob_test)
        roc_auc_test = roc_auc_score(y_test, y_prob_test)
        f1_test = f1_score(y_test, (y_prob_test >= 0.5).astype(int))

        biz_threshold = _find_business_threshold(y_test, y_prob_test)
        y_pred_biz = (y_prob_test >= biz_threshold).astype(int)
        prec_biz = precision_score(y_test, y_pred_biz, zero_division=0)
        rec_biz = recall_score(y_test, y_pred_biz, zero_division=0)

        cal_curve = _calibration_curve_decile(y_test, y_prob_test)

        # Per-plant PR-AUC
        plant_prauc: Dict[str, float] = {}
        if "plant_id" in df.columns:
            test_df = df[test_mask].copy()
            for pid, grp in test_df.groupby("plant_id"):
                idx_in_test = grp.index - X_test.index[0]
                valid_idx = [i for i in idx_in_test if 0 <= i < len(y_prob_test)]
                if valid_idx and y_test[valid_idx].sum() > 0:
                    plant_prauc[str(pid)] = round(
                        average_precision_score(y_test[valid_idx], y_prob_test[valid_idx]), 4
                    )
                else:
                    plant_prauc[str(pid)] = 0.0

        report = {
            "pr_auc": round(pr_auc_test, 4),
            "roc_auc": round(roc_auc_test, 4),
            "f1_at_0_5": round(f1_test, 4),
            "business_threshold": round(biz_threshold, 4),
            "precision_at_biz_threshold": round(prec_biz, 4),
            "recall_at_biz_threshold": round(rec_biz, 4),
            "calibration_curve": cal_curve,
            "per_plant_pr_auc": plant_prauc,
            "cv_fold_metrics": fold_metrics,
            "mean_best_iteration": mean_best_iter,
            "test_set_rows": len(X_test),
            "test_set_positives": int(y_test.sum()),
            "include_inferred": include_inferred,
        }

        log.info(
            "evaluation_complete",
            pr_auc=report["pr_auc"],
            roc_auc=report["roc_auc"],
            f1=report["f1_at_0_5"],
            threshold=report["business_threshold"],
        )

    # ── Save artifacts ──────────────────────────────────────────────
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = config.ARTIFACTS_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(calibrated_model, f)
    log.info("model_saved", path=str(model_path))

    # Save the uncalibrated base model for SHAP TreeExplainer
    base_model_path = config.ARTIFACTS_DIR / "base_model.pkl"
    with open(base_model_path, "wb") as f:
        pickle.dump(final_model, f)
    log.info("base_model_saved", path=str(base_model_path))

    threshold_path = config.ARTIFACTS_DIR / "threshold.json"
    threshold_val = report.get("business_threshold", 0.5)
    with open(threshold_path, "w") as f:
        json.dump({"business_threshold": threshold_val}, f, indent=2)

    features_path_out = config.ARTIFACTS_DIR / "feature_columns.json"
    with open(features_path_out, "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    report_path = config.ARTIFACTS_DIR / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("training_report_saved", path=str(report_path))

    return report


def _generate_report_no_test(
    fold_metrics: List[Dict[str, float]],
    mean_best_iter: int,
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """Generate a report when no valid test set exists.

    Uses cross-validation metrics as the primary evaluation.

    Args:
        fold_metrics: List of per-fold metric dicts.
        mean_best_iter: Mean best iteration from CV.
        df: Full DataFrame.

    Returns:
        Report dict.
    """
    avg_pr_auc = np.mean([m["pr_auc"] for m in fold_metrics]) if fold_metrics else 0.0
    avg_roc_auc = np.mean([m["roc_auc"] for m in fold_metrics]) if fold_metrics else 0.0

    log.warning(
        "using_cv_metrics_as_report",
        avg_pr_auc=f"{avg_pr_auc:.4f}",
        note="Test set empty or has no positives",
    )

    return {
        "pr_auc": round(float(avg_pr_auc), 4),
        "roc_auc": round(float(avg_roc_auc), 4),
        "f1_at_0_5": 0.0,
        "business_threshold": 0.5,
        "precision_at_biz_threshold": 0.0,
        "recall_at_biz_threshold": 0.0,
        "calibration_curve": [],
        "per_plant_pr_auc": {},
        "cv_fold_metrics": fold_metrics,
        "mean_best_iteration": mean_best_iter,
        "test_set_rows": 0,
        "test_set_positives": 0,
        "note": "No valid test set — metrics are from cross-validation averages.",
    }


# =====================================================================
# CLI
# =====================================================================
def main() -> None:
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(description="Train SolarMind failure model")
    parser.add_argument(
        "--include-inferred",
        action="store_true",
        help="Include inferred labels in training data",
    )
    args = parser.parse_args()

    report = train(include_inferred=args.include_inferred)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  PR-AUC:            {report['pr_auc']}")
    print(f"  ROC-AUC:           {report['roc_auc']}")
    print(f"  F1 @ 0.5:          {report['f1_at_0_5']}")
    print(f"  Business threshold:{report['business_threshold']}")
    print(f"  Artifacts saved to:{config.ARTIFACTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

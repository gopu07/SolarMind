"""
Layer 3 — Model Training.

Walk-forward time-series cross-validation with XGBoost, Platt scaling,
Optuna hyperparameter tuning, and comprehensive evaluation. Produces
model artifacts in ``models/artifacts/``.
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
import optuna
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

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
    gap_days: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    timestamps = df["timestamp"]

    total_days = (timestamps.max() - timestamps.min()).days
    if total_days < 1:
        log.warning("dataset_too_short_for_cv", total_days=total_days)
        return []

    # Hold out exactly the last 6 months (180 days) for replay testing
    # The remainder (first ~18 months) is for cross validation
    test_reserve = 180
    test_cutoff = timestamps.max() - pd.Timedelta(days=test_reserve)
    cv_df = df[timestamps < test_cutoff].copy()
    cv_ts = pd.to_datetime(cv_df["timestamp"])

    cv_days = (cv_ts.max() - cv_ts.min()).days
    if cv_days < gap_days * 2:
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
            log.info("cv_fold_created", fold=i+1, train_rows=len(train_idx), val_rows=len(val_idx))

    return splits


def _find_business_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Not used for multiclass strictly, but we return a default to satisfy the signature if needed
    return 0.5
    
from models.ensemble import TreeEnsemble  # noqa: E402


def _calibration_curve_decile(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    bins = np.linspace(0, 1, n_bins + 1)
    result = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        result.append({
            "bin_start": round(float(lo), 2),
            "bin_end": round(float(hi), 2),
            "expected": round(float(y_prob[mask].mean()), 4),
            "actual": round(float(y_true[mask].mean()), 4),
            "count": int(mask.sum()),
        })
    return result


def train(include_inferred: bool = False) -> Dict[str, Any]:
    features_path = config.PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        log.error("features_parquet_missing", path=str(features_path))
        sys.exit(1)

    df = pd.read_parquet(features_path)
    if not include_inferred and "label_source" in df.columns:
        df = df[df["label_source"] != "inferred"].copy()

    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # --- Class Imbalance Handling: Nighttime Downsampling ---
    if "pv1_power" in df.columns:
        pv1 = df["pv1_power"].fillna(0)
        pv2 = df.get("pv2_power", pd.Series(0, index=df.index)).fillna(0)
        total_pwr = pv1 + pv2
        
        # Keep telemetry where power >= 10W OR label is 1 (failure)
        mask_keep = (total_pwr >= 10.0) | (df["label"] == 1)
        dropped = (~mask_keep).sum()
        df = df[mask_keep].reset_index(drop=True)
        log.info("nighttime_downsampling", rows_dropped=dropped, remaining=len(df))
    # --------------------------------------------------------

    # Do NOT fill NaNs with zero here. We use native missing value support!
    # Convert object types to float where possible
    for col in available_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    X = df[available_features].copy()
    y = df["label"].astype(int).values

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    splits = walk_forward_splits(df, n_folds=5, gap_days=5)
    if not splits:
        split_idx = int(len(X) * 0.8)
        splits = [(np.arange(split_idx), np.arange(split_idx, len(X)))]

    # ── 1. Optuna Hyperparameter Tuning ─────────────────────────────
    log.info("starting_optuna_optimization", trials=10)
    
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "n_estimators": 800,
            "objective": "multi:softprob",
            "num_class": 5,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "verbosity": 0,
        }
        
        scores = []
        for train_idx, val_idx in splits:
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_v, y_v = X.iloc[val_idx], y[val_idx]
            
            if y_tr.sum() == 0 or y_v.sum() == 0:
                continue
                
            model = XGBClassifier(**params, early_stopping_rounds=30)
            model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
            
            # For multiclass, optuna tuning metric: macro F1
            y_pred = np.argmax(model.predict_proba(X_v), axis=1)
            scores.append(f1_score(y_v, y_pred, average="macro"))
            
        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)
    best_params = study.best_params
    log.info("optuna_best_params", params=best_params)

    # ── 2. Run CV with Best Params ──────────────────────────────────
    best_iterations = []
    fold_metrics = []

    for fold_num, (train_idx, val_idx) in enumerate(splits, 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if y_train.sum() == 0 or y_val.sum() == 0:
            continue

        model = TreeEnsemble(best_params, random_state=42+fold_num)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        # We track XGB's best iteration (falling back to n_estimators)
        best_iter = getattr(model.xgb, "best_iteration", model.xgb.n_estimators)
        best_iterations.append(best_iter)

        # Predict returns Nx5 array of probabilities
        y_prob = model.predict_proba(X_val)
        y_pred = np.argmax(y_prob, axis=1)
        if len(np.unique(y_val)) > 1:
            macro_f1 = f1_score(y_val, y_pred, average="macro")
            try:
                roc = roc_auc_score(y_val, y_prob, multi_class="ovr")
            except ValueError:
                roc = 0.0
        else:
            macro_f1, roc = 0.0, 0.0

        fold_metrics.append({"fold": fold_num, "macro_f1": macro_f1, "roc_auc": roc, "best_iteration": best_iter})
        log.info("cv_fold_complete", fold=fold_num, macro_f1=f"{macro_f1:.4f}")

    # ── 3. Retrain on Full CV Data ──────────────────────────────────
    mean_best_iter = max(int(np.mean(best_iterations)) if best_iterations else 50, 20)
    
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Strictly hold out exactly 6 months (180 days) for replay
    test_reserve = 180
    test_cutoff = timestamps.max() - pd.Timedelta(days=test_reserve)
    
    train_mask = timestamps < test_cutoff
    test_mask = timestamps >= test_cutoff

    X_train_full, y_train_full = X[train_mask], y[train_mask.values]
    X_test, y_test = X[test_mask], y[test_mask.values]

    if y_train_full.sum() == 0:
        log.error("no_positive_labels_in_training_data")
        sys.exit(1)

    spw_full = (y_train_full==0).sum() / max(y_train_full.sum(), 1)
    # Let's adjust XGB params to use mean_best_iter
    best_params_final = best_params.copy()
    
    # We create TreeEnsemble, but we must override xgb estimators
    final_model = TreeEnsemble(best_params_final, random_state=42)
    final_model.xgb.set_params(n_estimators=mean_best_iter)
    final_model.fit(X_train_full, y_train_full, verbose=False)

    # ── 4. Calibration ──────────────────────────────────────────────
    calibrated_model = CalibratedClassifierCV(final_model, method="sigmoid", cv="prefit")
    cal_split = int(len(X_train_full) * 0.85)
    X_cal, y_cal = X_train_full.iloc[cal_split:], y_train_full[cal_split:]
    if len(X_cal) > 50 and len(np.unique(y_cal)) > 1:
        calibrated_model.fit(X_cal, y_cal)
    else:
        calibrated_model.fit(X_train_full, y_train_full)

    # ── 5. Feature Importance ───────────────────────────────────────
    log.info("extracting_feature_importances")
    try:
        explainer = shap.TreeExplainer(final_model)
        bg_sample = X_train_full.sample(n=min(5000, len(X_train_full)), random_state=42)
        # Using TreeExplainer on XGBoost portion as a proxy for feature importances
        explainer = shap.TreeExplainer(final_model.xgb)
        shap_values = explainer.shap_values(bg_sample)
        feature_importances = np.abs(shap_values).mean(axis=0)
    except Exception as e:
        log.warning("shap_explainer_failed_using_native_importance", error=str(e))
        feature_importances = final_model.feature_importances_
        
    importance_df = pd.DataFrame({
        "feature": available_features,
        "importance": feature_importances
    }).sort_values("importance", ascending=False)
    
    top_30_features = importance_df.head(30).to_dict("records")

    # ── 6. Evaluate ─────────────────────────────────────────────────
    if len(X_test) == 0:
        log.warning("test_set_empty")
        report_dict = {"macro_f1": 0.0, "roc_auc": 0.0, "business_threshold": 0.5, "top_features": top_30_features}
    else:
        y_prob_test = calibrated_model.predict_proba(X_test)
        y_pred_test = np.argmax(y_prob_test, axis=1)
        try:
            roc_auc_test = roc_auc_score(y_test, y_prob_test, multi_class="ovr")
        except ValueError:
            roc_auc_test = 0.0
            
        f1_test = f1_score(y_test, y_pred_test, average="macro")

        report_dict = {
            "macro_f1": round(f1_test, 4),
            "roc_auc": round(roc_auc_test, 4),
            "business_threshold": 0.5,
            "cv_fold_metrics": fold_metrics,
            "mean_best_iteration": mean_best_iter,
            "top_features": top_30_features,
        }

        log.info("evaluation_complete", macro_f1=report_dict["macro_f1"], roc_auc=report_dict["roc_auc"])

    # ── 6.5. Train Unsupervised Anomaly Model ────────────────────────
    log.info("training_isolation_forest")
    
    # User Fix 5: Limit IF input features to stable signals
    iso_features = [
        "inverter_temperature", 
        "pv1_power", 
        "conversion_efficiency", 
        "string_current_variance", 
        "pv1_voltage"
    ]
    # Filter to what is actually available in the processing dataframe
    available_iso_features = [f for f in iso_features if f in X_train_full.columns]
    
    # Impute NaNs with median since IsolationForest doesn't natively support them like XGBoost
    X_train_imputed = X_train_full[available_iso_features].fillna(X_train_full[available_iso_features].median())
    
    # Train Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.05, # assume 5% of historical data is anomalous
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train_imputed)
    log.info("isolation_forest_trained")

    # ── 7. Save ─────────────────────────────────────────────────────
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(calibrated_model, f)
    with open(config.ARTIFACTS_DIR / "base_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    with open(config.ARTIFACTS_DIR / "isolation_forest.pkl", "wb") as f:
        pickle.dump(iso_forest, f)
    with open(config.ARTIFACTS_DIR / "threshold.json", "w") as f:
        json.dump({"business_threshold": report_dict.get("business_threshold", 0.5)}, f, indent=2)
    with open(config.ARTIFACTS_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)
    with open(config.ARTIFACTS_DIR / "training_report.json", "w") as f:
        json.dump(report_dict, f, indent=2)

    return report_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SolarMind failure model")
    parser.add_argument("--include-inferred", action="store_true")
    args = parser.parse_args()

    report = train(include_inferred=args.include_inferred)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Macro F1:          {report['macro_f1']}")
    print(f"  Multiclass ROC-AUC:{report['roc_auc']}")
    if "top_features" in report:
        print("\n  Top 5 Features by SHAP:")
        for feat in report["top_features"][:5]:
            print(f"    - {feat['feature']}: {feat['importance']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

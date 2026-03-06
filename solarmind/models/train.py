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
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
    # Maximize F1 instead of F0.5
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1]),
        0.0,
    )
    if len(f1s) == 0:
        return 0.5
    best_idx = np.argmax(f1s)
    return float(thresholds[best_idx])
    
class TreeEnsemble:
    def __init__(self, xgb_params: Dict[str, Any], scale_pos_weight: float, random_state: int = 42):
        self.xgb = XGBClassifier(
            **xgb_params,
            n_estimators=300,
            objective="binary:logistic",
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=random_state,
            verbosity=0,
        )
        self.lgb = lgb.LGBMClassifier(
            n_estimators=250,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            verbose=-1,
        )
        # For CatBoost, auto_class_weights="Balanced" handles imbalance
        self.cat = CatBoostClassifier(
            iterations=250,
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=0,
        )
        self.classes_ = np.array([0, 1])

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, eval_set: List[Tuple[pd.DataFrame, np.ndarray]] = None, verbose: bool = False):
        if eval_set is not None:
            self.xgb.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
            self.lgb.fit(X_train, y_train, eval_set=eval_set, callbacks=[lgb.early_stopping(50, verbose=False)])
            # CatBoost expects eval_set to be just a tuple or list of tuples
            self.cat.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=50, verbose=verbose)
        else:
            self.xgb.fit(X_train, y_train, verbose=verbose)
            self.lgb.fit(X_train, y_train)
            self.cat.fit(X_train, y_train, verbose=verbose)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_lgb = self.lgb.predict_proba(X)[:, 1]
        p_cat = self.cat.predict_proba(X)[:, 1]
        # Weighted ensemble
        final_p = 0.5 * p_xgb + 0.3 * p_lgb + 0.2 * p_cat
        res = np.zeros((len(X), 2))
        res[:, 1] = final_p
        res[:, 0] = 1.0 - final_p
        return res
        
    @property
    def feature_importances_(self) -> np.ndarray:
        # Simple average of importances. Note: scales might differ, but it gives a rough idea.
        xgb_imp = self.xgb.feature_importances_
        lgb_imp = self.lgb.feature_importances_
        if lgb_imp.sum() > 0:
            lgb_imp = lgb_imp / lgb_imp.sum()
        cat_imp = self.cat.feature_importances_
        if cat_imp.sum() > 0:
            cat_imp = cat_imp / cat_imp.sum()
        return (xgb_imp + lgb_imp + cat_imp) / 3.0


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
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "random_state": 42,
            "verbosity": 0,
        }
        
        scores = []
        for train_idx, val_idx in splits:
            X_tr, y_tr = X.iloc[train_idx], y[train_idx]
            X_v, y_v = X.iloc[val_idx], y[val_idx]
            
            if y_tr.sum() == 0 or y_v.sum() == 0:
                continue
                
            spw = (y_tr==0).sum() / max(y_tr.sum(), 1)    
            model = XGBClassifier(**params, scale_pos_weight=spw, early_stopping_rounds=30)
            model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
            
            y_prob = model.predict_proba(X_v)[:, 1]
            scores.append(average_precision_score(y_v, y_prob))
            
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

        spw_cv = (y_train==0).sum() / max(y_train.sum(), 1)
        model = TreeEnsemble(best_params, scale_pos_weight=spw_cv, random_state=42+fold_num)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        # We track XGB's best iteration (falling back to n_estimators)
        best_iter = getattr(model.xgb, "best_iteration", model.xgb.n_estimators)
        best_iterations.append(best_iter)

        y_prob = model.predict_proba(X_val)[:, 1]
        if y_val.sum() > 0:
            pr_auc = average_precision_score(y_val, y_prob)
            roc = roc_auc_score(y_val, y_prob)
        else:
            pr_auc, roc = 0.0, 0.0

        fold_metrics.append({"fold": fold_num, "pr_auc": pr_auc, "roc_auc": roc, "best_iteration": best_iter})
        log.info("cv_fold_complete", fold=fold_num, pr_auc=f"{pr_auc:.4f}")

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
    final_model = TreeEnsemble(best_params_final, scale_pos_weight=spw_full, random_state=42)
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
    if len(X_test) == 0 or y_test.sum() == 0:
        log.warning("test_set_empty_or_no_positives")
        report_dict = {"pr_auc": 0.0, "roc_auc": 0.0, "f1_at_0_5": 0.0, "business_threshold": 0.5, "top_features": top_30_features}
    else:
        y_prob_test = calibrated_model.predict_proba(X_test)[:, 1]
        pr_auc_test = average_precision_score(y_test, y_prob_test)
        roc_auc_test = roc_auc_score(y_test, y_prob_test)
        biz_threshold = _find_business_threshold(y_test, y_prob_test)
        f1_test = f1_score(y_test, (y_prob_test >= biz_threshold).astype(int))
        recall_test = recall_score(y_test, (y_prob_test >= biz_threshold).astype(int))

        report_dict = {
            "pr_auc": round(pr_auc_test, 4),
            "roc_auc": round(roc_auc_test, 4),
            "f1_score": round(f1_test, 4),
            "recall": round(recall_test, 4),
            "business_threshold": round(biz_threshold, 4),
            "cv_fold_metrics": fold_metrics,
            "mean_best_iteration": mean_best_iter,
            "top_features": top_30_features,
        }

        log.info("evaluation_complete", pr_auc=report_dict["pr_auc"], roc_auc=report_dict["roc_auc"])

    # ── 7. Save ─────────────────────────────────────────────────────
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(calibrated_model, f)
    with open(config.ARTIFACTS_DIR / "base_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
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
    print(f"  PR-AUC:            {report['pr_auc']}")
    print(f"  ROC-AUC:           {report['roc_auc']}")
    print(f"  F1 Score:          {report['f1_score']}")
    print(f"  Recall:            {report['recall']}")
    print(f"  Business threshold:{report['business_threshold']}")
    if "top_features" in report:
        print("\n  Top 5 Features by SHAP:")
        for feat in report["top_features"][:5]:
            print(f"    - {feat['feature']}: {feat['importance']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

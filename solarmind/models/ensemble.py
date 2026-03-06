"""
Shared model ensemble classes for SolarMind.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

class TreeEnsemble:
    _estimator_type = "classifier"

    def __init__(self, xgb_params: Dict[str, Any], random_state: int = 42):
        self.xgb_params = xgb_params
        self.random_state = random_state
        
        # Determine if class weights are provided in xgb_params
        self.xgb = XGBClassifier(
            **xgb_params,
            n_estimators=300,
            objective="multi:softprob",
            num_class=5,
            eval_metric="mlogloss",
            random_state=random_state,
            verbosity=0,
        )
        self.lgb = lgb.LGBMClassifier(
            n_estimators=250,
            objective="multiclass",
            num_class=5,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
        # For CatBoost, auto_class_weights="Balanced" handles imbalance
        self.cat = CatBoostClassifier(
            iterations=250,
            loss_function="MultiClass",
            classes_count=5,
            auto_class_weights="Balanced",
            random_seed=random_state,
            verbose=0,
        )
        self.classes_ = np.array([0, 1, 2, 3, 4])

    def get_params(self, deep=True):
        return {
            "xgb_params": self.xgb_params,
            "random_state": self.random_state
        }

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

    def _pad_proba(self, p: np.ndarray) -> np.ndarray:
        if p.shape[1] < 5:
            padded = np.zeros((p.shape[0], 5))
            padded[:, :p.shape[1]] = p
            return padded
        elif p.shape[1] > 5:
            return p[:, :5]
        return p

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_xgb = self._pad_proba(self.xgb.predict_proba(X))
        p_lgb = self._pad_proba(self.lgb.predict_proba(X))
        p_cat = self._pad_proba(self.cat.predict_proba(X))
        # Weighted ensemble
        final_p = 0.5 * p_xgb + 0.3 * p_lgb + 0.2 * p_cat
        return final_p
        
    @property
    def feature_importances_(self) -> np.ndarray:
        # Average importance from all models as a proxy
        f1 = self.xgb.feature_importances_
        f2 = self.lgb.feature_importances_
        # CatBoost uses a different method for feature importance
        f3 = self.cat.get_feature_importance()
        # Normalize f3 to same scale as others
        f3 = f3 / (f3.sum() + 1e-9)
        return (f1 + f2 + f3) / 3.0

    def get_booster(self):
        """Returns the base XGB booster for SHAP calculation."""
        return self.xgb.get_booster()

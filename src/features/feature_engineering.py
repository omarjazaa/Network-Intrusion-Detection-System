"""
Feature engineering module for scaling features and separating target labels.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []

    def _drop_target_columns(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        df_clean = df.copy()
        candidate_targets = []
        if target_col:
            candidate_targets.append(target_col)
        candidate_targets.extend(['attack_type', 'label', 'target', 'prediction', 'Predicted_Attack'])

        for col in candidate_targets:
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col])

        return df_clean

    def _align_to_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing trained features and order columns exactly like fit time."""
        if not self.feature_names:
            return df.copy()

        aligned = df.copy()
        for feature_name in self.feature_names:
            if feature_name not in aligned.columns:
                aligned[feature_name] = 0

        # Keep only trained features and preserve column order.
        return aligned.reindex(columns=self.feature_names, fill_value=0)

    def fit_transform(self, df: pd.DataFrame, target_col: str = 'attack_type') -> Tuple[pd.DataFrame, pd.Series]:
        df_clean = self._drop_target_columns(df, target_col=target_col)

        # البحث عن عمود الهدف
        actual_target = target_col if target_col in df.columns else None
        if actual_target is None:
            for col in ['attack_type', 'label', 'target', 'prediction', 'Predicted_Attack']:
                if col in df.columns:
                    actual_target = col
                    break

        if actual_target and actual_target in df.columns:
            y = df[actual_target]
            X = df_clean.drop(columns=[actual_target], errors='ignore')
        else:
            X = df_clean
            y = pd.Series(['Normal'] * len(X))

        # اختيار الأعمدة الرقمية فقط للتحجيم (Scaling)
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.feature_names = X.columns.tolist()

        if len(numeric_cols) > 0:
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])

        X = self._align_to_training_features(X)
        return X, y

    def transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        df_clean = self._drop_target_columns(df, target_col=target_col)

        # Align first so the scaler receives the same feature set it saw during fit.
        df_clean = self._align_to_training_features(df_clean)

        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            df_clean[numeric_cols] = self.scaler.transform(df_clean[numeric_cols])

        return df_clean
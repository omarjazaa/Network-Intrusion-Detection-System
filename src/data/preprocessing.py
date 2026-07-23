"""Data preprocessing utilities for NIDS training and inference."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_data(
    df: pd.DataFrame,
    drop_duplicates: bool = True,
    drop_all_na: bool = False,
) -> pd.DataFrame:
    """Basic cleaning used by both training and inference pipelines."""
    df_clean = df.copy()

    # Keep row count stable during inference by only dropping duplicates when requested.
    if drop_duplicates:
        df_clean = df_clean.drop_duplicates()

    if drop_all_na:
        df_clean = df_clean.dropna(how="all")
    return df_clean


class DataPreprocessor:
    """Encodes categorical traffic fields with backward-compatible state handling."""

    def __init__(self, categorical_cols: Optional[list] = None):
        self.categorical_cols = categorical_cols or ["protocol_type", "service", "flag"]
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.unknown_replacements: Dict[str, str] = {}
        self.is_fitted = False

    def _ensure_state(self) -> None:
        """Populate fields that may be missing in legacy pickles."""
        if not hasattr(self, "categorical_cols") or self.categorical_cols is None:
            self.categorical_cols = ["protocol_type", "service", "flag"]
        if not hasattr(self, "label_encoders") or self.label_encoders is None:
            self.label_encoders = {}
        if not hasattr(self, "unknown_replacements") or self.unknown_replacements is None:
            self.unknown_replacements = {}
        if not hasattr(self, "is_fitted"):
            self.is_fitted = False

        # Infer fitted state for artifacts created before `is_fitted` existed.
        if (not self.is_fitted) and isinstance(self.label_encoders, dict) and self.label_encoders:
            self.is_fitted = True

        # Ensure unknown fallback values exist for each known encoder.
        for col, encoder in self.label_encoders.items():
            if col not in self.unknown_replacements:
                classes = getattr(encoder, "classes_", None)
                if classes is not None and len(classes) > 0:
                    self.unknown_replacements[col] = str(classes[0])

    def ensure_compatibility(self) -> "DataPreprocessor":
        """Public compatibility hook for callers after loading pickles."""
        self._ensure_state()
        return self

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._ensure_state()

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        self._ensure_state()
        df_clean = clean_data(df, drop_duplicates=True, drop_all_na=True)

        # Reinitialize learned state on every fit.
        self.label_encoders = {}
        self.unknown_replacements = {}

        for col in self.categorical_cols:
            if col not in df_clean.columns:
                continue

            series = df_clean[col].astype(str).fillna("missing")
            encoder = LabelEncoder()
            encoder.fit(series)

            self.label_encoders[col] = encoder
            self.unknown_replacements[col] = str(encoder.classes_[0])

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._ensure_state()
        if not self.is_fitted:
            raise ValueError("DataPreprocessor is not fitted. Run fit() before transform().")

        df_clean = clean_data(df, drop_duplicates=False, drop_all_na=False)

        for col in self.categorical_cols:
            if col not in df_clean.columns:
                continue
            if col not in self.label_encoders:
                continue

            encoder = self.label_encoders[col]
            fallback = self.unknown_replacements.get(col)
            if fallback is None:
                classes = getattr(encoder, "classes_", [])
                fallback = str(classes[0]) if len(classes) > 0 else "missing"

            series = df_clean[col].astype(str).fillna(fallback)
            known_classes = set(encoder.classes_)
            series = series.where(series.isin(known_classes), fallback)
            df_clean[col] = encoder.transform(series)

        return df_clean

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
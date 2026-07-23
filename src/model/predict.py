import os
import joblib
import pandas as pd

class NIDSPredictor:
    def __init__(self, models_dir=os.path.join("models", "trained")):
        self.models_dir = models_dir
        self.models = {}
        self.preprocessor = None
        self.feature_engineer = None
        self.feature_names = []
        self._load_artifacts()

    def _load_artifacts(self):
        # تحميل الملفات المجهزة
        self.preprocessor = joblib.load(os.path.join(self.models_dir, "preprocessor.pkl"))
        self.feature_engineer = joblib.load(os.path.join(self.models_dir, "feature_engineer.pkl"))

        # Backward compatibility for older serialized preprocessors.
        if hasattr(self.preprocessor, "ensure_compatibility"):
            self.preprocessor.ensure_compatibility()
        
        # تحميل النماذج
        if os.path.exists(os.path.join(self.models_dir, "random_forest.pkl")):
            self.models['random_forest'] = joblib.load(os.path.join(self.models_dir, "random_forest.pkl"))
        if os.path.exists(os.path.join(self.models_dir, "xgboost.pkl")):
            self.models['xgboost'] = joblib.load(os.path.join(self.models_dir, "xgboost.pkl"))

        self.feature_names = getattr(self.feature_engineer, 'feature_names', [])

    def predict(self, df_raw: pd.DataFrame, model_name: str = 'random_forest') -> pd.DataFrame:
        df_clean = self.preprocessor.transform(df_raw)
        
        # Inference must never refit feature engineering state.
        X = self.feature_engineer.transform(df_clean)
        
        model = self.models.get(model_name, list(self.models.values())[0])
        
        preds = model.predict(X)
        
        # إذا كان نموذج XGBoost يحتوي على تسميات الفئات
        if hasattr(model, 'target_classes_'):
            preds = [model.target_classes_[i] for i in preds]

        if len(preds) != len(df_raw):
            raise ValueError(
                "Prediction row count mismatch after preprocessing. "
                "Check input cleaning and feature pipeline compatibility."
            )

        df_results = df_raw.copy()
        df_results['prediction'] = preds
        return df_results
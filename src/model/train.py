import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# إضافة المسار الرئيسي للمشروع
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.loader import load_data
from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer

# المسارات والعمود المستهدف
TARGET_COL = "attack_type"
MODELS_DIR = os.path.join("models", "trained")


def _clear_old_artifacts(models_dir: str) -> None:
    """Remove stale artifacts so new pickles are always schema-compatible."""
    artifact_names = [
        "random_forest.pkl",
        "xgboost.pkl",
        "preprocessor.pkl",
        "feature_engineer.pkl",
    ]
    for artifact in artifact_names:
        artifact_path = os.path.join(models_dir, artifact)
        if os.path.exists(artifact_path):
            os.remove(artifact_path)

def train_models(data_path=None, models_dir=MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    _clear_old_artifacts(models_dir)
    
    if data_path is None:
        data_path = os.path.join("data", "sample", "sample_network_traffic.csv")
        
    print(f"[+] Loading dataset from: {data_path}")
    df_raw = load_data(data_path)
    
    print(f"[+] Processing and cleaning dataset ({len(df_raw)} rows)...")
    preprocessor = DataPreprocessor()
    df_cleaned = preprocessor.fit_transform(df_raw)
    
    # تحديد اسم عمود الهدف
    target_column = None
    for col in [TARGET_COL, 'label', 'target', 'prediction']:
        if col in df_cleaned.columns:
            target_column = col
            break
            
    if not target_column:
        # إذا لم يكن عمود الهدف موجوداً، سننشئ عمود هدف افتراضي للتدريب
        target_column = 'attack_type'
        df_cleaned[target_column] = 'Normal'

    # فصل المتغيرات وهندسة الخصائص
    feature_engineer = FeatureEngineer()
    X, y = feature_engineer.fit_transform(df_cleaned, target_col=target_column)
    
    # 1. تدريب Random Forest
    print("[+] Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    # 2. تدريب XGBoost
    print("[+] Training XGBoost model...")
    y_encoded, y_uniques = pd.factorize(y)
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    xgb_model.fit(X, y_encoded)
    xgb_model.target_classes_ = list(y_uniques)
    
    # حفظ النماذج المحسنة
    print(f"[+] Saving trained models and preprocessors to: {models_dir}")
    joblib.dump(rf_model, os.path.join(models_dir, "random_forest.pkl"))
    joblib.dump(xgb_model, os.path.join(models_dir, "xgboost.pkl"))
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))
    joblib.dump(feature_engineer, os.path.join(models_dir, "feature_engineer.pkl"))
    
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    train_models()
"""
Unit tests for data loading, preprocessing, feature engineering, model training,
prediction, and evaluation components.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.loader import load_csv_data, validate_schema, get_dataset_summary
from src.data.preprocessing import clean_data, DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.model.train import train_models
from src.model.predict import NIDSPredictor
from src.model.evaluate import evaluate_model


class TestNIDSPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_data_path = os.path.join(PROJECT_ROOT, 'data/sample/sample_network_traffic.csv')
        cls.models_dir = os.path.join(PROJECT_ROOT, 'models/trained')
        cls.df_sample = load_csv_data(cls.sample_data_path)

    def test_01_loader(self):
        """Test dataset loading and schema validation."""
        self.assertFalse(self.df_sample.empty, "Loaded sample dataset should not be empty.")
        is_valid, missing = validate_schema(self.df_sample)
        self.assertTrue(is_valid, f"Sample dataset should pass schema validation. Missing: {missing}")

        summary = get_dataset_summary(self.df_sample)
        self.assertIn('total_records', summary)
        self.assertGreater(summary['total_records'], 0)

    def test_02_preprocessing(self):
        """Test data cleaning and DataPreprocessor encoding."""
        cleaned_df = clean_data(self.df_sample)
        self.assertEqual(cleaned_df.isnull().sum().sum(), 0, "Cleaned dataset should have zero missing values.")

        preprocessor = DataPreprocessor()
        X_encoded, y_encoded = preprocessor.fit_transform(cleaned_df)

        self.assertIsNotNone(y_encoded, "Target variable should be encoded.")
        self.assertTrue(preprocessor.is_fitted, "Preprocessor should mark itself as fitted.")
        self.assertIn('protocol_type', preprocessor.encoders)

        # Test inverse transform target
        y_rev = preprocessor.inverse_transform_target(y_encoded)
        self.assertEqual(len(y_rev), len(y_encoded))

    def test_03_feature_engineering(self):
        """Test domain feature creation and scaling."""
        cleaned_df = clean_data(self.df_sample)
        preprocessor = DataPreprocessor()
        X_df, _ = preprocessor.fit_transform(cleaned_df)

        feat_eng = FeatureEngineer()
        X_scaled = feat_eng.fit_transform(X_df)

        self.assertTrue(feat_eng.is_fitted, "FeatureEngineer should mark itself as fitted.")
        self.assertEqual(X_scaled.shape[0], len(X_df))
        self.assertGreater(X_scaled.shape[1], X_df.shape[1], "Engineered features should add new columns.")

    def test_04_train_models(self):
        """Test full model training pipeline and artifact saving."""
        summary = train_models(self.sample_data_path, output_dir=self.models_dir)

        self.assertIn('random_forest', summary)
        self.assertIn('xgboost', summary)

        rf_acc = summary['random_forest']['accuracy']
        xgb_acc = summary['xgboost']['accuracy']

        self.assertGreaterEqual(rf_acc, 0.70, f"Random Forest accuracy ({rf_acc:.2f}) should be >= 0.70")
        self.assertGreaterEqual(xgb_acc, 0.70, f"XGBoost accuracy ({xgb_acc:.2f}) should be >= 0.70")

    def test_05_prediction_engine(self):
        """Test NIDSPredictor inference on new data."""
        predictor = NIDSPredictor(models_dir=self.models_dir)
        predictions = predictor.predict(self.df_sample, model_name='xgboost')

        self.assertIn('Predicted_Attack', predictions.columns)
        self.assertIn('Is_Malicious', predictions.columns)
        self.assertIn('Confidence', predictions.columns)
        self.assertEqual(len(predictions), len(self.df_sample))

    def test_06_evaluation_metrics(self):
        """Test evaluation metrics calculator."""
        y_true = np.array([0, 1, 0, 1, 2])
        y_pred = np.array([0, 1, 0, 0, 2])
        metrics = evaluate_model(y_true, y_pred, class_names=['Normal', 'DoS', 'PortScan'])

        self.assertIn('accuracy', metrics)
        self.assertIn('f1_macro', metrics)
        self.assertIn('confusion_matrix', metrics)
        self.assertEqual(metrics['accuracy'], 0.8)


if __name__ == '__main__':
    unittest.main()

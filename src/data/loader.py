"""
Loader module for reading and validating network intrusion datasets.
"""

import pandas as pd
from typing import Union, Tuple, List, Dict, Any

REQUIRED_FEATURES = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'count', 'srv_count',
    'serror_rate', 'rerror_rate', 'same_srv_rate', 'diff_srv_rate'
]

TARGET_COLUMN = 'attack_type'


def load_csv_data(source: Union[str, Any]) -> pd.DataFrame:
    """
    Loads network traffic CSV data from a file path or file-like buffer.
    """
    try:
        df = pd.read_csv(source)
        # تنظيف أسماء الأعمدة من المسافات الزائدة
        df.columns = [str(col).strip() for col in df.columns]
        return df
    except Exception as e:
        raise ValueError(f"Failed to read CSV dataset: {str(e)}")


# 💡 إضافة اسم مستعار للدالة لضمان عدم حدوث ImportError عند استدعائها باسم load_data
load_data = load_csv_data


def validate_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates that the loaded DataFrame contains all minimum required feature columns.
    Returns (is_valid, missing_columns).
    """
    missing_cols = [col for col in REQUIRED_FEATURES if col not in df.columns]
    is_valid = len(missing_cols) == 0
    return is_valid, missing_cols


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary metadata for a network traffic DataFrame.
    """
    # البحث عن عمود الهدف بأكثر من اسم محتمل
    target_col = None
    for col in [TARGET_COLUMN, 'label', 'target', 'prediction']:
        if col in df.columns:
            target_col = col
            break

    has_target = target_col is not None
    attack_counts = df[target_col].value_counts().to_dict() if has_target else {}
    
    summary = {
        'total_records': len(df),
        'total_columns': len(df.columns),
        'columns': list(df.columns),
        'missing_values': int(df.isnull().sum().sum()),
        'duplicate_rows': int(df.duplicated().sum()),
        'has_target': has_target,
        'attack_distribution': attack_counts
    }
    return summary
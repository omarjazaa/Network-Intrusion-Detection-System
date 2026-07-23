"""
Evaluation module for assessing NIDS machine learning classification performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes standard classification evaluation metrics for NIDS model predictions.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = accuracy_score(y_true_arr, y_pred_arr)
    
    prec_macro = precision_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
    rec_macro = recall_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)
    f1_macro = f1_score(y_true_arr, y_pred_arr, average='macro', zero_division=0)

    prec_weighted = precision_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true_arr, y_pred_arr, average='weighted', zero_division=0)

    labels = list(np.unique(np.concatenate([y_true_arr, y_pred_arr])))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    
    if class_names is not None and len(class_names) >= len(labels) and all(isinstance(label, (int, np.integer)) for label in labels):
        report_names = [class_names[int(label)] if int(label) < len(class_names) else str(label) for label in labels]
    else:
        report_names = [str(label) for label in labels]

    cls_report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        target_names=report_names,
        output_dict=True,
        zero_division=0
    )
    cls_report_str = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        target_names=report_names,
        zero_division=0
    )

    results = {
        'accuracy': float(acc),
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(prec_weighted),
        'recall_weighted': float(rec_weighted),
        'f1_weighted': float(f1_weighted),
        'confusion_matrix': cm.tolist(),
        'class_names': report_names,
        'classification_report_dict': cls_report_dict,
        'classification_report_str': cls_report_str
    }
    
    return results

"""
Visualization helpers for creating Plotly interactive charts in Streamlit.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

# لوحة الألوان المخصصة للأمن السيبراني (Cybersecurity Palette)
COLOR_MAP = {
    'Normal': '#10b981',        # Emerald Green (آمن)
    'DoS': '#ef4444',           # Crimson Red (هجوم DoS)
    'PortScan': '#f59e0b',      # Amber Orange (فحص المنافذ)
    'BruteForce': '#8b5cf6',    # Purple (قوة غاشمة)
    'Botnet': '#ec4899',        # Pink (شبكة بوت)
    'Other': '#3b82f6'          # Electric Blue
}


def plot_attack_distribution(df: pd.DataFrame, label_col: str = 'prediction') -> go.Figure:
    """
    إنشاء رسم بياني دائري (Donut Chart) لتوزيع حركة المرور والهجمات.
    """
    # التأكد من وجود العمود المطلوبة
    target_col = label_col if label_col in df.columns else ('Predicted_Attack' if 'Predicted_Attack' in df.columns else df.columns[-1])
    
    counts = df[target_col].value_counts().reset_index()
    counts.columns = ['Attack_Type', 'Count']
    
    colors = [COLOR_MAP.get(attack, '#94a3b8') for attack in counts['Attack_Type']]
    
    fig = px.pie(
        counts,
        names='Attack_Type',
        values='Count',
        hole=0.45,
        title="🛡️ توزيع حركة مرور الشبكة والتهديدات المكتشفة",
        color='Attack_Type',
        color_discrete_map=COLOR_MAP
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(colors=colors))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter, sans-serif'),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    return fig


def plot_confusion_matrix_heatmap(cm: np.ndarray, class_names: List[str]) -> go.Figure:
    """
    رسم مصفوفة الارتباك (Confusion Matrix) لتقييم دقة النموذج.
    """
    cm_array = np.array(cm)
    
    fig = px.imshow(
        cm_array,
        x=class_names,
        y=class_names,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="التنبؤ المتوقع (Predicted)", y="الحقيقة (Actual Target)", color="العدد"),
        title="🎯 مصفوفة الارتباك وتقييم الدقة (Confusion Matrix)"
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter, sans-serif')
    )
    return fig

# دالة مرادفة لتفادي خطأ تسمية الدوال
def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> go.Figure:
    return plot_confusion_matrix_heatmap(cm, class_names)


def plot_feature_importance(model: Any, feature_names: List[str], top_n: int = 10) -> go.Figure:
    """
    رسم أفق يوضح أهم الخصائص المعتمدة في كشف التهديد من قبل الذكاء الاصطناعي.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    # التأكد من مطابقة طول الخصائص مع المصفوفة
    min_len = min(len(feature_names), len(importances))
    df_imp = pd.DataFrame({
        'Feature': feature_names[:min_len],
        'Importance': importances[:min_len]
    }).sort_values(by='Importance', ascending=True).tail(top_n)

    fig = px.bar(
        df_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f"⭐ أهم {len(df_imp)} خصائص اعتمد عليها النموذج لكشف التهديد",
        color='Importance',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter, sans-serif'),
        coloraxis_showscale=False
    )
    return fig


def plot_protocol_attack_breakdown(df: pd.DataFrame, label_col: str = 'prediction') -> go.Figure:
    """
    رسم بياني مقسم لأنواع الهجمات حسب البروتوكول (TCP, UDP, ICMP).
    """
    target_col = label_col if label_col in df.columns else ('Predicted_Attack' if 'Predicted_Attack' in df.columns else df.columns[-1])

    if 'protocol_type' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="عمود 'protocol_type' غير موجود في البيانات الحالية.", showarrow=False, font=dict(color='#ef4444', size=14))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return fig

    pivot = df.groupby(['protocol_type', target_col]).size().reset_index(name='Count')
    
    fig = px.bar(
        pivot,
        x='protocol_type',
        y='Count',
        color=target_col,
        title="🌐 توزيع التهديدات بحسب بروتوكول الشبكة",
        labels={'protocol_type': 'بروتوكول الشبكة', 'Count': 'عدد الحزم', target_col: 'نوع التهديد'},
        color_discrete_map=COLOR_MAP
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter, sans-serif'),
        barmode='stack'
    )
    return fig


def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    مقارنة أداء النماذج المختلفة (Random Forest vs XGBoost).
    """
    models, metrics, values = [], [], []

    metric_keys = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for model_name, m_data in metrics_dict.items():
        if isinstance(m_data, dict):
            for key, label in zip(metric_keys, metric_labels):
                if key in m_data:
                    models.append(model_name.upper().replace('_', ' '))
                    metrics.append(label)
                    values.append(m_data[key])

    df_metrics = pd.DataFrame({
        'Model': models,
        'Metric': metrics,
        'Value': values
    })

    fig = px.bar(
        df_metrics,
        x='Metric',
        y='Value',
        color='Model',
        barmode='group',
        text_auto='.3f',
        title="📊 مقارنة مقاييس الأداء والدقة بين النماذج",
        color_discrete_sequence=['#38bdf8', '#8b5cf6']
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0', family='Inter, sans-serif'),
        yaxis=dict(range=[0, 1.05])
    )
    return fig
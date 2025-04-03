import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.utils.helpers import CLASS_LABELS, FEATURE_NAMES

def plot_feature_importance(model):
    """Plot feature importance for the selected model"""
    if not hasattr(model, 'named_steps') or not hasattr(model.named_steps['clf'], 'feature_importances_'):
        st.warning("Selected model doesn't support feature importance visualization")
        return
    
    importances = model.named_steps['clf'].feature_importances_
    importance_df = pd.DataFrame({"Feature": FEATURE_NAMES, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance")
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(model, X_test, y_test):
    """Plot confusion matrix for the selected model"""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=CLASS_LABELS,
                yticklabels=CLASS_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    
    return fig

def plot_roc_curve(model, X_test, y_test):
    """Plot ROC curve for the selected model"""
    # Check if model supports probability estimates
    if not hasattr(model, 'predict_proba'):
        st.warning("Selected model does not support probability estimates for ROC curve.")
        return None
        
    # Binarize the output
    y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    
    # Calculate ROC curve and ROC area for each class
    y_score = model.predict_proba(X_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i in range(4):
        ax.plot(fpr[i], tpr[i], label=f"Class {CLASS_LABELS[i]} (AUC = {roc_auc[i]:.2f})")

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    
    return fig

def plot_metrics_comparison(metrics_df):
    """Plot metrics comparison for all models"""
    metrics_df = metrics_df.set_index("Model")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.plot(kind='bar', ax=ax)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_ylim(0.85, 1.0)  # Adjust y-axis for better visualization
    ax.legend(loc="lower right")
    plt.tight_layout()
    
    return fig
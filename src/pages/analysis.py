import streamlit as st
import pandas as pd

from src.components.sidebar import model_selector
from src.utils.helpers import load_models, load_encoders, load_test_data, get_model_metrics, CLASS_LABELS
from src.components.visualizations import plot_feature_importance, plot_confusion_matrix, plot_roc_curve

def show_analysis():
    """Display the analysis page with model metrics and visualizations"""
    
    # Title and description
    st.title("📊 Model Analysis")
    st.markdown("Explore model performance metrics and visualizations.")
    
    # Model selection in sidebar
    with st.sidebar:
        model_name = model_selector()
    
    # Load models and test data
    models = load_models()
    label_encoder, scaler = load_encoders()
    X_test, y_test = load_test_data()
    
    # Get selected model
    model = models[model_name]
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Model Metrics", "🔍 Feature Importance", "🧮 Confusion Matrix", "📉 ROC Curve"])
    
    with tab1:
        st.subheader("📊 Model Performance Summary")
        
        # Get model metrics
        metrics_df = get_model_metrics()
        metrics_df_indexed = metrics_df.set_index("Model")
        
        # Highlight the selected model
        st.dataframe(metrics_df_indexed.style.highlight_max(axis=0))
        
        # Show model details
        st.subheader(f"{model_name} Model Details")
        
        # Find the metrics for the selected model
        model_row = metrics_df[metrics_df["Model"] == model_name]
        
        # Handle case where model isn't in metrics dataframe
        if len(model_row) == 0:
            st.warning(f"No metrics data available for {model_name}")
            model_metrics = {
                "Model": model_name,
                "Accuracy": 0.0,
                "F1-Score": 0.0,
                "ROC-AUC": 0.0
            }
        else:
            model_metrics = {
                "Model": model_name,
                "Accuracy": model_row["Accuracy"].values[0],
                "F1-Score": model_row["F1-Score"].values[0],
                "ROC-AUC": model_row["ROC-AUC"].values[0],
            }
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{model_metrics['Accuracy']:.2f}")
        col2.metric("F1-Score", f"{model_metrics['F1-Score']:.2f}")
        col3.metric("ROC-AUC", f"{model_metrics['ROC-AUC']:.2f}")
        
        st.caption("These metrics are based on cross-validation results.")
    
    with tab2:
        st.subheader("🔍 Feature Importance")
        
        # Try to plot feature importance
        fig = plot_feature_importance(model)
        
        if fig is not None:
            st.pyplot(fig)
            st.markdown("""
            **Feature importance** shows which input parameters have the most significant impact on the model's predictions.
            Higher values indicate more influential features.
            """)
        else:
            st.info(f"The {model_name} model doesn't provide feature importance. Feature importance visualization is typically available for tree-based models like Random Forest and Decision Tree, but not for models like Neural Networks (MLPClassifier) and SVM.")
            
            # Suggest alternative models
            st.markdown("""
            **For feature importance visualization, try using:**
            - Random Forest
            - Decision Tree
            - AdaBoost
            """)
    
    with tab3:
        st.subheader("🧮 Confusion Matrix")
        
        fig = plot_confusion_matrix(model, X_test, y_test)
        st.pyplot(fig)
        
        st.markdown("""
        The **confusion matrix** shows how well the model classifies each category:
        - Rows represent the true classes
        - Columns represent the predicted classes
        - Diagonal cells (top-left to bottom-right) show correct predictions
        - Off-diagonal cells show misclassifications
        """)
        
        # Display class information
        st.markdown("### Quality Class Information")
        for i, label in enumerate(CLASS_LABELS):
            st.markdown(f"**Class {i}**: {label}")
    
    with tab4:
        st.subheader("📉 ROC Curve")
        
        fig = plot_roc_curve(model, X_test, y_test)
        
        if fig is not None:
            st.pyplot(fig)
            st.markdown("""
            The **ROC Curve** (Receiver Operating Characteristic) shows the performance of the classification model:
            - A curve closer to the top-left corner indicates better performance
            - The AUC (Area Under Curve) ranges from 0 to 1, with higher values indicating better performance
            - An AUC of 0.5 represents random guessing (the diagonal line)
            """)
        else:
            st.info(f"The {model_name} model doesn't support probability estimates required for ROC curve visualization.")
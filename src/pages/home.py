import streamlit as st
from src.utils.helpers import get_model_metrics
from src.components.visualizations import plot_metrics_comparison

def show_home():
    """Display the home page with general information and metrics"""
    
    # Title and introduction
    st.title("🏭 Injection Molding Quality Dashboard")
    st.caption("### Most Accurate Model for Predicting Quality Class is Decision Tree")
    
    # Introduction
    st.markdown("""
    This dashboard allows you to predict the quality class of injection molded parts based on process parameters.
    
    Use the sidebar navigation to explore different sections of the dashboard:
    - **Make Prediction**: Enter process parameters to predict quality class
    - **Model Analysis**: Explore model performance metrics and visualizations
    """)
    
    # Display key features
    st.subheader("📋 Key Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Quality Prediction**: Predict the quality class of injection molded parts
        - **Multiple Models**: Compare performance of different ML models
        - **Interactive Parameters**: Adjust process parameters in real-time
        """)
    
    with col2:
        st.markdown("""
        - **Visualization**: View feature importance and model performance
        - **Export Results**: Download prediction results as CSV
        - **Model Metrics**: Compare accuracy, F1-score, and ROC-AUC
        """)
    
    # Display model comparison
    st.subheader("📊 Model Performance Overview")
    metrics_df = get_model_metrics()
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.dataframe(metrics_df.set_index("Model").style.highlight_max(axis=0))
    
    with col2:
        fig = plot_metrics_comparison(metrics_df)
        st.pyplot(fig)
    
    # Quality class explanation
    st.subheader("🏷️ Quality Classes Explained")
    
    quality_classes = {
        "Waste": "Parts that fail quality control and cannot be used.",
        "Target": "Ideal parts that meet all quality requirements perfectly.",
        "Acceptable": "Parts that meet minimum quality requirements and can be used.",
        "Inefficient": "Parts that are usable but produced with suboptimal efficiency."
    }
    
    for class_name, description in quality_classes.items():
        st.markdown(f"**{class_name}**: {description}")
    
    # Footer
    st.markdown("---")
    st.markdown("Developed by: 4127677")
import streamlit as st

def show_home():
    """Display the home page with general information and metrics"""
    
     # Title and Subtitle
    st.title("🏭 Injection Molding Quality Dashboard")
    st.caption("Developed by: 4127677 | Final Year AI Coursework")

    st.markdown("""
    This interactive dashboard allows you to predict the **quality class** of plastic injection molded parts using machine learning models trained on process parameters.  
    It supports multiple classifiers, live predictions, model comparisons, and performance visualizations.
    """)

    # === Project Overview ===
    st.subheader("📌 Project Overview")
    st.markdown("""
    Plastic injection molding is a key manufacturing technique where monitoring process quality is critical.  
    This tool helps operators and engineers classify production parts into:
    - **Waste**
    - **Target**
    - **Acceptable**
    - **Inefficient**

    Based on inputs like temperature, pressure, timing, and torque, this ML-powered system recommends quality class in real-time.
    """)

    # === Data Description ===
    st.subheader("🧪 Data Features Used")
    st.markdown("""
    The following process parameters are used to predict part quality:
    - Cycle time, Mold temperature, Melt temperature
    - Injection and back pressure
    - Torque (mean/peak), Closing and Clamping force
    - Shot volume, Time to fill, Screw position
    """)

    # === Navigation Guide ===
    st.subheader("🧭 How to Use This Dashboard")
    st.markdown("""
    Use the sidebar to access the main sections:
    - **Make Prediction**: Input parameters and get live quality class
    - **Model Analysis**: View feature importance, confusion matrix, ROC curves
    - **Metrics Summary**: Compare performance of models
    - **Export Results**: Download predictions and visual outputs
    """)

    # === Key Features ===
    st.subheader("🚀 Key Features")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - 🧠 Supports multiple ML models (RF, DT, SVM, AdaBoost)
        - 🛠️ Input process parameters via sidebar
        - 🔐 Optional login system with Streamlit Authenticator
        """)

    with col2:
        st.markdown("""
        - 📊 View confusion matrix, ROC curve, and feature importance
        - 📥 Export results as CSV
        - 📈 Live prediction and result explanation
        """)
    
    
    # Footer
     # === Footer ===
    st.markdown("---")
    st.info("This dashboard is part of the Artificial Intelligence Coursework at London South Bank University. Built using Streamlit.")
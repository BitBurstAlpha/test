import streamlit as st
from src.pages.home import show_home
from src.pages.prediction import show_prediction
from src.pages.analysis import show_analysis

# App configuration
st.set_page_config(
    page_title="Injection Molding Quality Dashboard",
    page_icon="🏭",
    layout="wide"
)

# Initialize session state if needed
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Sidebar navigation
with st.sidebar:
    st.image("assets/8_1sasa11.jpg", width=120)
    st.title("Navigation")
    
    if st.button("🏠 Home"):
        st.session_state.page = 'home'
    
    if st.button("🔍 Make Prediction"):
        st.session_state.page = 'prediction'
    
    if st.button("📊 Model Analysis"):
        st.session_state.page = 'analysis'
    
    st.sidebar.markdown("---")
    st.sidebar.caption("Developed by: 4127677")

# Render the selected page
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'prediction':
    show_prediction()
elif st.session_state.page == 'analysis':
    show_analysis()
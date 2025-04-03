import streamlit as st

def get_input_parameters():
    """
    Get input parameters from the sidebar
    Returns a dictionary with all input values
    """
    
    st.header("🔧 Model Inputs")
    
    # Machine Parameters section
    st.markdown("### 🧩 Machine Parameters")
    cycle_time = st.number_input("ZUx - Cycle time", 0.0, 30.0, 15.0)
    plasticizing_time = st.number_input("ZDx - Plasticizing time", 0.0, 20.0, 10.0)
    closing_force = st.number_input("SKx - Closing force", 0.0, 500.0, 250.0)

    # Temperature Settings section
    st.markdown("### 🌡️ Temperature Settings")
    mold_temp = st.number_input("Mold temperature", 0.0, 300.0, 150.0)
    melt_temp = st.number_input("Melt temperature", 0.0, 300.0, 150.0)

    # Pressure & Torque section
    st.markdown("### ⚙️ Pressure & Torque")
    injection_pressure = st.number_input("APVs - Injection pressure peak", 0.0, 3000.0, 1500.0)
    back_pressure = st.number_input("APSs - Back pressure peak", 0.0, 200.0, 100.0)
    clamping_force = st.number_input("SKs - Clamping force peak", 0.0, 1000.0, 500.0)
    torque_mean = st.number_input("Mm - Torque mean", 0.0, 10.0, 5.0)
    torque_peak = st.number_input("Ms - Torque peak", 0.0, 20.0, 10.0)

    # Timings & Volume section
    st.markdown("### ⏱️ Timings & Volume")
    time_to_fill = st.number_input("Time to fill", 0.0, 5.0, 2.5)
    shot_volume = st.number_input("SVo - Shot volume", 0.0, 1000.0, 500.0)
    screw_position = st.number_input("CPn - Screw pos. end of hold", 0.0, 100.0, 50.0)
    
    # Create a dictionary with all inputs
    inputs = {
        'cycle_time': cycle_time,
        'mold_temp': mold_temp,
        'injection_pressure': injection_pressure,
        'time_to_fill': time_to_fill,
        'shot_volume': shot_volume,
        'screw_position': screw_position,
        'plasticizing_time': plasticizing_time,
        'closing_force': closing_force,
        'clamping_force': clamping_force,
        'back_pressure': back_pressure,
        'torque_mean': torque_mean,
        'torque_peak': torque_peak,
        'melt_temp': melt_temp
    }
    
    return inputs

def model_selector():
    """
    Create a model selector dropdown
    Returns the selected model name
    """
    models = ["Random Forest", "Decision Tree", "SVM", "AdaBoost"]
    return st.selectbox("Choose Model", models)
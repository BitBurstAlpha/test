import streamlit as st
import numpy as np

from src.components.sidebar import get_input_parameters, model_selector
from src.utils.helpers import load_models, load_encoders, predict_quality, export_prediction

def show_prediction():
    """Display the prediction page with input parameters and prediction results"""
    
    # Title and description
    st.title("🔍 Quality Class Prediction")
    st.markdown("Enter process parameters in the sidebar and click Predict to see the quality class.")
    
    # Get input parameters from sidebar
    with st.sidebar:
        inputs = get_input_parameters()
        model_name = model_selector()
    
    # Load models and encoders
    models = load_models()
    label_encoder, scaler = load_encoders()
    
    # Model selection
    model = models[model_name]
    
    # Create prediction layout
    st.subheader("Selected Model: " + model_name)
    st.info("This model predicts the quality class based on the input parameters.")
    
    # Predict button
    if st.button("🔍 Predict Quality Class", type="primary"):
        # Prepare input data
        input_data = np.array([[
            inputs['cycle_time'], 
            inputs['mold_temp'], 
            inputs['injection_pressure'], 
            inputs['time_to_fill'], 
            inputs['shot_volume'],
            inputs['screw_position'], 
            inputs['plasticizing_time'], 
            inputs['closing_force'], 
            inputs['clamping_force'],
            inputs['back_pressure'], 
            inputs['torque_mean'], 
            inputs['torque_peak'], 
            inputs['melt_temp']
        ]])
        
        # Make prediction
        pred_label, _ = predict_quality(model, input_data, scaler, label_encoder)
        
        # Display prediction result
        st.subheader("Predicted Quality Class:")
        
        # Style the result based on quality class
        if pred_label == "Target":
            st.success(f"🌟 {pred_label}")
        elif pred_label == "Acceptable":
            st.info(f"✅ {pred_label}")
        elif pred_label == "Inefficient":
            st.warning(f"⚠️ {pred_label}")
        else:  # "Waste"
            st.error(f"❌ {pred_label}")
        
        # Explanation
        st.write(f"Based on the input values, the model predicts the part quality as **{pred_label}**.")
        st.markdown("This result is based on your current process inputs. Adjust values to explore different predictions.")
        
        # Export option
        result_df = export_prediction(model_name, pred_label)
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Prediction", 
            csv, 
            file_name="prediction_result.csv", 
            mime="text/csv"
        )
        
        # Parameter summary
        with st.expander("📋 View Input Parameter Summary"):
            # Convert inputs to DataFrame for better display
            params_df = pd.DataFrame({
                "Parameter": list(inputs.keys()),
                "Value": list(inputs.values())
            })
            st.dataframe(params_df)
    
    # Instructions for users
    with st.expander("ℹ️ How to Use This Page"):
        st.markdown("""
        1. **Adjust Parameters**: Use the sliders in the sidebar to set process parameters
        2. **Select Model**: Choose which ML model to use for prediction
        3. **Make Prediction**: Click the 'Predict Quality Class' button
        4. **Interpret Result**: View the predicted quality class and explanation
        5. **Export**: Download the prediction result as a CSV file if needed
        6. **Iterate**: Adjust parameters and make new predictions to optimize your process
        """)
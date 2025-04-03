import joblib
import numpy as np
import pandas as pd

# Class labels
CLASS_LABELS = ["Waste", "Target", "Acceptable", "Inefficient"]

# Feature names for better display
FEATURE_NAMES = [
    "ZUx - Cycle time", 
    "Mold temperature", 
    "APVs - Specific injection pressure peak value",
    "time_to_fill", 
    "SVo - Shot volume", 
    "CPn - Screw position at the end of hold pressure",
    "ZDx - Plasticizing time", 
    "SKx - Closing force", 
    "SKs - Clamping force peak value",
    "APSs - Specific back pressure peak value", 
    "Mm - Torque mean value current cycle",
    "Ms - Torque peak value current cycle", 
    "Melt temperature"
]

def load_models():
    """Load all ML models from the models directory"""
    best_models = {
        "Random Forest": joblib.load("models/random_forest_model.pkl"),
        "Decision Tree": joblib.load("models/dt_model.pkl"),
        "SVM": joblib.load("models/svm_model.pkl"),
        "ANN": joblib.load("models/ann_model.pkl"),
        "AdaBoost": joblib.load("models/ada_model.pkl")
    }
    return best_models

def load_encoders():
    """Load label encoder and scaler from the models directory"""
    label_encoder = joblib.load("models/label_encoder.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return label_encoder, scaler

def load_test_data():
    """Load test data for model evaluation"""
    X_test = joblib.load("models/X_test.pkl")
    y_test = joblib.load("models/y_test.pkl")
    return X_test, y_test

def predict_quality(model, input_data, scaler, label_encoder):
    """Predict quality class based on input parameters"""
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    pred_index = label_encoder.inverse_transform([pred])[0]
    pred_label = CLASS_LABELS[int(pred_index)]
    return pred_label, pred_index

def export_prediction(model_name, pred_label):
    """Create a dataframe for prediction export"""
    result_df = pd.DataFrame({"Model": [model_name], "Predicted Quality Class": [pred_label]})
    return result_df

def get_model_metrics():
    """Get model performance metrics"""
    results_data = {
        "Model": ["Random Forest", "Decision Tree", "SVM", "ANN"],
        "Accuracy": [0.97, 0.94, 0.93, 0.92],
        "F1-Score": [0.96, 0.93, 0.92, 0.91],
        "ROC-AUC": [0.98, 0.95, 0.94, 0.93]
    }
    return pd.DataFrame(results_data)
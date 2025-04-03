import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# === Title and Branding ===
st.image("assets/8_1sasa11.jpg", width=120)
st.markdown("###  Injection Molding Quality Dashboard")
st.caption("### Most Accurate Model for Predicting Quality Class is Decision Tree")
st.markdown("This dashboard allows you to predict the quality class of injection molded parts based on process parameters.")
st.markdown("Developed by: 4127677")

# === Load saved models ===
best_models = {
    "Random Forest": joblib.load("models/random_forest_model.pkl"),
    "Decision Tree": joblib.load("models/dt_model.pkl"),
    "SVM": joblib.load("models/svm_model.pkl"),
    "ANN": joblib.load("ann_model.pkl"),
    "AdaBoost": joblib.load("models/ada_model.pkl")
}

# === Select model from dropdown ===
model_name = st.selectbox("Choose Model", list(best_models.keys()))
model = best_models[model_name]

# === Load encoder and scaler ===
label_encoder = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

# === Sidebar layout for inputs ===
with st.sidebar:
    st.header("🔧 Model Inputs")
    st.markdown("### 🧩 Machine Parameters")
    cycle_time = st.number_input("ZUx - Cycle time", 0.0, 30.0, 15.0)
    plasticizing_time = st.number_input("ZDx - Plasticizing time", 0.0, 20.0, 10.0)
    closing_force = st.number_input("SKx - Closing force", 0.0, 500.0, 250.0)

    st.markdown("### 🌡️ Temperature Settings")
    mold_temp = st.number_input("Mold temperature", 0.0, 300.0, 150.0)
    melt_temp = st.number_input("Melt temperature", 0.0, 300.0, 150.0)

    st.markdown("### ⚙️ Pressure & Torque")
    injection_pressure = st.number_input("APVs - Injection pressure peak", 0.0, 3000.0, 1500.0)
    back_pressure = st.number_input("APSs - Back pressure peak", 0.0, 200.0, 100.0)
    clamping_force = st.number_input("SKs - Clamping force peak", 0.0, 1000.0, 500.0)
    torque_mean = st.number_input("Mm - Torque mean", 0.0, 10.0, 5.0)
    torque_peak = st.number_input("Ms - Torque peak", 0.0, 20.0, 10.0)

    st.markdown("### ⏱️ Timings & Volume")
    time_to_fill = st.number_input("Time to fill", 0.0, 5.0, 2.5)
    shot_volume = st.number_input("SVo - Shot volume", 0.0, 1000.0, 500.0)
    screw_position = st.number_input("CPn - Screw pos. end of hold", 0.0, 100.0, 50.0)

# === User Instructions ===
st.info("Enter process parameters in the sidebar and click Predict to see the quality class.")

# === Prediction ===
class_labels = ["Waste", "Target", "Acceptable", "Inefficient"]

if st.button("🔍 Predict Quality Class"):
    input_data = np.array([[
        cycle_time, mold_temp, injection_pressure, time_to_fill, shot_volume,
        screw_position, plasticizing_time, closing_force, clamping_force,
        back_pressure, torque_mean, torque_peak, melt_temp
    ]])

    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    pred_index = label_encoder.inverse_transform([pred])[0]
    pred_label = class_labels[int(pred_index)]

    st.subheader("Predicted Quality Class:")
    st.success(f"🌟 {pred_label}")
    st.write(f"Based on the input values, the model predicts the part quality as **{pred_label}**.")
    st.markdown("This result is based on your current process inputs. Adjust values to explore different predictions.")

    # === Export Prediction ===
    result_df = pd.DataFrame({"Model": [model_name], "Predicted Quality Class": [pred_label]})
    csv = result_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Prediction", csv, file_name="prediction_result.csv", mime="text/csv")

# === Feature Importance (optional) ===
if st.checkbox("📊 Show Feature Importance"):
    importances = model.named_steps['clf'].feature_importances_
    features = [
        "ZUx - Cycle time", "Mold temperature", "APVs - Specific injection pressure peak value",
        "time_to_fill", "SVo - Shot volume", "CPn - Screw position at the end of hold pressure",
        "ZDx - Plasticizing time", "SKx - Closing force", "SKs - Clamping force peak value",
        "APSs - Specific back pressure peak value", "Mm - Torque mean value current cycle",
        "Ms - Torque peak value current cycle", "Melt temperature"
    ]
    importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df["Feature"], importance_df["Importance"])
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance - Random Forest")
    st.pyplot(fig)

# === Confusion Matrix (optional) ===
if st.checkbox("📉 Show Confusion Matrix"):
    y_test = joblib.load("models/y_test.pkl")
    X_test = joblib.load("models/X_test.pkl")
    y_pred_conf = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_conf)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_labels,
                yticklabels=class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# === Model Metrics Summary ===
if st.checkbox("📈 Show Model Metrics Summary"):
    st.subheader("📊 Model Performance Summary")

    results_data = {
        "Model": ["Random Forest", "Decision Tree", "SVM", "ANN"],
        "Accuracy": [0.97, 0.94, 0.93, 0.92],
        "F1-Score": [0.96, 0.93, 0.92, 0.91],
        "ROC-AUC": [0.98, 0.95, 0.94, 0.93]
    }

    results_df = pd.DataFrame(results_data)
    results_df.set_index("Model", inplace=True)

    st.dataframe(results_df.style.highlight_max(axis=0))
    st.markdown("Note: These metrics are based on cross-validation results.")
    st.markdown("You can explore different models using the dropdown above.")

# === ROC Curve (Optional) ===
if st.checkbox("🧠 Show ROC Curve"):
    y_test = joblib.load("models/y_test.pkl")
    X_test = joblib.load("models/X_test.pkl")

    y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    else:
        st.warning("Selected model does not support probability estimates.")
        y_score = None

    if y_score is not None:
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(4):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig, ax = plt.subplots()
        for i in range(4):
            ax.plot(fpr[i], tpr[i], label=f"Class {class_labels[i]} (AUC = {roc_auc[i]:.2f})")

        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        st.pyplot(fig)
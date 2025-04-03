# 🏭 Injection Molding Quality Dashboard

A Streamlit-based interactive dashboard to predict the **quality class** of injection-molded parts using machine learning models.

## 📌 Features

- 🔢 Input process parameters through a user-friendly sidebar
- 🤖 Predicts part quality using models like:
  - Random Forest
  - Decision Tree
  - SVM
  - AdaBoost
- 📈 Visualizations:
  - Feature Importance
  - Confusion Matrix
  - ROC Curve
- 📥 Export prediction results as CSV

## 🧠 Quality Classes

| Class Label | Meaning       |
|-------------|---------------|
| 0           | Waste         |
| 1           | Target        |
| 2           | Acceptable    |
| 3           | Inefficient   |

## 📁 Project Structure


## 🚀 Deployment

This app is deployed using **Streamlit Cloud**.  
👉 [Live App URL] (https://injection-molding-quality-dashboard.streamlit.app/)

## 🛠️ Installation (Local)

```bash
git clone git@github.com:BitBurstAlpha/Plastic_injection_moulding_Dashboard.git
cd your-repo
pip install -r requirements.txt
streamlit run app.py


---

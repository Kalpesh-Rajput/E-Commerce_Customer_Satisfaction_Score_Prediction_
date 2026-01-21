# 📊 Customer Satisfaction (CSAT) Prediction System

## 🔍 Project Overview

[Click Here](https://e-commercecustomersatisfactionscoreprediction-n6cqwmednyah43dz.streamlit.app/)

This project predicts **Customer Satisfaction (CSAT) levels** for an e-commerce platform using customer service interaction data.
A **Neural Network model** is trained on historical interactions and deployed as an **interactive Streamlit web application**.

The system helps identify how service channels, interaction categories, agents, shifts, and management impact customer satisfaction.

---

## 🎯 Objectives

* Analyze key drivers of customer satisfaction
* Predict CSAT levels using a trained neural network
* Deploy a real-time prediction system using Streamlit
* Ensure production-grade preprocessing consistency

---

## 🗂️ Project Structure

```
project_root/
│── app.py                 # Streamlit application
│── csat_model.keras       # Trained neural network model
│── encoder.pkl            # Saved OneHotEncoder (training pipeline)
│── scaler.pkl             # Saved scaler
│── requirements.txt       # Required Python dependencies
│── README.md              # Project documentation
```

---

## 📁 Dataset Description

The dataset includes customer interaction details such as:

* Service Channel
* Interaction Category and Sub-Category
* Agent Name, Manager, and Shift
* Customer Satisfaction (CSAT) Score

Both **categorical and numerical features** are used to train the model.

---

## 🧹 Data Preprocessing

To ensure accurate predictions, the following preprocessing steps were applied:

### Missing Value Handling

* Columns with excessive missing values were removed
* Text-based categorical fields were imputed where required
* Rows with missing unique identifiers were dropped

### Categorical Encoding

* Categorical features were encoded using **OneHotEncoder**
* The trained encoder was saved as `encoder.pkl`
* The same encoder is reused during deployment to avoid feature mismatch

### Feature Scaling

* Numeric features were scaled using a fitted scaler
* The scaler was saved and reused during inference

---

## 📊 Exploratory Data Analysis (EDA)

Key analyses performed:

* CSAT score distribution
* CSAT variation across service channels
* Low CSAT interaction categories
* Agent shift and tenure impact
* Manager and supervisor performance
* CSAT trends over time

Multiple visualizations (bar charts, box plots, histograms, time-series plots) were used to derive business insights.

---

## 🤖 Model Development

* **Model Type:** Neural Network (Keras / TensorFlow)
* **Architecture:** Dense layers with ReLU activation
* **Output:** Softmax classification for CSAT levels (1–5)

### Regularization Techniques

* Dropout layers
* L2 regularization
* EarlyStopping
* ReduceLROnPlateau

These techniques help prevent overfitting and improve generalization.

---

## 📈 Model Performance

* Training Accuracy: ~70%
* Validation Accuracy: ~69%
* Test Accuracy: ~69%

The small gap between training and validation accuracy indicates stable generalization.

---

## 🚀 Streamlit Deployment

The trained model is deployed using **Streamlit**, allowing users to:

* Select interaction details through a clean UI
* Generate CSAT predictions in real time
* View results using interactive gauge and confidence charts

### Why Encoder Reuse Matters

Neural networks require **exactly the same input feature structure** during inference as during training.
The saved `encoder.pkl` ensures feature alignment and prevents input shape mismatch errors.

---

## 🛠️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow / Keras
* Streamlit
* Plotly

---

## ▶️ How to Run the Application

### 1. Install Dependencies

```bash
uv add -r requirements.txt
```

### 2. Run Streamlit App

```bash
uv run streamlit run app.py
```

---

## 🧠 Key Takeaways

* Consistent preprocessing is critical for ML deployment
* Reusing trained encoders avoids inference failures
* Streamlit enables rapid deployment of ML models
* The system provides actionable insights into customer satisfaction

---

## 🔮 Future Enhancements

* Bulk prediction using CSV upload
* Model comparison with tree-based algorithms
* Real-time database integration
* Advanced explainability using SHAP

---

## ✅ Conclusion

This project demonstrates a **complete machine learning lifecycle** — from data analysis and model training to **production-ready deployment**.
The CSAT Prediction System provides both **business insights** and a **scalable prediction framework**.

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import time

import sys
import os
sys.path.append(os.path.abspath("app"))

from custom_preprocessor import CustomPreprocessor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

st.title("Churn Prediction App")

# --- Load original dataset for column names ---
@st.cache_data
def cargar_dataset():
    return pd.read_csv("data/telco_churn.csv")

df = cargar_dataset()

# --- Load test set ---
@st.cache_data
def cargar_test_set():
    df_test = pd.read_csv("data/test_set.csv")
    X = df_test.drop(columns=["Churn"])
    y = df_test["Churn"]
    return X, y

X_test, y_test = cargar_test_set()

# --- Load preprocessor ---
@st.cache_resource
def cargar_preprocesador():
    df = pd.read_csv("data/telco_churn.csv")
    X = df.drop(columns=["Churn"])
    preprocessor = CustomPreprocessor()
    preprocessor.fit(X)
    return preprocessor

preprocessor = cargar_preprocesador()

# --- Available models dictionary ---
modelos_disponibles = {
    "Logistic Regression (Full)": "models/logistic_full.pkl",
    "Logistic Regression (Top Features)": "models/logistic_top.pkl",
    "Random Forest": "models/random_forest_full.pkl",
    "CatBoost": "models/catboost_full.pkl"
}

# --- Sidebar ---
st.sidebar.header("Model Selection")
modelo_seleccionado = st.sidebar.selectbox("Choose a model to predict Churn", list(modelos_disponibles.keys()))
modelo_path = modelos_disponibles[modelo_seleccionado]

# Flag: Are we using the reduced model?
modelo_reducido = "logistic_top" in modelo_path

# --- Load selected model ---
@st.cache_resource
def cargar_modelo(path):
    return joblib.load(path)

modelo = cargar_modelo(modelo_path)
modelo_tamano_bytes = os.path.getsize(modelo_path)
modelo_tamano_kb = modelo_tamano_bytes / 1024
modelo_tamano_str = f"{modelo_tamano_kb:.2f} KB" if modelo_tamano_kb < 1024 else f"{modelo_tamano_kb/1024:.2f} MB"
st.sidebar.success(f"Model loaded successfully: {modelo_seleccionado}\n\nModel size: {modelo_tamano_str}")

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.markdown(f"### Model Metrics and Confusion Matrix")

# --- Transform data ---
if modelo_reducido:
    top_features = joblib.load("models/top_features.pkl")
    X_test_transformado = preprocessor.transform(X_test)
    X_test_transformado = X_test_transformado[top_features]
else:
    X_test_transformado = preprocessor.transform(X_test)

# --- Prediction and metrics ---
y_pred = modelo.predict(X_test_transformado)
y_prob = modelo.predict_proba(X_test_transformado)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
f1 = f1_score(y_test, y_pred)

# --- Create two columns ---
col1, col2 = st.columns(2)

# --- Model Metrics ---
with col1:
    fig1, ax1 = plt.subplots(figsize=(4, 3.5))
    metrics = {"Accuracy": accuracy, "ROC AUC": roc_auc, "F1 Score": f1}
    ax1.barh(list(metrics.keys()), list(metrics.values()), color="skyblue")

    for i, (name, value) in enumerate(metrics.items()):
        ax1.text(
            x=value / 2, y=i,
            s=f"{value:.2%}",
            va='center', ha='center',
            color='black', fontsize=10, weight='bold'
        )

    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Score", fontsize=10)
    ax1.set_title("Model Metrics", fontsize=11)
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    fig1.tight_layout()
    st.pyplot(fig1)

# --- Confussion Matrix ---
with col2:
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax2, colorbar=False)
    ax2.set_title("Confusion Matrix", fontsize=11)
    fig2.tight_layout()
    st.pyplot(fig2)

st.markdown("### Feature Importance from Random Forest")

# --- Load Random Forest ---
modelo_rf = joblib.load("models/random_forest_full.pkl")

# --- Load preprocessor for column names ---
feature_names = preprocessor.feature_names

# --- Get importances ---
importancias = modelo_rf.feature_importances_

# --- Ordered DataFrame ---
df_importancia = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importancias
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

# --- Prepare smoothed curve ---
x = np.arange(len(df_importancia))
y = df_importancia["Importance"].values

fig, ax = plt.subplots(figsize=(7, max(3.5, len(df_importancia) * 0.25)))
ax.barh(df_importancia["Feature"], df_importancia["Importance"], color="skyblue")

if len(x) > 3:
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spline = make_interp_spline(x, y, k=3)
    y_smooth = spline(x_smooth)
    ax.plot(y_smooth, x_smooth, color='crimson', linewidth=2)
else:
    ax.plot(y, x, color='crimson', linewidth=2)

# --- Style ---
ax.set_title("Feature Importance (Random Forest)", fontsize=11)
ax.set_xlabel("Importance", fontsize=10)
ax.tick_params(labelsize=8)
ax.invert_yaxis()

fig.tight_layout()
st.pyplot(fig)

st.markdown("""
#### Top Features:

1. **TotalCharges:** Represents the total amount billed to the customer since they joined the service. It is a strong indicator of customer longevity and value.  
2. **tenure:** Number of months the customer has been with the company. Customers with shorter tenure tend to have a higher probability of churning, as shown in the EDA.  
3. **MonthlyCharges:** Monthly amount paid by the customer. This is related to the perceived value or cost of the service.  
4. **Contract_Month-to-month:** Indicates whether the customer is on a month-to-month contract. This type of contract typically has higher churn rates due to the lack of long-term commitment.  
5. **OnlineSecurity_No:** Customers who do not have online security. The absence of value-added services may increase dissatisfaction.
""")

st.subheader("New Observation Input")

user_input = {}

if modelo_reducido:
    st.info("This model requires only 5 key fields")

    cols = st.columns(3)

    with cols[0]:
        user_input["TotalCharges"] = st.number_input("Total Charges", min_value=0.0, step=0.1)
    with cols[1]:
        user_input["tenure"] = st.number_input("Tenure (months)", min_value=0, step=1)
    with cols[2]:
        user_input["MonthlyCharges"] = st.number_input("Monthly Charges", min_value=0.0, step=0.1)

    cols2 = st.columns(2)
    with cols2[0]:
        user_input["Contract"] = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    with cols2[1]:
        user_input["OnlineSecurity"] = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
else:
    st.info("This model requires all original columns")

    # Group numeric inputs in rows of 3
    num_cols = preprocessor.num_cols
    for i in range(0, len(num_cols), 3):
        cols = st.columns(3)
        for j, col in enumerate(num_cols[i:i+3]):
            with cols[j]:
                user_input[col] = st.number_input(f"{col}", value=0.0, step=0.1)

    # Group categorical inputs in rows of 3
    cat_cols = preprocessor.cat_cols
    for i in range(0, len(cat_cols), 3):
        cols = st.columns(3)
        for j, col in enumerate(cat_cols[i:i+3]):
            with cols[j]:
                opciones = sorted(df[col].dropna().unique().tolist())
                user_input[col] = st.selectbox(f"{col}", opciones)

if st.button("Predict Churn"):
    # Check if user input is empty
    columnas_esperadas = preprocessor.num_cols + preprocessor.cat_cols
    entrada_completa = {col: user_input.get(col, None) for col in columnas_esperadas}
    df_nueva_obs = pd.DataFrame([entrada_completa])

    # Check if all required columns are present
    if modelo_reducido:
        top_features = joblib.load("models/top_features.pkl")
        X_transformada = preprocessor.transform(df_nueva_obs)
        X_transformada = X_transformada[top_features]
    else:
        X_transformada = preprocessor.transform(df_nueva_obs)

    # --- Make prediction ---
    start = time.time()
    pred_clase = modelo.predict(X_transformada)[0]
    pred_prob = modelo.predict_proba(X_transformada)[0][1]
    end = time.time()
    tiempo_pred = end - start

    # --- Display results ---
    st.info(f"Churn probability: {pred_prob:.2%}")
    st.success(f"Prediction: {'Churn' if pred_clase == 1 else 'No Churn'}")
    st.caption(f"Execution time: {tiempo_pred:.4f} seconds")
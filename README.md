# Telco Customer Churn - Streamlit App

This Streamlit application allows you to explore a customer churn dataset and interactively predict whether a customer is likely to churn based on different machine learning models.

---

## Project Structure

```
project/
│
├── data/                         # Contains datasets
│   ├── telco_churn.csv
│   └── test_set.csv
│
├── models/                       # Contains trained models and top features
│   ├── logistic_full.pkl
│   ├── logistic_top.pkl
│   ├── random_forest_full.pkl
│   ├── catboost_full.pkl
│   └── top_features.pkl
│
├── pages/                        # Streamlit multipage system
│   └── Prediction.py             # Prediction interface
│
├── custom_preprocessor.py        # Contains the custom preprocessor
├── EDA.py                        # Main Exploratory Data Analysis interface
```

---

## How to Run the App Locally

### 1. Clone this repository

```bash
git clone https://github.com/ByronVelasco/StreamlitApp.git
cd StreamlitApp
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

---

### 4. Run the app

Make sure you're in the project root directory (same level as `EDA.py`):

```bash
streamlit run EDA.py
```

---

## Available Pages

- **EDA**: Explore the dataset with summary statistics, distributions, and churn proportions.
- **Prediction**: Choose a model and make predictions using new inputs or evaluate it on test data.

You can switch between pages using the sidebar.

---

## Models Included

| Model                             | Description                          |
|----------------------------------|--------------------------------------|
| Logistic Regression (Full)       | Trained on all features              |
| Logistic Regression (Top)        | Trained on 5 most important features |
| Random Forest                    | Full-feature Random Forest model     |
| CatBoost                         | CatBoost classifier with tuning      |

---

## Notes

- The `custom_preprocessor.py` handles preprocessing: imputation, encoding, and scaling.
- Models were trained with `GridSearchCV` and `RandomizedSearchCV` and saved with `joblib`.
- All predictions are made after applying the same preprocessing pipeline.

---

## Explanation of the Web Application

This Streamlit web application is organized into two interactive pages:

---

### First Page – **EDA (Exploratory Data Analysis)**

This section allows users to explore the dataset in detail before using machine learning models:

- **About Dataset**: General context and description of the Telco Customer Churn dataset, including variable explanations.
- **Dataset Preview**: Displays the first rows of the dataset for quick inspection.
- **Descriptive Statistics by Variable**: Allows the user to select any column and view statistical summaries (`count`, `mean`, `std`, etc.).
- **Distribution of the Selected Column by `Churn`**: Users can select a column to visualize its distribution segmented by the `Churn` target (e.g., histograms or bar plots).
- **Distribution of the Target Variable: `Churn`**: Visualizes the proportion of churn vs. non-churn customers both numerically and graphically.

---

### Second Page – **Prediction**

This section allows the user to interact with machine learning models and perform predictions:

- **Model Selection**: Choose one of the following trained models:
  - `Logistic Regression (Full)`
  - `Logistic Regression (Top Features)`
  - `Random Forest`
  - `CatBoost`
  
  The sidebar also displays the **file size** of the selected model.

- **Model Metrics and Confusion Matrix**: Displays performance metrics (`Accuracy`, `ROC AUC`, `F1 Score`) and the confusion matrix based on the test set.

- **Feature Importance from Random Forest**: Visual explanation of the 5 most important features identified by the Random Forest model:
  1. `TotalCharges`
  2. `tenure`
  3. `MonthlyCharges`
  4. `Contract_Month-to-month`
  5. `OnlineSecurity_No`

- **New Observation Input**: Dynamic form that adjusts depending on the selected model. The user can enter values for a new customer.

- **Predict Churn Button**: After submitting the input, the app:
  - Predicts the **churn probability**
  - Returns the predicted class: **Churn** or **No Churn**
  - Displays the **execution time** of the prediction

---

## Author

Byron Velasco – Maestría en Ciencia de Datos – 2025
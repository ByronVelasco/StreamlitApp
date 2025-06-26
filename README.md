# Telco Customer Churn - Streamlit App

This Streamlit application allows you to explore a customer churn dataset and interactively predict whether a customer is likely to churn based on different machine learning models.

---

## Project Structure

```
project/
│
├── app/                          # Contains the custom preprocessor
│   └── custom_preprocessor.py
│
├── data/                         # Contains datasets
│   ├── telco_churn.csv
│   └── test_set.csv
│
├── models/                       # Contains trained models and top features
│   ├── preprocessor.pkl
│   ├── logistic_full.pkl
│   ├── logistic_top.pkl
│   ├── random_forest_full.pkl
│   ├── catboost_full.pkl
│   └── top_features.pkl
│
├── pages/                        # Streamlit multipage system
│   └── Prediction.py             # Prediction interface
│
├── image.png                     # Header image for the EDA page
├── EDA.py                        # Main Exploratory Data Analysis interface
```

---

## How to Run the App Locally

### 1. Clone this repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
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

## Author

Byron Velasco – Maestría en Ciencia de Datos – 2025
import pandas as pd
import joblib
from app.custom_preprocessor import CustomPreprocessor

df = pd.read_csv("data/telco_churn.csv")
X = df.drop(columns=["Churn"])

preprocessor = CustomPreprocessor()
preprocessor.fit(X)

joblib.dump(preprocessor, "models/preprocessor.pkl")
print("Preprocessor re-exported")
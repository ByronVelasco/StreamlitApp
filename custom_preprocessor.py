import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class CustomPreprocessor(BaseEstimator, TransformerMixin):
	def __init__(self):
		self.pipeline = None
		self.feature_names = None

	def fit(self, X, y=None):
		X = X.copy()
		
		# Detectar columnas numéricas y categóricas
		self.num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
		self.cat_cols = X.select_dtypes(include=['object']).columns.tolist()

		# Pipeline para numéricas
		num_pipeline = Pipeline([
				('imputer', SimpleImputer(strategy='median')),
				('scaler', RobustScaler())
		])

		# Pipeline para categóricas
		cat_pipeline = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

		# Combinación de pipelines
		self.pipeline = ColumnTransformer([
				('num', num_pipeline, self.num_cols),
				('cat', cat_pipeline, self.cat_cols)
		])

		# Ajustar pipeline
		self.pipeline.fit(X)

		# Guardar nombres de columnas finales
		cat_feature_names = self.pipeline.named_transformers_['cat'].get_feature_names_out(self.cat_cols)
		self.feature_names = self.num_cols + list(cat_feature_names)
		
		return self

	def transform(self, X, y=None):
		X = X.copy()
		X_transformed = self.pipeline.transform(X)
		return pd.DataFrame(X_transformed, columns=self.feature_names, index=X.index)
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

class MakeBMI(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['bmi'] = X['Weight'] / (X['Height'] * X['Height']) * 10
        X['bmi'] = X['bmi'].astype(float)
        return X

class MTRANSTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.unique_values_ = ['Bike', 'Motorbike', 'Public_Transportation', 'Walking']
        return self
    def transform(self, X):
        X = X.copy()
        for val in self.unique_values_:
            X[f'MTRANS_{val}'] = (X['MTRANS'] == val).astype(int)
        X.drop('MTRANS', axis=1, inplace=True)
        return X

class CALCTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['CALC'].replace({'Sometimes': 1, 'Frequently': 2, 'Always': 3, 'No': 0}, inplace=True)
        X['CALC'] = X['CALC'].astype(int)
        return X

class CAECTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X['CAEC'].replace({'Sometimes': 1, 'Frequently': 2, 'Always': 3, 'No': 0}, inplace=True)
        X['CAEC'] = X['CAEC'].astype(int)
        return X

class GenderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.unique_values_ = ['Male']
        return self
    def transform(self, X):
        X = X.copy()
        for val in self.unique_values_:
            X[f'Gender_{val}'] = (X['Gender'] == val).astype(int)
        X.drop('Gender', axis=1, inplace=True)
        return X


class ScaleContinuous(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.continuous_dict = {
            'Age': [14.000000, 61.000000],
            'Height': [1.450000, 1.975663],
            'Weight': [39.000000, 165.057269],
            'FCVC': [1.000000, 3.000000],
            'NCP': [1.000000, 4.000000],
            'CH2O': [1.000000, 3.000000],
            'FAF': [0.000000, 3.000000],
            'TUE': [0.000000, 2.000000],
            'bmi': [128.68540707483768, 549.9799136125479]
        }
        return self
    def transform(self, X):
        X = X.copy()
        continuous_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'bmi']
        for feat in continuous_features:
            X[feat] = (X[feat] - self.continuous_dict[feat][0]) / (self.continuous_dict[feat][1] - self.continuous_dict[feat][0])
        return X

# ----------------------------------------------------- ##
class Model:
    def __init__(self, ppl_path, model_path):
        self.ppl_path = ppl_path
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        with open(self.ppl_path, 'rb') as f:
            self.ppl = pickle.load(f)

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict(self, input_df):
        predictions = self.model.predict(input_df)
        return predictions
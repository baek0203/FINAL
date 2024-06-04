import pickle
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


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
            X[feat] = (X[feat] - self.continuous_dict[feat][0]) / (
                        self.continuous_dict[feat][1] - self.continuous_dict[feat][0])
        return X


class Model:
    def __init__(self, pipeline_path: str, model_path: str):
        self.pipeline_path = pipeline_path
        self.model_path = model_path
        self.pipeline = self.load_pipeline()
        self.model = self.load_model()

    def load_pipeline(self):
        with open(self.pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
        return pipeline

    def load_model(self):
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def predict(self, data: pd.DataFrame):
        transformed_data = self.pipeline.transform(data)
        predictions = self.model.predict(transformed_data)
        return predictions

# Example usage:
# model = Model('ppl.pkl', 'model.pickle')
# new_data = pd.DataFrame({ ... })  # Your new data here
# predictions = model.predict(new_data)

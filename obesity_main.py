import streamlit as st
import pandas as pd
from obesity_model import Model
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



# Initialize the model
model = Model('ppl.pkl', 'model.pickle')

# Streamlit app
st.title('Obesity Prediction')

# Collect user input
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=0, max_value=100, value=25)
height = st.number_input('Height (m)', min_value=0.0, max_value=3.0, value=1.75)
weight = st.number_input('Weight (kg)', min_value=0.0, max_value=300.0, value=70.0)
family_history_with_overweight = st.selectbox('Family History with Overweight', [0, 1])
favc = st.selectbox('Frequent Consumption of High Caloric Food', [0, 1])
fcvc = st.number_input('Frequency of Consumption of Vegetables (FCVC)', min_value=1.0, max_value=3.0, value=2.0)
ncp = st.number_input('Number of Main Meals (NCP)', min_value=1.0, max_value=4.0, value=3.0)
caec = st.selectbox('Consumption of Food Between Meals (CAEC)', ['No', 'Sometimes', 'Frequently', 'Always'])
ch2o = st.number_input('Consumption of Water (CH2O)', min_value=1.0, max_value=3.0, value=2.0)
scc = st.selectbox('Monitoring of Caloric Intake', [0, 1])
faf = st.number_input('Physical Activity Frequency (FAF)', min_value=0.0, max_value=3.0, value=1.0)
tue = st.number_input('Time Using Technology Devices (TUE)', min_value=0.0, max_value=2.0, value=1.0)
calc = st.selectbox('Consumption of Alcohol (CALC)', ['No', 'Sometimes', 'Frequently', 'Always'])
mtrans = st.selectbox('Transportation Used', ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking'])

# Create a dictionary with the input data
input_data = {
    'Gender': [gender],
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'family_history_with_overweight': [family_history_with_overweight],
    'FAVC': [favc],
    'FCVC': [fcvc],
    'NCP': [ncp],
    'CAEC': [caec],
    'CH2O': [ch2o],
    'SCC': [scc],
    'FAF': [faf],
    'TUE': [tue],
    'CALC': [calc],
    'MTRANS': [mtrans]
}

# Convert the input data to a DataFrame
input_df = pd.DataFrame(input_data)

# Display the input data for verification
st.write('Input Data')
st.dataframe(input_df)

# Make a prediction when the button is pressed
if st.button('Predict'):
    predictions = model.predict(input_df)
    st.write('Predictions')
    st.write(predictions)

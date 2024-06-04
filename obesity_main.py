import streamlit as st
import pandas as pd
import obesity_model
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from obesity_model import Model

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

    #######################################################################


# Streamlit 애플리케이션 설정
st.title("비만도 진단")

# 성별 선택 라디오 버튼 추가
gender_option = st.radio('성별을 선택하세요', ('남성', '여성'))
Gender = gender_option

# 사용자로부터 나이, 키, 몸무게 입력 받기
age = st.number_input('나이', min_value=0, max_value=120, value=25)
height = st.number_input('키 (cm)', min_value=0, max_value=250, value=170)
weight = st.number_input('몸무게 (kg)', min_value=0, max_value=200, value=70)

# 추가 질문
family_history = st.radio('과체중이나 비만을 겪은 가족원이 있나요?', ('예', '아니오'))
high_calorie_food = st.radio('고칼로리 음식을 빈번하게 먹나요?', ('예', '아니오'))
vegetable_intake = st.radio('식사할 때 보통 야채를 먹나요?', ('그렇지 않다', '가끔', '항상'))
meal_frequency = st.radio('하루에 식사를 몇 번 하나요?', ('1~2번', '3번', '3번 이상'))
snacking_frequency = st.radio('식사 사이에 음식을 얼마나 섭취하나요?', ('하지 않음', '가끔', '자주', '항상'))
water_intake = st.radio('하루에 물을 얼마나 마시나요?', ('1L 미만', '1~2L', '2L 이상'))
calorie_check = st.radio('매일 섭취한 칼로리를 확인하나요?', ('예', '아니오'))
physical_activity = st.radio('신체 활동(운동)을 얼마나 자주 하나요?', ('하지 않음', '1~2일', '2~4일', '4~5일'))
screen_time = st.radio('스마트폰, 비디오게임, TV, 컴퓨터 등의 전자기기를 얼마나 사용하나요?', ('0~2시간', '3~5시간', '5시간 이상'))
alcohol_consumption = st.radio('술을 얼마나 자주 마시나요?', ('마시지 않음', '가끔', '자주', '매일'))
transportation_mode = st.radio('어떤 교통수단을 주로 사용하나요?', ('자가용', '오토바이', '자전거', '대중교통', '도보'))

if st.button('제출'):
    # 입력된 데이터 저장
    input_data = {
        'Gender': [Gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [family_history],
        'FAVC': [high_calorie_food],
        'FCVC': [vegetable_intake],
        'NCP': [meal_frequency],
        'CAEC': [snacking_frequency],
        'CH2O': [water_intake],
        'SCC': [calorie_check],
        'FAF': [physical_activity],
        'TUE': [screen_time],
        'CALC': [alcohol_consumption],
        'MTRANS': [transportation_mode]
    }
    # 데이터 프레임으로 변환
    df = pd.DataFrame(input_data)

    # Model 클래스 인스턴스를 생성하고 예측을 수행합니다.
    model = Model('ppl.pkl', 'model.pickle')
    prediction = model.predict(df)

    # 결과 출력
    st.write(f"예측된 비만도 수준: {prediction[0]}")
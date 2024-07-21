import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data with st.cache_data
@st.cache_data
def load_data():
    data = pd.read_csv('insurance (1).csv')
    return data

data = load_data()

# Display university logo
st.image('/home/alexmendes/insurance_price_predict/iu.png', width=200)

# Title
st.title('Insurance Charges Prediction')

# Exploratory Data Analysis
st.subheader('Exploratory Data Analysis')
st.write(data.head())

# Preprocessing
st.subheader('Preprocessing')
# (Add your preprocessing code here)

# Model Training
st.subheader('Model Training')
# Split data into training and testing sets
X = data.drop('charges', axis=1)
y = data['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Train Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

st.write('Models trained successfully!')

# Model Evaluation
st.subheader('Model Evaluation')
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

lr_r2 = r2_score(y_test, lr_pred)
rf_r2 = r2_score(y_test, rf_pred)

st.write('Linear Regression RMSE: ', lr_rmse)
st.write('Random Forest RMSE: ', rf_rmse)
st.write('Linear Regression R²: ', lr_r2)
st.write('Random Forest R²: ', rf_r2)

# Prediction
st.subheader('Make a Prediction')
if st.button('Predict Charges'):
    # (Add your prediction code here)
    st.write('Prediction: ...')



import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Title and subtitle
st.title("Insurance Charges Prediction")
st.subheader("Introduction to Data Science")
st.subheader("Task 3: Machine Learning")
st.write("Student: Alex Arnold de Almeida Mendes")

# Display logo
st.image("/home/alexmendes/insurance_price_predict/iu.png", width=200)

# Load and display data
st.header("Exploratory Data Analysis")
df = pd.read_csv("insurance.csv")
st.write(df.head())

# Plot histogram for numerical features
st.subheader("Histograms of Numerical Features")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Histogram for 'age'
sns.histplot(df['age'], ax=axes[0], kde=True, color='skyblue')
axes[0].set_title('Age Distribution')

# Histogram for 'bmi'
sns.histplot(df['bmi'], ax=axes[1], kde=True, color='salmon')
axes[1].set_title('BMI Distribution')

# Histogram for 'charges'
sns.histplot(df['charges'], ax=axes[2], kde=True, color='green')
axes[2].set_title('Charges Distribution')

st.pyplot(fig)

# Plot scatter plots for relationships between features
st.subheader("Scatter Plots")
fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# Scatter plot for 'bmi' vs 'charges'
sns.scatterplot(x='bmi', y='charges', data=df, ax=axes[0], color='purple')
axes[0].set_title('BMI vs Charges')

# Scatter plot for 'age' vs 'charges'
sns.scatterplot(x='age', y='charges', data=df, ax=axes[1], color='orange')
axes[1].set_title('Age vs Charges')

st.pyplot(fig)

# Preprocessing
st.header("Preprocessing")
st.write("Handling categorical data with One-Hot Encoding")

df_processed = pd.get_dummies(df, drop_first=True)
st.write(df_processed.head())

# Split data
X = df_processed.drop("charges", axis=1)
y = df_processed["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
st.header("Model Training")
lr_model = LinearRegression()
rf_model = RandomForestRegressor(random_state=42)

lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

st.write("Models trained successfully!")

# Model evaluation
st.header("Model Evaluation")
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

lr_r2 = r2_score(y_test, lr_pred)
rf_r2 = r2_score(y_test, rf_pred)

# Function to style the values
def style_value_rmse(value, best_value):
    """Style the RMSE values based on the best value."""
    return 'green' if value <= best_value else 'red'

def style_value_r2(value, best_value):
    """Style the R² values based on the best value."""
    return 'green' if value >= best_value else 'red'

# Get colors for styling
lr_rmse_color = style_value_rmse(lr_rmse, rf_rmse)
rf_rmse_color = style_value_rmse(rf_rmse, rf_rmse)

lr_r2_color = style_value_r2(lr_r2, rf_r2)
rf_r2_color = style_value_r2(rf_r2, rf_r2)

# Display styled values
st.write("### Model Performance")

st.markdown(f"**Linear Regression RMSE:** <span style='color:{lr_rmse_color};'>{lr_rmse:.2f}</span>", unsafe_allow_html=True)
st.markdown(f"**Random Forest RMSE:** <span style='color:{rf_rmse_color};'>{rf_rmse:.2f}</span>", unsafe_allow_html=True)

st.markdown(f"**Linear Regression R²:** <span style='color:{lr_r2_color};'>{lr_r2:.2f}</span>", unsafe_allow_html=True)
st.markdown(f"**Random Forest R²:** <span style='color:{rf_r2_color};'>{rf_r2:.2f}</span>", unsafe_allow_html=True)

# User input for prediction
st.header("Make a Prediction")

age = st.number_input("Age", min_value=18, max_value=100, value=25)
sex = st.selectbox("Sex", options=["male", "female"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=10, value=0)
smoker = st.selectbox("Smoker", options=["yes", "no"])
region = st.selectbox("Region", options=["northwest", "southeast", "southwest", "northeast"])

if st.button("Predict Charges"):
    input_data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [1 if sex == "male" else 0],
        "smoker_yes": [1 if smoker == "yes" else 0],
        "region_northwest": [1 if region == "northwest" else 0],
        "region_southeast": [1 if region == "southeast" else 0],
        "region_southwest": [1 if region == "southwest" else 0]
    })
    
    prediction = rf_model.predict(input_data)
    st.write("Predicted Charges (Annual): ${:,.2f}".format(prediction[0]))

# Conclusion
st.header("Conclusion")
st.write("""
The Random Forest Regressor model performs better in predicting the insurance charges compared to the Linear Regression model. This is evident from the lower RMSE and higher R² values for the Random Forest model.
From the exploratory data analysis and visualizations, it is clear that factors like BMI and smoking status have a significant impact on the insurance charges.
""")

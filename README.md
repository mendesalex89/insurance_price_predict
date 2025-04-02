# Insurance Charges Prediction

## Table of Contents
- [Introduction](#introduction)
- [Machine Learning Paradigms](#machine-learning-paradigms)
- [Chosen Algorithm](#chosen-algorithm)
- [How the Algorithm Works](#how-the-algorithm-works)
- [Real-World Application](#real-world-application)
- [Application Overview](#application-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Introduction
This project demonstrates the use of machine learning to predict insurance charges based on factors such as:
- Age
- Sex
- BMI (Body Mass Index)
- Number of children
- Smoking status
- Region

Dataset: `insurance.csv`

## 🧠 Machine Learning Paradigms
Machine learning paradigms include:
- **Supervised Learning**: Trained on labeled data (e.g., Linear Regression, Decision Trees)
- **Unsupervised Learning**: Finds patterns in unlabeled data (e.g., K-Means Clustering)
- **Semi-Supervised Learning**: Combines labeled and unlabeled data

## 📊 Chosen Algorithm: Linear Regression
Linear Regression models the relationship between a dependent variable and independent variables using a linear approach.

## ⚙️ How the Algorithm Works
The algorithm fits a linear equation to observed data:

Y = β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ

Where:
- Y = Insurance charges (dependent variable)
- X₁...Xₙ = Independent variables
- β₀...βₙ = Learned coefficients

## 🌍 Real-World Application
Predicting insurance charges helps companies set fair pricing based on individual risk factors and characteristics.

## 📌 Application Overview
A Streamlit web application that allows users to:
✅ Explore dataset visualizations  
✅ Train ML models (Linear Regression & Random Forest)  
✅ Evaluate model performance  
✅ Predict insurance charges via user input  

## ⚡ Installation
To run locally:
```bash
# Clone repository
git clone https://github.com/alexmendes/insurance_price_predict.git
cd insurance_price_predict

# Create and activate virtual environment
python3 -m venv myenv
source myenv/bin/activate  # Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py


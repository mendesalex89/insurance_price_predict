# insurance_price_predict


Insurance Charges Prediction

📌 Table of Contents

Introduction

Machine Learning Paradigms

Chosen Algorithm

How the Algorithm Works

Real-World Application

Application Overview

Installation

Usage

Contributing

License

🔍 Introduction

This project demonstrates the use of machine learning to predict insurance charges based on factors such as:

Age

Sex

BMI (Body Mass Index)

Number of children

Smoking status

Region

The dataset used is insurance.csv.

🧠 Machine Learning Paradigms

Machine learning paradigms include:

Supervised Learning: The model is trained on labeled data. Examples: Linear Regression, Decision Trees, Classification Algorithms.

Unsupervised Learning: The model finds patterns in unlabeled data. Example: K-Means Clustering.

Semi-Supervised Learning: A mix of labeled and unlabeled data to enhance learning accuracy.

📊 Chosen Algorithm: Linear Regression

Linear Regression is a statistical method that models the relationship between a dependent variable and one or more independent variables. It assumes a linear relationship between inputs and outputs.

⚙️ How the Algorithm Works

In Linear Regression, the algorithm fits a linear equation to the observed data:



Where:

Y = Dependent variable (Insurance charges)

X₁, X₂, ..., Xₙ = Independent variables

β₀, β₁, ..., βₙ = Coefficients the algorithm learns

🌍 Real-World Application

A practical use of Linear Regression is predicting insurance charges based on personal characteristics. This helps insurance companies set fair and accurate pricing for clients.

📌 Application Overview

This application, built with Streamlit, allows users to:
✅ Explore the dataset through visualizations
✅ Train machine learning models (Linear Regression & Random Forest)
✅ Evaluate model performance
✅ Predict insurance charges based on user input

⚡ Installation

To run the Streamlit application locally, follow these steps:

Clone the repository:

git clone https://github.com/alexmendes/insurance_price_predict.git
cd insurance_price_predict

Create and activate a virtual environment:

python3 -m venv myenv
source myenv/bin/activate  # On Windows use: myenv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

🚀 Usage

After running the Streamlit app, you will see an interactive interface where you can:

Explore data

Train models

Evaluate performance

Make predictions











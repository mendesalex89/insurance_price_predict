# insurance_price_predict

## Análise do Deploy



## Deploy Analysis

## Exploratory Data Analysis
Description: This section shows the first few rows of the health insurance dataset.
Columns:

age: Age of the primary beneficiary.
sex: Gender of the primary beneficiary (male or female).
bmi: Body Mass Index (BMI), a rough estimate of body fat.
children: Number of dependent children covered by the health insurance.
smoker: Whether the beneficiary is a smoker (yes) or a non-smoker (no).
region: The region where the beneficiary resides in the US (northwest, southeast, southwest, etc.).
charges: Individual medical costs (the prediction target).


## Preprocessing

Description: This section shows the transformation of categorical data into dummy variables (One-Hot Encoding).
Transformation:
The categorical columns sex, smoker, and region are converted into several binary columns (0 or 1).
Examples of new columns:

sex_male: 1 if the beneficiary is male, 0 if female.
smoker_yes: 1 if the beneficiary is a smoker, 0 if a non-smoker.
region_northwest, region_southeast, etc.: 1 if the beneficiary resides in the respective region, 0 if not.
Model Training
Description: This section shows the process of training linear regression and random forest regression models.
Steps:

The dataset is divided into independent variables (X) and the dependent variable (y), where y is the charges column.
The data is split into training (80%) and testing (20%) sets.
Two models are trained:
Linear Regression (LinearRegression): A simple statistical model that assumes a linear relationship between the independent and dependent variables.
Random Forest Regressor (RandomForestRegressor): A machine learning model that uses multiple decision trees to improve accuracy and control overfitting.

## Model Evaluation
Description: This section presents the evaluation of the trained models using the test set.
Metrics:

RMSE (Root Mean Squared Error): Measures the difference between predicted and actual values. Lower values indicate a better model.
R² (R-Squared): Measures the proportion of variance in the dependent variable that is predictable from the independent variables. Values closer to 1 indicate a better model.
Results:
Linear Regression:
RMSE: 5796.284659276273
R²: 0.7835929767120724
Random Forest:
RMSE: 4576.299916517115
R²: 0.8651034292144947
Interpretation: The Random Forest Regression model performs better (lower RMSE and higher R²) compared to the Linear Regression model.
Summary
Exploratory Data Analysis: Provides an overview of the original data, helping to understand the structure and content of the dataset.
Preprocessing: Details the transformation of categorical data into dummy variables, facilitating the use of these data in machine learning models.
Model Training: Explains the process of training linear regression and random forest regression models.
Model Evaluation: Assesses the performance of the models using metrics like RMSE and R², highlighting the effectiveness of each model in predicting medical costs.














## Exploratory Data Analysis (Análise Exploratória de Dados)

Descrição: Esta seção mostra as primeiras linhas do dataset de seguro de saúde.
Colunas:
age: Idade do beneficiário principal.
sex: Sexo do beneficiário principal (masculino ou feminino).
bmi: Índice de Massa Corporal (IMC), uma medida aproximada de gordura corporal.
children: Número de filhos dependentes cobertos pelo seguro de saúde.
smoker: Se o beneficiário é fumante (yes) ou não fumante (no).
region: A região onde o beneficiário reside nos EUA (noroeste, sudeste, sudoeste, etc.).
charges: Custos médicos individuais (o objetivo da previsão).

## Preprocessing (Pré-processamento)
Descrição: Esta seção mostra a transformação de dados categóricos em variáveis dummy (One-Hot Encoding).
Transformação:
As colunas categóricas sex, smoker e region são convertidas em várias colunas binárias (0 ou 1).
Exemplos de novas colunas:
sex_male: 1 se o beneficiário é do sexo masculino, 0 se feminino.
smoker_yes: 1 se o beneficiário é fumante, 0 se não fumante.
region_northwest, region_southeast, etc.: 1 se o beneficiário reside na respectiva região, 0 se não.

## Model Training (Treinamento do Modelo)

Descrição: Esta seção mostra o processo de treinamento dos modelos de regressão linear e regressão com floresta aleatória.
Passos:
O dataset é dividido em variáveis independentes (X) e a variável dependente (y), onde y é a coluna charges.
Os dados são divididos em conjuntos de treinamento (80%) e teste (20%).
Dois modelos são treinados:
Regressão Linear (LinearRegression): Um modelo estatístico simples que assume uma relação linear entre as variáveis independentes e dependente.
Regressor de Floresta Aleatória (RandomForestRegressor): Um modelo de aprendizado de máquina que utiliza múltiplas árvores de decisão para melhorar a precisão e controlar o overfitting.

## Model Evaluation (Avaliação do Modelo)

Descrição: Esta seção apresenta a avaliação dos modelos treinados usando o conjunto de teste.
Métricas:
RMSE (Root Mean Squared Error - Erro Quadrático Médio da Raiz): Mede a diferença entre os valores previstos e os valores reais. Valores menores indicam um modelo melhor.
R² (R-Squared - Coeficiente de Determinação): Mede a proporção da variância na variável dependente que é previsível a partir das variáveis independentes. Valores mais próximos de 1 indicam um modelo melhor.
Resultados:
Linear Regression:
RMSE: 5796.284659276273
R²: 0.7835929767120724
Random Forest:
RMSE: 4576.299916517115
R²: 0.8651034292144947
Interpretação: O modelo de Regressão com Floresta Aleatória apresenta melhor desempenho (menor RMSE e maior R²) comparado ao modelo de Regressão Linear.
Resumo
Exploratory Data Analysis: Mostra uma visão geral dos dados originais, ajudando a entender a estrutura e o conteúdo do dataset.
Preprocessing: Detalha a transformação de dados categóricos em variáveis dummy, facilitando o uso desses dados em modelos de aprendizado de máquina.
Model Training: Explica o processo de treinamento dos modelos de regressão linear e regressão com floresta aleatória.
Model Evaluation: Avalia a performance dos modelos utilizando métricas como RMSE e R², destacando a eficácia de cada modelo na previsão de custos médicos.
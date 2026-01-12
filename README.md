Vehicle Price Prediction

This project aims to predict used vehicle prices using Machine Learning techniques, comparing different regression models and evaluating their performance through statistical metrics.

The analysis includes:

Exploratory Data Analysis (EDA)

Feature engineering

Data transformations and categorical variable encoding

Training and evaluation of multiple regression models

📊 Dataset

The model uses a used vehicle price dataset that includes information such as:

Make and model

Vehicle year

Mileage

Fuel type, transmission, and body type

Accident history and condition

Vehicle price (target variable)

⚠️ The dataset is not included in this repository due to GitHub file size limitations (>100 MB).

🔗 Dataset source

The dataset can be downloaded from:

Kaggle:
https://www.kaggle.com/datasets/metawave/vehicle-price-prediction

Once downloaded, it must be placed in the following path:
data/vehicle_price_prediction.csv

🧠 Implemented models

The following models were trained and compared:

Linear Regression

Random Forest Regressor

XGBoost Regressor

LightGBM Regressor

Evaluation metrics

R² (Coefficient of Determination)

RMSE (Root Mean Squared Error)

Additionally, a residual analysis was performed to evaluate prediction error behavior.

🛠️ Libraries used

pandas

numpy

matplotlib

seaborn

scikit-learn

tensorflow

xgboost

lightgbm

⚙️ Installation and environment setup

It is recommended to create a virtual environment containing the libraries listed above before running the project.

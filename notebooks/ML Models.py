# %%
# importar librerias
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


# %%
# leer dataset
df = pd.read_csv("../data/vehicle_price_prediction.csv")


# %%
# EDA

# Primeras 10 filas

print(df.head())

# estadisticas basicas

print(df.describe().round(2))

# forma de la table

print(df.shape)

# valores null por columna

print(df.isnull().sum())

# Porcentaje de valores null en columnas

print((df.isnull().sum() / (len(df))) * 100)

# valores unicos

columns = df[
    [
        "make",
        "model",
        "transmission",
        "trim",
        "fuel_type",
        "drivetrain",
        "body_type",
        "exterior_color",
        "accident_history",
        "seller_type",
        "condition",
    ]
]

for col in columns:
    print(f"Valores únicos en la columna '{col}':")
    print(df[col].unique())

print(df.info())


# %%

# transformaciones

# rellenar nulls

df["accident_history"] = df["accident_history"].fillna("Unknown")
print(df.isnull().sum())

# %%

# graficos

# conteo de marcas

make_counts = df["make"].value_counts()


plt.figure(figsize=(20, 10))
plt.bar(make_counts.index, make_counts.values, color="skyblue")
plt.title("Gráfico de Barras")
plt.xlabel("Categorías")
plt.ylabel("Valores")
plt.show()

# autos agrupados por año


plt.figure(figsize=(10, 10))
plt.hist(df["year"], bins=25, color="skyblue", edgecolor="black")
plt.title("Gráfico de Barras")
plt.xlabel("Categorías")
plt.ylabel("Valores")
plt.show()

# distribucion marcas vs precios

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 8))
sns.boxplot(x="make", y="price", data=df)
plt.title("Distribución de Precios por Marca")
plt.xlabel("Marca")
plt.ylabel("Precio")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# distribucion precio vs kilometraje

plt.figure(figsize=(20, 8))
sns.lmplot(x="mileage_per_year", y="price", data=df)
plt.title("Distribución de Precios por Marca")
plt.xlabel("millaje")
plt.ylabel("Precio")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# matriz de confusion, correlacion de variables numericas

numeric_df = df.select_dtypes(include=["int64", "float"])

corr_data = numeric_df.corr()

# heatmap

plt.figure(figsize=(12, 9))

sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Heatmap de Correlación entre Variables Numéricas")
plt.show()

# %%


# feature engineering

df["vehicle_age_FE"] = 2025 - df["year"]

df["MI_per_year"] = df["mileage"] / (df["vehicle_age_FE"])

print(df[["MI_per_year", "mileage_per_year"]])


# %%

# One hot encoding de variables binarias

df_FE = pd.get_dummies(
    df,
    columns=[
        "fuel_type",
        "transmission",
        "drivetrain",
        "trim",
        "body_type",
        "exterior_color",
        "exterior_color",
        "accident_history",
        "condition",
        "seller_type",
    ],
    dtype=int,
)


# %%

# target encoding para variables multicategorias

mean_price_per_make = df_FE.groupby("make")["price"].mean()
df_FE["make_encoded"] = df_FE["make"].map(mean_price_per_make)

mean_price_per_model = df_FE.groupby("model")["price"].mean()
df_FE["model_encoded"] = df_FE["model"].map(mean_price_per_model)
# %%

# Nueva matriz de correlacion

numeric_df = df_FE.select_dtypes(include=["Int64", "float"])

corr_data = numeric_df.corr()

plt.figure(figsize=(30, 12))

sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
plt.title("Heatmap de Correlación entre Variables Numéricas")
plt.show()
# %%

print(df_FE.dtypes)
# %%

# asignacion de features y valor a predecir en X y Y

features = ["vehicle_age", "owner_count", "engine_hp", "make_encoded", "model_encoded"]

x = df_FE[features]
y = df["price"]


# %%

# division de entrenamiento y validacion
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)


# %%

# escalado para X

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

# %%

# entrenamiento de regresion lineal

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(x_train_scaled, y_train)

# %%

# prediccion

y_pred = model.predict(x_test_scaled)

# %%

# metricas de evaluacion

from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

r2_reg_lin = r2_score(y_test, y_pred)
rmse_reg_lin = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² en test:", round(r2_reg_lin, 4))
print("RMSE en test:", round(rmse_reg_lin, 2))

# %%

# calcular residuos

residuals = y_test - y_pred

plt.figure(figsize=(12, 7))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color="red", linestyle="--")
plt.title("Gráfico de Residuos vs. Valores Predichos", fontsize=16)
plt.xlabel("Valores Predichos (Precio del Vehículo)", fontsize=12)
plt.ylabel("Residuos (Error = Real - Predicho)", fontsize=12)
plt.grid(True)
plt.show()

# %%
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

# %%

# random forest
rf = RandomForestRegressor(
    n_estimators=300, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = root_mean_squared_error(y_test, y_pred_rf)

print("Random Forest")
print("R2:", rf_r2)
print("RMSE:", rf_rmse)

# %%

# XG Boost


xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
)

xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

xgb_r2 = r2_score(y_test, y_pred_xgb)
xgb_rmse = root_mean_squared_error(y_test, y_pred_xgb)

print("XGBoost")
print("R2:", xgb_r2)
print("RMSE:", xgb_rmse)

# %%

# Light GBM

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

lgb_model.fit(X_train, y_train)

y_pred_lgb = lgb_model.predict(X_test)

lgb_r2 = r2_score(y_test, y_pred_lgb)
lgb_rmse = root_mean_squared_error(y_test, y_pred_lgb)

print("LightGBM")
print("R2:", lgb_r2)
print("RMSE:", lgb_rmse)


# %%

# comparacion de modelos

results = pd.DataFrame(
    {
        "Modelo": ["Regresión Lineal", "Random Forest", "XGBoost", "LightGBM"],
        "R2": [r2_reg_lin, rf_r2, xgb_r2, lgb_r2],
        "RMSE": [rmse_reg_lin, rf_rmse, xgb_rmse, lgb_rmse],
    }
)


results


# %%

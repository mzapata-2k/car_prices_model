# Vehicle Price Prediction

Este proyecto tiene como objetivo predecir el precio de vehículos usados utilizando técnicas de Machine Learning, comparando distintos modelos de regresión y evaluando su desempeño mediante métricas estadísticas.

El análisis incluye:
- Exploración y análisis descriptivo de los datos (EDA)
- Ingeniería de características (feature engineering)
- Transformaciones y codificación de variables categóricas
- Entrenamiento y evaluación de múltiples modelos de regresión

---

## 📊 Dataset

El modelo utiliza un dataset de precios de vehículos usados que incluye información como:
- Marca y modelo
- Año del vehículo
- Kilometraje
- Tipo de combustible, transmisión y carrocería
- Historial de accidentes y condición
- Precio del vehículo (variable objetivo)

⚠️ **El dataset no está incluido en este repositorio** debido a restricciones de tamaño de GitHub (>100 MB).

### 🔗 Fuente del dataset
El dataset puede descargarse desde:
- **Kaggle**:  
  https://www.kaggle.com/datasets/metawave/vehicle-price-prediction
Una vez descargado, debe colocarse en la siguiente ruta: data/vehicle_price_prediction.csv


---

## 🧠 Modelos implementados

Se entrenaron y compararon los siguientes modelos:

- Regresión Lineal
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

### Métricas de evaluación
- **R² (Coeficiente de determinación)**
- **RMSE (Root Mean Squared Error)**

Además, se realizó un análisis de residuos para evaluar el comportamiento del error de predicción.

---

## 🛠️ Librerías utilizadas

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- xgboost
- lightgbm

---

## ⚙️ Instalación y entorno

Se recomienda crear un entorno virtual que contenga las librerias refrenciadas anteriormente antes de ejecutar el proyecto.


  


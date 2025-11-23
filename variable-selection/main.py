import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# Cargar el archivo CSV
housing_df = pd.read_csv("./california_housing_train.csv")

# Ver las primeras filas
print(housing_df.head())

# Calcular la matriz de correlación
correlation_matrix = housing_df.corr()

# Mostrar el mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of California Housing Dataset')
plt.show()

# Separar las variables independientes de la variable objetivo
X = housing_df.drop('median_house_value', axis=1)
y = housing_df['median_house_value']

# Mostrar ejemplos de X e y
print(X.head())
print(y.head())

# Calcular la información mutua
mutual_info_scores = mutual_info_regression(X, y)

# Crear una serie con los puntajes
mutual_info_series = pd.Series(mutual_info_scores, index=X.columns)

# Ordenar los puntajes de mayor a menor
sorted_mutual_info = mutual_info_series.sort_values(ascending=False)

# Mostrar resultados
print(sorted_mutual_info)

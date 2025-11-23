import seaborn as sns
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Cargar dataset Titanic
titanic = sns.load_dataset('titanic')

# Mostrar primeras filas
print(titanic.head())

# Ver cuántos valores faltan tiene cada columna
print(titanic.isnull().sum())

# Seleccionar columnas para imputación
data = titanic[['age', 'fare', 'pclass', 'sex']]

# Convertir variables categóricas a dummies
data = pd.get_dummies(data, drop_first=True)

# Crear el imputador MICE (IterativeImputer)
imputer = IterativeImputer(max_iter=10, random_state=0)

# Aplicar el imputador
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Mostrar datos originales e imputados
print("Datos originales")
print(data.head(30))

print("\nDatos imputados")
print(data_imputed.head(30))


# -----------------------------------------------------------------------
# Ejercicio 1: MICE en dataset Life Expectancy
# -----------------------------------------------------------------------

# Cargar dataset
life = pd.read_csv('LifeExpectancyData.csv')

# Revisar valores faltantes
print(life.isnull().sum())

# Seleccionar columnas numéricas
life_numerical = life.select_dtypes(include=['float64', 'int64'])

# Crear imputador
imputer = IterativeImputer(max_iter=10, random_state=0)

# Imputar
life_imputed_array = imputer.fit_transform(life_numerical)

# Convertir a DataFrame
life_imputed = pd.DataFrame(life_imputed_array, columns=life_numerical.columns)

# Mostrar resultados
print("Original (numéricas):")
print(life_numerical.head())

print("\nImputado:")
print(life_imputed.head())

print("\nValores faltantes después de imputar:")
print(life_imputed.isnull().sum())


# -----------------------------------------------------------------------
# Ejercicio 2: MICE en dataset Planets
# -----------------------------------------------------------------------

# Cargar dataset planets
planets = sns.load_dataset('planets')

# Seleccionar columnas numéricas
planets_numerical = planets.select_dtypes(include=['float64', 'int64'])

# Crear imputador
imputer = IterativeImputer(max_iter=10, random_state=0)

# Imputar
planets_imputed_array = imputer.fit_transform(planets_numerical)

# Convertir a DataFrame
planets_imputed = pd.DataFrame(planets_imputed_array, columns=planets_numerical.columns)

# Mostrar resultados
print("Original (numéricas):")
print(planets_numerical.head())

print("\nImputado:")
print(planets_imputed.head())

print("\nValores faltantes después de imputar:")
print(planets_imputed.isnull().sum())

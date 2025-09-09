import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import os

#------ Configuración de URL y nombre de archivo ------#
url = "https://ecobici.cdmx.gob.mx/wp-content/uploads/2025/04/2025-03.csv"
csv_file_name = "2025-03.csv"

#------ Verificar si el archivo ya existe ------#
if os.path.exists(csv_file_name):
    print(f"[INFO] Archivo {csv_file_name} ya existe. Usando archivo local.")
else:
    print(f"[INFO] Descargando datos desde: {url}")
    try:
        response = requests.get(url, timeout=1200)
        response.raise_for_status()
        print("[INFO] Descarga completada con éxito")

        #------ Guardar archivo CSV ------#
        with open(csv_file_name, "wb") as f:
            f.write(response.content)
        print(f"[INFO] Archivo CSV guardado como {csv_file_name}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] No se pudo descargar el archivo: {e}")
        exit()

#------ Leer CSV con pandas ------#
print(f"[INFO] Leyendo datos desde {csv_file_name}")
df_raw = pd.read_csv(csv_file_name)
print(f"[INFO] Extracción completada. Registros cargados: {df_raw.shape[0]}")
print(f"[INFO] Columnas: {df_raw.columns.tolist()}")

#------ Crear columna Duracion_minutos combinando fecha y hora ------#
print("[INFO] Calculando duración en minutos...")
df_raw['FechaHora_Retiro'] = pd.to_datetime(df_raw['Fecha_Retiro'] + ' ' + df_raw['Hora_Retiro'], dayfirst=True)
df_raw['FechaHora_Arribo'] = pd.to_datetime(df_raw['Fecha_Arribo'] + ' ' + df_raw['Hora_Arribo'], dayfirst=True)
df_raw['Duracion_minutos'] = (df_raw['FechaHora_Arribo'] - df_raw['FechaHora_Retiro']).dt.total_seconds() / 60
print("[INFO] Duración calculada y columna 'Duracion_minutos' agregada.")

#------ Crear columna Tipo_viaje ------#
print("[INFO] Clasificando tipo de viaje...")
conditions = [
    df_raw['Duracion_minutos'] < 10,
    (df_raw['Duracion_minutos'] >= 10) & (df_raw['Duracion_minutos'] <= 30),
    df_raw['Duracion_minutos'] > 30
]
choices = ['Corto', 'Medio', 'Largo']
df_raw['Tipo_viaje'] = np.select(conditions, choices, default='Desconocido')
print("[INFO] Columna 'Tipo_viaje' agregada.")

#------ Eliminar columnas de fecha/hora originales ------#
df_raw.drop(['Fecha_Retiro', 'Hora_Retiro', 'Fecha_Arribo', 'Hora_Arribo'], axis=1, inplace=True)
print("[INFO] Columnas de fecha y hora originales eliminadas.")

#------ Función para mostrar conteos y gráficas ------#
def mostrar_estadisticas(df):
    # Conteo por categoría de duración
    print("[INFO] Conteo de viajes por tipo de duración:")
    conteo_tipo = df['Tipo_viaje'].value_counts()
    print(conteo_tipo)
    
    plt.figure(figsize=(6,4))
    conteo_tipo.plot(kind='bar', color='skyblue')
    plt.title('Conteo de viajes por tipo de duración')
    plt.xlabel('Tipo de viaje')
    plt.ylabel('Número de viajes')
    plt.xticks(rotation=0)
    plt.show()
    
    # Conteo por género
    print("\n[INFO] Conteo de viajes por género:")
    conteo_genero = df['Genero_Usuario'].value_counts()
    print(conteo_genero)
    
    plt.figure(figsize=(6,4))
    conteo_genero.plot(kind='bar', color='lightgreen')
    plt.title('Conteo de viajes por género')
    plt.xlabel('Género de usuario')
    plt.ylabel('Número de viajes')
    plt.xticks(rotation=0)
    plt.show()

#------ Mostrar estadísticas y gráficas ------#
mostrar_estadisticas(df_raw)

#------ Preparar datos para regresión lineal ------#
print("[INFO] Preparando datos para regresión lineal...")
X = df_raw[['Edad_Usuario', 'Genero_Usuario', 'Ciclo_Estacion_Retiro', 'Ciclo_EstacionArribo']].copy()
y = df_raw['Duracion_minutos']

#------ Imputar valores nulos ------#
X['Edad_Usuario'] = X['Edad_Usuario'].fillna(X['Edad_Usuario'].median())
X['Genero_Usuario'] = X['Genero_Usuario'].fillna('O')
X['Ciclo_Estacion_Retiro'] = X['Ciclo_Estacion_Retiro'].fillna('Unknown')
X['Ciclo_EstacionArribo'] = X['Ciclo_EstacionArribo'].fillna('Unknown')
print("[INFO] Valores nulos imputados.")

#------ Codificar variables categóricas ------#
le_genero = LabelEncoder()
X['Genero_Usuario'] = le_genero.fit_transform(X['Genero_Usuario'])
le_retiro = LabelEncoder()
X['Ciclo_Estacion_Retiro'] = le_retiro.fit_transform(X['Ciclo_Estacion_Retiro'].astype(str))
le_arribo = LabelEncoder()
X['Ciclo_EstacionArribo'] = le_arribo.fit_transform(X['Ciclo_EstacionArribo'].astype(str))
print("[INFO] Variables categóricas codificadas.")

#------ Entrenar modelo de regresión ------#
print("[INFO] Entrenando modelo de regresión lineal...")
modelo = LinearRegression()
modelo.fit(X, y)
print("[INFO] Modelo entrenado.")

#------ Predecir duración y agregar columna ------#
df_raw['Duracion_predicha'] = modelo.predict(X)
print("[INFO] Columna 'Duracion_predicha' agregada.")

#------ Mostrar algunas predicciones ------#
print("[INFO] Primeras 10 predicciones de duración (minutos):")
print(np.round(df_raw['Duracion_predicha'].head(10), 2))

#------ Comparación duración real vs predicha ------#
comparacion = pd.DataFrame({'Duracion_real': df_raw['Duracion_minutos'],
                            'Duracion_predicha': df_raw['Duracion_predicha']})
print("[INFO] Comparación real vs predicha (primeros 10 viajes):")
print(comparacion.head(10))

#------ Graficar comparación ------#
print("[INFO] Graficando comparación de duración real vs predicha...")
plt.figure(figsize=(10,5))
plt.plot(comparacion['Duracion_real'].values[:50], label='Real', marker='o')
plt.plot(comparacion['Duracion_predicha'].values[:50], label='Predicha', marker='x')
plt.title('Comparación de duración real vs predicha (primeros 50 viajes)')
plt.xlabel('Viaje')
plt.ylabel('Duración (minutos)')
plt.legend()
plt.show()
print("[INFO] Proceso completado.")

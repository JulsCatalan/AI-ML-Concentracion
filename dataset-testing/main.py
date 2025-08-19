import pandas as pd
import requests

url = "https://ecobici.cdmx.gob.mx/wp-content/uploads/2025/04/2025-03.csv"
csv_file_name = "2025-03.csv"

print(f"Descargando datos desde: {url}")
try:
    response = requests.get(url, timeout=1200)
    response.raise_for_status()
    print("Descarga completada con éxito")

    # Guardar el archivo CSV
    with open(csv_file_name, "wb") as f:
        f.write(response.content)
    print(f"Archivo CSV guardado como {csv_file_name}")

    # Leer el CSV con pandas
    print(f"Leyendo datos desde {csv_file_name}")
    df_raw = pd.read_csv(csv_file_name)
    print("Extracción completada con éxito")
    print(f"Se cargaron {df_raw.shape[0]} registros")
    print("Tamaño del DataFrame:", df_raw.shape)
    print("\nPrevisualización del DataFrame:")
    print(df_raw.head(50))  # reemplazamos display() por print()

except requests.exceptions.RequestException as e:
    print(f"Error al descargar los datos: {e}")
    df_raw = pd.DataFrame()

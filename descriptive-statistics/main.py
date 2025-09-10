import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

plt.style.use('default')

def cargar_datasets():
    print("-" * 60)
    print("CARGANDO DATASETS")
    
    # Dataset Wine Quality
    wine_file = "winequality-red.csv"
    if os.path.exists(wine_file):
        print(f"Archivo {wine_file} encontrado localmente, cargando...")
        wine_data = pd.read_csv(wine_file, delimiter=";")
    else:
        print("Descargando dataset Wine Quality desde UCI...")
        url_wine = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        wine_data = pd.read_csv(url_wine, delimiter=";")
        wine_data.to_csv(wine_file, sep=";", index=False)
        print(f"Dataset guardado como {wine_file}")
    print(f"Dataset Wine cargado: {wine_data.shape[0]} filas, {wine_data.shape[1]} columnas")
    
    # Dataset Ecobici
    ecobici_file = "2025-03.csv"
    if os.path.exists(ecobici_file):
        print(f"Archivo {ecobici_file} encontrado localmente, cargando...")
        ecobici_data = pd.read_csv(ecobici_file)
        print(f"Dataset Ecobici cargado: {ecobici_data.shape[0]} filas, {ecobici_data.shape[1]} columnas")
    else:
        print("Descargando dataset Ecobici desde CDMX...")
        url_ecobici = "https://ecobici.cdmx.gob.mx/wp-content/uploads/2025/04/2025-03.csv"
        try:
            ecobici_data = pd.read_csv(url_ecobici)
            ecobici_data.to_csv(ecobici_file, index=False)
            print(f"Dataset guardado como {ecobici_file}")
            print(f"Dataset Ecobici cargado: {ecobici_data.shape[0]} filas, {ecobici_data.shape[1]} columnas")
        except Exception as e:
            print(f"Error cargando Ecobici: {e}")
            # Crear datos simulados si no se puede descargar
            np.random.seed(42)
            ecobici_data = pd.DataFrame({
                'duracion_minutos': np.random.exponential(20, 1000),
                'distancia_km': np.random.gamma(2, 2, 1000),
                'edad_usuario': np.random.normal(35, 12, 1000),
                'estacion_origen': np.random.randint(1, 100, 1000),
                'estacion_destino': np.random.randint(1, 100, 1000)
            })
            print("Usando datos simulados de Ecobici para demostración")
    
    return wine_data, ecobici_data

def medidas_tendencia_central(data, nombre_dataset, columna_analizar):
    print(f"\n{'-'*50}")
    print(f"MEDIDAS DE TENDENCIA CENTRAL - {nombre_dataset.upper()}")
    print(f"Columna analizada: {columna_analizar}")
    
    # Calcular medidas
    media = data[columna_analizar].mean()
    mediana = data[columna_analizar].median()
    moda = data[columna_analizar].mode()
    
    print(f"Media: {media:.4f}")
    print(f"Mediana: {mediana:.4f}")
    print(f"Moda: {moda.iloc[0]:.4f}" if len(moda) > 0 else "Moda: No hay moda única")
    
    # Crear visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Medidas de Tendencia Central - {nombre_dataset}\nColumna: {columna_analizar}', 
                 fontsize=16, fontweight='bold')
    
    # Histograma con medidas marcadas
    axes[0,0].hist(data[columna_analizar], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.2f}')
    axes[0,0].axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
    if len(moda) > 0:
        axes[0,0].axvline(moda.iloc[0], color='orange', linestyle='--', linewidth=2, 
                         label=f'Moda: {moda.iloc[0]:.2f}')
    axes[0,0].set_title('Distribución con Medidas Centrales')
    axes[0,0].set_xlabel(columna_analizar)
    axes[0,0].set_ylabel('Frecuencia')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Boxplot
    axes[0,1].boxplot(data[columna_analizar], vert=True)
    axes[0,1].set_title('Boxplot')
    axes[0,1].set_ylabel(columna_analizar)
    axes[0,1].grid(True, alpha=0.3)
    
    # Gráfico de barras de medidas
    medidas = ['Media', 'Mediana', 'Moda']
    valores = [media, mediana, moda.iloc[0] if len(moda) > 0 else 0]
    colores = ['red', 'green', 'orange']
    
    axes[1,0].bar(medidas, valores, color=colores, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Comparación de Medidas Centrales')
    axes[1,0].set_ylabel('Valor')
    for i, v in enumerate(valores):
        axes[1,0].text(i, v + max(valores)*0.01, f'{v:.2f}', ha='center', va='bottom')
    axes[1,0].grid(True, alpha=0.3)
    
    # Gráfico de densidad
    axes[1,1].hist(data[columna_analizar], bins=30, density=True, alpha=0.7, 
                   color='lightblue', edgecolor='black')
    axes[1,1].axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.2f}')
    axes[1,1].axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
    axes[1,1].set_title('Distribución de Densidad')
    axes[1,1].set_xlabel(columna_analizar)
    axes[1,1].set_ylabel('Densidad')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {'media': media, 'mediana': mediana, 'moda': moda.iloc[0] if len(moda) > 0 else None}

def medidas_dispersion(data, nombre_dataset, columna_analizar):
    print(f"\n{'-'*50}")
    print(f"MEDIDAS DE DISPERSIÓN - {nombre_dataset.upper()}")
    print(f"Columna analizada: {columna_analizar}")
    
    # Calcular medidas de dispersión
    rango = data[columna_analizar].max() - data[columna_analizar].min()
    varianza = data[columna_analizar].var()
    desv_std = data[columna_analizar].std()
    
    # Calcular quartiles
    Q1 = data[columna_analizar].quantile(0.25)
    Q2 = data[columna_analizar].quantile(0.50)
    Q3 = data[columna_analizar].quantile(0.75)
    IQR = Q3 - Q1
    
    print(f"Rango: {rango:.4f}")
    print(f"Varianza: {varianza:.4f}")
    print(f"Desviación Estándar: {desv_std:.4f}")
    print(f"Q1 (25th Percentile): {Q1:.4f}")
    print(f"Q2 (Mediana): {Q2:.4f}")
    print(f"Q3 (75th Percentile): {Q3:.4f}")
    print(f"IQR (Rango Intercuartílico): {IQR:.4f}")
    
    # Crear visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Medidas de Dispersión - {nombre_dataset}\nColumna: {columna_analizar}', 
                 fontsize=16, fontweight='bold')
    
    # Histograma con desviación estándar
    media = data[columna_analizar].mean()
    axes[0,0].hist(data[columna_analizar], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,0].axvline(media, color='red', linestyle='-', linewidth=2, label=f'Media: {media:.2f}')
    axes[0,0].axvline(media + desv_std, color='green', linestyle='--', linewidth=2, 
                     label=f'+1 SD: {media + desv_std:.2f}')
    axes[0,0].axvline(media - desv_std, color='green', linestyle='--', linewidth=2, 
                     label=f'-1 SD: {media - desv_std:.2f}')
    axes[0,0].set_title('Distribución con Desviación Estándar')
    axes[0,0].set_xlabel(columna_analizar)
    axes[0,0].set_ylabel('Frecuencia')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Boxplot con quartiles
    bp = axes[0,1].boxplot(data[columna_analizar], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[0,1].set_title('Boxplot con Quartiles')
    axes[0,1].set_ylabel(columna_analizar)
    axes[0,1].text(1.1, Q1, f'Q1: {Q1:.2f}', transform=axes[0,1].get_yaxis_transform())
    axes[0,1].text(1.1, Q2, f'Q2: {Q2:.2f}', transform=axes[0,1].get_yaxis_transform())
    axes[0,1].text(1.1, Q3, f'Q3: {Q3:.2f}', transform=axes[0,1].get_yaxis_transform())
    axes[0,1].grid(True, alpha=0.3)
    
    # Gráfico de barras de medidas de dispersión
    medidas = ['Rango', 'Varianza', 'Desv. Std', 'IQR']
    valores = [rango, varianza, desv_std, IQR]
    colores = ['purple', 'orange', 'red', 'blue']
    
    axes[1,0].bar(medidas, valores, color=colores, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Medidas de Dispersión')
    axes[1,0].set_ylabel('Valor')
    for i, v in enumerate(valores):
        axes[1,0].text(i, v + max(valores)*0.01, f'{v:.2f}', ha='center', va='bottom')
    axes[1,0].grid(True, alpha=0.3)
    
    # Gráfico de violín
    axes[1,1].violinplot([data[columna_analizar]], positions=[1], widths=0.5)
    axes[1,1].set_title('Gráfico de Violín')
    axes[1,1].set_ylabel(columna_analizar)
    axes[1,1].set_xticks([1])
    axes[1,1].set_xticklabels([columna_analizar])
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'rango': rango, 'varianza': varianza, 'desv_std': desv_std,
        'Q1': Q1, 'Q2': Q2, 'Q3': Q3, 'IQR': IQR
    }

def resumen_estadistico_completo(data, nombre_dataset):
    print(f"\n{'-'*60}")
    print(f"RESUMEN ESTADÍSTICO COMPLETO - {nombre_dataset.upper()}")
    
    # Información básica del dataset
    print(f"Forma del dataset: {data.shape}")
    print(f"Columnas numéricas: {data.select_dtypes(include=[np.number]).columns.tolist()}")
    
    # Estadísticas descriptivas generales
    print(f"\nEstadísticas Descriptivas:")
    print(data.describe())
    
    # Verificar valores faltantes
    print(f"\nValores Faltantes:")
    missing = data.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No hay valores faltantes")
    
    return data.describe()

def main():
    print("ANÁLISIS DE ESTADÍSTICAS DESCRIPTIVAS")
    print("Datasets: Wine Quality y Ecobici CDMX")
    
    # Cargar ambos datasets
    wine_data, ecobici_data = cargar_datasets()
    
    # Mostrar vista previa de los datos
    print(f"\nPrimeras 5 filas del dataset Wine:")
    print(wine_data.head())
    
    print(f"\nPrimeras 5 filas del dataset Ecobici:")
    print(ecobici_data.head())
    
    # Análisis completo del dataset Wine
    print(f"\n" + "="*80)
    print("ANÁLISIS COMPLETO DEL DATASET WINE")
    print("="*80)
    
    # Resumen estadístico del dataset Wine
    wine_stats = resumen_estadistico_completo(wine_data, "Wine Quality")
    
    # Análisis detallado de la columna 'alcohol'
    wine_central = medidas_tendencia_central(wine_data, "Wine Quality", "alcohol")
    wine_dispersion = medidas_dispersion(wine_data, "Wine Quality", "alcohol")
    
    # Análisis de la columna 'quality'
    wine_central_quality = medidas_tendencia_central(wine_data, "Wine Quality", "quality")
    wine_dispersion_quality = medidas_dispersion(wine_data, "Wine Quality", "quality")
    
    # Análisis completo del dataset Ecobici
    print(f"\n" + "="*80)
    print("ANÁLISIS COMPLETO DEL DATASET ECOBICI")
    print("="*80)
    
    # Resumen estadístico del dataset Ecobici
    ecobici_stats = resumen_estadistico_completo(ecobici_data, "Ecobici CDMX")
    
    # Identificar columnas numéricas en Ecobici
    columnas_numericas_ecobici = ecobici_data.select_dtypes(include=[np.number]).columns.tolist()
    
    if columnas_numericas_ecobici:
        # Análisis de la primera columna numérica
        primera_columna = columnas_numericas_ecobici[0]
        ecobici_central = medidas_tendencia_central(ecobici_data, "Ecobici CDMX", primera_columna)
        ecobici_dispersion = medidas_dispersion(ecobici_data, "Ecobici CDMX", primera_columna)
        
        # Análisis de segunda columna si existe
        if len(columnas_numericas_ecobici) > 1:
            segunda_columna = columnas_numericas_ecobici[1]
            ecobici_central_2 = medidas_tendencia_central(ecobici_data, "Ecobici CDMX", segunda_columna)
            ecobici_dispersion_2 = medidas_dispersion(ecobici_data, "Ecobici CDMX", segunda_columna)
    
    # Resumen final comparativo
    print(f"\n" + "="*80)
    print("RESUMEN FINAL DE ANÁLISIS")
    print("="*80)
    
    print("\nDataset Wine Quality - Columna 'alcohol':")
    print(f"  Media: {wine_central['media']:.4f}")
    print(f"  Mediana: {wine_central['mediana']:.4f}")
    print(f"  Desviación Estándar: {wine_dispersion['desv_std']:.4f}")
    
    if columnas_numericas_ecobici:
        print(f"\nDataset Ecobici CDMX - Columna '{primera_columna}':")
        print(f"  Media: {ecobici_central['media']:.4f}")
        print(f"  Mediana: {ecobici_central['mediana']:.4f}")
        print(f"  Desviación Estándar: {ecobici_dispersion['desv_std']:.4f}")
    
    print(f"\nAnálisis completado exitosamente!")
    print(f"Fecha de finalización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
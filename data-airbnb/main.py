import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuración para mostrar gráficas
plt.style.use('default')

# Cargar el dataset
print("=== CARGA DE DATOS ===")
df = pd.read_csv('listings.csv') 
print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"Primeras 5 filas:")
print(df.head())

# Limpieza de datos de precio
def clean_price(price):
    """Convierte precio de string a float"""
    if pd.isna(price):
        return np.nan
    # Remover símbolos de moneda y convertir a float
    price_str = str(price).replace('$', '').replace(',', '')
    try:
        return float(price_str)
    except:
        return np.nan

# Aplicar limpieza de precios
df['price_clean'] = df['price'].apply(clean_price)

print("\n=== INFORMACIÓN GENERAL DEL DATASET ===")
print(f"Valores nulos en precio: {df['price_clean'].isna().sum()}")
print(f"Precio mínimo: ${df['price_clean'].min():.2f}")
print(f"Precio máximo: ${df['price_clean'].max():.2f}")

# =================== PROMEDIO DE PRECIOS ===================
print("\n" + "="*50)
print("1. ANÁLISIS DE PRECIOS PROMEDIO")
print("="*50)

# Filtrar precios válidos (eliminar outliers extremos)
df_clean = df[df['price_clean'] > 0].copy()
precio_promedio = df_clean['price_clean'].mean()
precio_mediano = df_clean['price_clean'].median()

print(f"Precio promedio: ${precio_promedio:.2f}")
print(f"Precio mediano: ${precio_mediano:.2f}")
print(f"Desviación estándar: ${df_clean['price_clean'].std():.2f}")

# Gráfica de precio promedio
plt.figure(figsize=(10, 6))
plt.bar(['Promedio', 'Mediana'], [precio_promedio, precio_mediano], 
        color=['skyblue', 'lightcoral'])
plt.title('Precio Promedio vs Mediana')
plt.ylabel('Precio ($)')
for i, v in enumerate([precio_promedio, precio_mediano]):
    plt.text(i, v + precio_promedio*0.01, f'${v:.2f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# =================== TIPO DE HABITACIÓN MÁS COMÚN ===================
print("\n" + "="*50)
print("2. TIPO DE HABITACIÓN MÁS COMÚN")
print("="*50)

room_type_counts = df['room_type'].value_counts()
tipo_mas_comun = room_type_counts.index[0]
cantidad_mas_comun = room_type_counts.iloc[0]

print("Distribución de tipos de habitación:")
for tipo, cantidad in room_type_counts.items():
    porcentaje = (cantidad / len(df)) * 100
    print(f"  {tipo}: {cantidad} ({porcentaje:.1f}%)")

print(f"\nTipo más común: {tipo_mas_comun} con {cantidad_mas_comun} alojamientos")

# Gráfica de tipos de habitación
plt.figure(figsize=(12, 6))
plt.bar(room_type_counts.index, room_type_counts.values, color='lightgreen')
plt.title('Cantidad por Tipo de Habitación')
plt.xlabel('Tipo de Habitación')
plt.ylabel('Cantidad')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =================== ALCALDÍAS CON MÁS ALOJAMIENTOS ===================
print("\n" + "="*50)
print("3. ALCALDÍAS/VECINDARIOS CON MÁS ALOJAMIENTOS")
print("="*50)

# Usar la columna neighbourhood_cleansed que suele tener mejor calidad de datos
neighbourhood_counts = df['neighbourhood_cleansed'].value_counts().head(10)

print("Top 10 alcaldías/vecindarios con más alojamientos:")
for i, (alcaldia, cantidad) in enumerate(neighbourhood_counts.items(), 1):
    porcentaje = (cantidad / len(df)) * 100
    print(f"{i:2d}. {alcaldia}: {cantidad} alojamientos ({porcentaje:.1f}%)")

# Gráfica de alcaldías
plt.figure(figsize=(12, 6))
plt.barh(neighbourhood_counts.index[::-1], neighbourhood_counts.values[::-1], color='salmon')
plt.title('Top 10 Alcaldías/Vecindarios con Más Alojamientos')
plt.xlabel('Número de Alojamientos')
plt.tight_layout()
plt.show()

# =================== HOSTS CON MÁS ALOJAMIENTOS ===================
print("\n" + "="*50)
print("4. HOSTS CON MÁS ALOJAMIENTOS")
print("="*50)

# Análisis de hosts
host_counts = df['host_name'].value_counts().head(10)

print("Top 10 hosts con más alojamientos:")
for i, (host, cantidad) in enumerate(host_counts.items(), 1):
    print(f"{i:2d}. {host}: {cantidad} alojamientos")

# Estadísticas de hosts
print(f"\nEstadísticas de hosts:")
print(f"Total de hosts únicos: {df['host_name'].nunique()}")
print(f"Promedio de alojamientos por host: {df.groupby('host_name').size().mean():.2f}")
print(f"Host con más alojamientos: {host_counts.index[0]} ({host_counts.iloc[0]} alojamientos)")

# Gráfica de hosts
plt.figure(figsize=(12, 6))
plt.bar(range(len(host_counts)), host_counts.values, color='gold')
plt.title('Top 10 Hosts con Más Alojamientos')
plt.xlabel('Ranking de Host')
plt.ylabel('Número de Alojamientos')
plt.xticks(range(len(host_counts)), [f"Host {i+1}" for i in range(len(host_counts))])
plt.tight_layout()
plt.show()

# =================== HISTOGRAMA DE PRECIOS ===================
print("\n" + "="*50)
print("5. HISTOGRAMA DE PRECIOS")
print("="*50)

# Estadísticas básicas de precios
print("Estadísticas de precios:")
print(f"Media: ${df_clean['price_clean'].mean():.2f}")
print(f"Mediana: ${df_clean['price_clean'].median():.2f}")
print(f"Moda: ${df_clean['price_clean'].mode().iloc[0]:.2f}")
print(f"Rango: ${df_clean['price_clean'].min():.2f} - ${df_clean['price_clean'].max():.2f}")

# Histograma de precios
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist(df_clean['price_clean'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
plt.title('Histograma de Precios (Todos)')
plt.xlabel('Precio ($)')
plt.ylabel('Frecuencia')
plt.axvline(precio_promedio, color='red', linestyle='--', label=f'Promedio: ${precio_promedio:.2f}')
plt.axvline(precio_mediano, color='green', linestyle='--', label=f'Mediana: ${precio_mediano:.2f}')
plt.legend()

plt.subplot(1, 2, 2)
# Histograma con escala logarítmica
plt.hist(df_clean['price_clean'], bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
plt.title('Histograma de Precios (Escala Log)')
plt.xlabel('Precio ($)')
plt.ylabel('Frecuencia')
plt.yscale('log')
plt.tight_layout()
plt.show()

# =================== HISTOGRAMA SIN OUTLIERS ===================
print("\n" + "="*50)
print("6. HISTOGRAMA DE PRECIOS SIN OUTLIERS")
print("="*50)

# Calcular outliers usando IQR
Q1 = df_clean['price_clean'].quantile(0.25)
Q3 = df_clean['price_clean'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtrar outliers
df_no_outliers = df_clean[(df_clean['price_clean'] >= lower_bound) & 
                          (df_clean['price_clean'] <= upper_bound)]

print(f"Rango original: ${df_clean['price_clean'].min():.2f} - ${df_clean['price_clean'].max():.2f}")
print(f"Rango sin outliers: ${lower_bound:.2f} - ${upper_bound:.2f}")
print(f"Datos originales: {len(df_clean)}")
print(f"Datos sin outliers: {len(df_no_outliers)} ({len(df_no_outliers)/len(df_clean)*100:.1f}%)")
print(f"Outliers eliminados: {len(df_clean) - len(df_no_outliers)}")

# Histograma sin outliers
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist(df_clean['price_clean'], bins=50, alpha=0.5, label='Con outliers', color='red')
plt.hist(df_no_outliers['price_clean'], bins=50, alpha=0.7, label='Sin outliers', color='blue')
plt.title('Comparación: Con vs Sin Outliers')
plt.xlabel('Precio ($)')
plt.ylabel('Frecuencia')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(df_no_outliers['price_clean'], bins=30, color='green', alpha=0.7, edgecolor='black')
plt.title('Histograma Sin Outliers')
plt.xlabel('Precio ($)')
plt.ylabel('Frecuencia')
plt.axvline(df_no_outliers['price_clean'].mean(), color='red', linestyle='--', 
           label=f'Promedio: ${df_no_outliers["price_clean"].mean():.2f}')
plt.legend()
plt.tight_layout()
plt.show()

# =================== CANTIDAD DE ALOJAMIENTOS POR TIPO ===================
print("\n" + "="*50)
print("7. CANTIDAD DE ALOJAMIENTOS POR TIPO DE PROPIEDAD")
print("="*50)

property_type_counts = df['property_type'].value_counts().head(15)

print("Top 15 tipos de propiedad:")
for i, (tipo, cantidad) in enumerate(property_type_counts.items(), 1):
    porcentaje = (cantidad / len(df)) * 100
    print(f"{i:2d}. {tipo}: {cantidad} ({porcentaje:.1f}%)")

# Gráficas de tipos de propiedad
plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.barh(property_type_counts.head(10).index[::-1], property_type_counts.head(10).values[::-1], 
         color='purple', alpha=0.7)
plt.title('Top 10 Tipos de Propiedad')
plt.xlabel('Cantidad')

plt.subplot(2, 1, 2)
plt.bar(range(len(property_type_counts)), property_type_counts.values, color='orange', alpha=0.7)
plt.title('Todos los Tipos de Propiedad (Top 15)')
plt.xlabel('Tipo de Propiedad')
plt.ylabel('Cantidad')
plt.xticks(range(len(property_type_counts)), property_type_counts.index, rotation=45, ha='right')
plt.tight_layout()
plt.show()

# =================== DISTRIBUCIÓN GEOGRÁFICA DE PRECIOS ===================
print("\n" + "="*50)
print("8. DISTRIBUCIÓN GEOGRÁFICA DE PRECIOS")
print("="*50)

# Filtrar datos con coordenadas válidas
df_geo = df_clean.dropna(subset=['latitude', 'longitude']).copy()

print(f"Datos con coordenadas válidas: {len(df_geo)}")
print(f"Rango latitud: {df_geo['latitude'].min():.4f} - {df_geo['latitude'].max():.4f}")
print(f"Rango longitud: {df_geo['longitude'].min():.4f} - {df_geo['longitude'].max():.4f}")

# Crear bins de precios para el mapa
df_geo['price_bin'] = pd.cut(df_geo['price_clean'], bins=5, labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'])

# Gráficas geográficas
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
scatter = plt.scatter(df_geo['longitude'], df_geo['latitude'], c=df_geo['price_clean'], 
                     cmap='viridis', alpha=0.6, s=20)
plt.colorbar(scatter, label='Precio ($)')
plt.title('Distribución Geográfica de Precios')
plt.xlabel('Longitud')
plt.ylabel('Latitud')

plt.subplot(2, 2, 2)
# Mapa de calor por vecindario
neighborhood_prices = df_geo.groupby('neighbourhood_cleansed')['price_clean'].mean().sort_values(ascending=False).head(10)
plt.barh(neighborhood_prices.index[::-1], neighborhood_prices.values[::-1], color='coral')
plt.title('Precio Promedio por Vecindario (Top 10)')
plt.xlabel('Precio Promedio ($)')

plt.subplot(2, 2, 3)
# Distribución de precios por zona geográfica (dividir en cuadrantes)
df_geo['lat_bin'] = pd.cut(df_geo['latitude'], bins=3, labels=['Sur', 'Centro', 'Norte'])
df_geo['lon_bin'] = pd.cut(df_geo['longitude'], bins=3, labels=['Oeste', 'Centro', 'Este'])
df_geo['zona'] = df_geo['lat_bin'].astype(str) + '-' + df_geo['lon_bin'].astype(str)

zona_prices = df_geo.groupby('zona')['price_clean'].mean().sort_values(ascending=False)
plt.bar(zona_prices.index, zona_prices.values, color='lightblue')
plt.title('Precio Promedio por Zona Geográfica')
plt.xlabel('Zona')
plt.ylabel('Precio Promedio ($)')
plt.xticks(rotation=45)

plt.subplot(2, 2, 4)
# Densidad de alojamientos por zona
zona_counts = df_geo['zona'].value_counts()
plt.bar(zona_counts.index, zona_counts.values, color='lightgreen')
plt.title('Distribución de Alojamientos por Zona')
plt.xlabel('Zona')
plt.ylabel('Cantidad de Alojamientos')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


print("ANÁLISIS FINAL")
print("-------------------------")

print(f" DATOS GENERALES:")
print(f"   • Total de alojamientos: {len(df):,}")
print(f"   • Precio promedio: ${precio_promedio:.2f}")
print(f"   • Precio mediano: ${precio_mediano:.2f}")

print(f"\n TIPOS DE ALOJAMIENTO:")
print(f"   • Tipo más común: {tipo_mas_comun} ({(cantidad_mas_comun/len(df)*100):.1f}%)")
print(f"   • Tipo de propiedad más común: {property_type_counts.index[0]} ({property_type_counts.iloc[0]} unidades)")

print(f"\n UBICACIONES:")
print(f"   • Alcaldía con más alojamientos: {neighbourhood_counts.index[0]} ({neighbourhood_counts.iloc[0]} unidades)")
print(f"   • Total de alcaldías/vecindarios: {df['neighbourhood_cleansed'].nunique()}")

print(f"\n HOSTS:")
print(f"   • Total de hosts: {df['host_name'].nunique():,}")
print(f"   • Host con más alojamientos: {host_counts.index[0]} ({host_counts.iloc[0]} alojamientos)")
print(f"   • Promedio de alojamientos por host: {df.groupby('host_name').size().mean():.1f}")

print(f"\n PRECIOS:")
print(f"   • Rango de precios: ${df_clean['price_clean'].min():.2f} - ${df_clean['price_clean'].max():.2f}")
print(f"   • Datos sin outliers: {len(df_no_outliers):,} ({len(df_no_outliers)/len(df_clean)*100:.1f}%)")
print(f"   • Vecindario más caro: {neighborhood_prices.index[0]} (${neighborhood_prices.iloc[0]:.2f} promedio)")


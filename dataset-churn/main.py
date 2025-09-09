import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configurar gráficos
plt.style.use('default')

# CARGAR LOS DATOS
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

print("Dataset cargado exitosamente!")
print(f"Dimensiones del dataset: {df.shape}")
print("\nInformación del dataset:")
print(df.info())
print("\nPrimeras 5 filas:")
print(df.head())

# LIMPIEZA DE DATOS
# TotalCharges tiene algunos valores en blanco
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

print(f"\nValores nulos después de la limpieza:")
print(df.isnull().sum())

# PREGUNTA 1: PERFIL GENERAL DEL CLIENTE
print("1. PERFIL GENERAL DEL CLIENTE")
print("-"*40)

# Calcular estadísticas descriptivas para las columnas numéricas principales
numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
desc_stats = df[numerical_columns].describe()

print("\nEstadísticas descriptivas para tenure, MonthlyCharges y TotalCharges:")
print(desc_stats.round(2))

# Comparar media y mediana para tenure
tenure_mean = df['tenure'].mean()
tenure_median = df['tenure'].median()

print(f"\nAnálisis detallado de tenure:")
print(f"Media: {tenure_mean:.2f} meses")
print(f"Mediana: {tenure_median:.2f} meses")
print(f"Desviación estándar: {df['tenure'].std():.2f} meses")

if tenure_mean > tenure_median:
    distribution_type = "asimétrica hacia la derecha (cola larga hacia valores altos)"
    customer_profile = "La mayoría de clientes son relativamente nuevos, pero hay algunos clientes muy antiguos que elevan la media"
elif tenure_mean < tenure_median:
    distribution_type = "asimétrica hacia la izquierda (cola larga hacia valores bajos)"
    customer_profile = "La mayoría de clientes tienen más tiempo, pero hay muchos clientes nuevos que reducen la media"
else:
    distribution_type = "aproximadamente simétrica"
    customer_profile = "Los clientes están distribuidos de manera equilibrada en términos de permanencia"

print(f"\nInterpretación:")
print(f"- La distribución es {distribution_type}")
print(f"- Perfil típico del cliente: {customer_profile}")

# PREGUNTA 2: VISUALIZACIÓN DE DISTRIBUCIONES
print("2. VISUALIZACIÓN DE DISTRIBUCIONES")
print("-"*40)

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Distribuciones de Tenure y MonthlyCharges', fontsize=16, fontweight='bold')

# Histograma de tenure
axes[0,0].hist(df['tenure'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Distribución de Tenure (Permanencia)', fontweight='bold')
axes[0,0].set_xlabel('Tenure (meses)')
axes[0,0].set_ylabel('Frecuencia')
axes[0,0].axvline(tenure_mean, color='red', linestyle='--', label=f'Media: {tenure_mean:.1f}')
axes[0,0].axvline(tenure_median, color='green', linestyle='--', label=f'Mediana: {tenure_median:.1f}')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Histograma de MonthlyCharges
monthly_mean = df['MonthlyCharges'].mean()
monthly_median = df['MonthlyCharges'].median()

axes[0,1].hist(df['MonthlyCharges'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
axes[0,1].set_title('Distribución de MonthlyCharges (Cargos Mensuales)', fontweight='bold')
axes[0,1].set_xlabel('MonthlyCharges ($)')
axes[0,1].set_ylabel('Frecuencia')
axes[0,1].axvline(monthly_mean, color='red', linestyle='--', label=f'Media: ${monthly_mean:.1f}')
axes[0,1].axvline(monthly_median, color='green', linestyle='--', label=f'Mediana: ${monthly_median:.1f}')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Box plot de tenure
box1 = axes[1,0].boxplot(df['tenure'], vert=True, patch_artist=True, 
                         boxprops=dict(facecolor='skyblue', alpha=0.7))
axes[1,0].set_title('Box Plot - Tenure', fontweight='bold')
axes[1,0].set_ylabel('Tenure (meses)')
axes[1,0].grid(True, alpha=0.3)

# Box plot de MonthlyCharges
box2 = axes[1,1].boxplot(df['MonthlyCharges'], vert=True, patch_artist=True,
                         boxprops=dict(facecolor='lightcoral', alpha=0.7))
axes[1,1].set_title('Box Plot - MonthlyCharges', fontweight='bold')
axes[1,1].set_ylabel('MonthlyCharges ($)')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Análisis de la forma de las distribuciones
print("\nAnálisis de la forma de las distribuciones:")

# Analizar tenure
tenure_skew = stats.skew(df['tenure'])
tenure_kurtosis = stats.kurtosis(df['tenure'])
print(f"\nTenure:")
print(f"Asimetría (skewness): {tenure_skew:.3f}")
print(f"Curtosis (kurtosis): {tenure_kurtosis:.3f}")

if abs(tenure_skew) < 0.5:
    tenure_shape = "aproximadamente simétrica"
elif tenure_skew > 0.5:
    tenure_shape = "asimétrica hacia la derecha (cola larga hacia valores altos)"
else:
    tenure_shape = "asimétrica hacia la izquierda (cola larga hacia valores bajos)"
print(f"Forma: {tenure_shape}")

# Analizar MonthlyCharges
monthly_skew = stats.skew(df['MonthlyCharges'])
monthly_kurtosis = stats.kurtosis(df['MonthlyCharges'])
print(f"\nMonthlyCharges:")
print(f"Asimetría (skewness): {monthly_skew:.3f}")
print(f"Curtosis (kurtosis): {monthly_kurtosis:.3f}")

if abs(monthly_skew) < 0.5:
    monthly_shape = "aproximadamente simétrica"
elif monthly_skew > 0.5:
    monthly_shape = "asimétrica hacia la derecha (cola larga hacia valores altos)"
else:
    monthly_shape = "asimétrica hacia la izquierda (cola larga hacia valores bajos)"
print(f"Forma: {monthly_shape}")

# Identificar patrones y picos interesantes
print(f"\nPatrones observados:")
print(f"- Tenure: Rango de {df['tenure'].min():.0f} a {df['tenure'].max():.0f} meses")
print(f"- MonthlyCharges: Rango de ${df['MonthlyCharges'].min():.2f} a ${df['MonthlyCharges'].max():.2f}")

# PREGUNTA 3: ANÁLISIS POR SEGMENTOS (CHURN VS NO CHURN)
print("3. ANÁLISIS POR SEGMENTOS (CHURN VS NO CHURN)")
print("-"*40)

# Verificar la distribución de Churn
print("Distribución de Churn:")
churn_counts = df['Churn'].value_counts()
print(churn_counts)
print(f"Porcentaje de Churn: {churn_counts['Yes']/len(df)*100:.1f}%")

# Agrupar por Churn y calcular estadísticas
churn_stats = df.groupby('Churn')[['tenure', 'MonthlyCharges']].agg(['median', 'mean', 'std', 'count']).round(2)
print(f"\nEstadísticas por grupo de Churn:")
print(churn_stats)

# Análisis específico de medianas
churn_yes_tenure_median = df[df['Churn'] == 'Yes']['tenure'].median()
churn_no_tenure_median = df[df['Churn'] == 'No']['tenure'].median()
churn_yes_monthly_median = df[df['Churn'] == 'Yes']['MonthlyCharges'].median()
churn_no_monthly_median = df[df['Churn'] == 'No']['MonthlyCharges'].median()

print(f"\nComparación de medianas:")
print(f"Tenure - Churn Yes: {churn_yes_tenure_median:.1f} meses vs Churn No: {churn_no_tenure_median:.1f} meses")
print(f"Diferencia: {churn_no_tenure_median - churn_yes_tenure_median:.1f} meses")
print(f"MonthlyCharges - Churn Yes: ${churn_yes_monthly_median:.2f} vs Churn No: ${churn_no_monthly_median:.2f}")
print(f"Diferencia: ${churn_yes_monthly_median - churn_no_monthly_median:.2f}")

# Interpretación de diferencias
print(f"\nDiferencias clave observadas:")
if churn_yes_tenure_median < churn_no_tenure_median:
    print(f"→ Los clientes que se van (churn) tienen {churn_no_tenure_median - churn_yes_tenure_median:.1f} meses MENOS de permanencia en promedio")
if churn_yes_monthly_median > churn_no_monthly_median:
    print(f"→ Los clientes que se van (churn) pagan ${churn_yes_monthly_median - churn_no_monthly_median:.2f} MÁS mensualmente")

# Crear visualizaciones comparativas
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Box plot para MonthlyCharges por Churn
sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[0])
axes[0].set_title('Distribución de MonthlyCharges por Churn', fontweight='bold', fontsize=14)
axes[0].set_xlabel('Churn', fontsize=12)
axes[0].set_ylabel('MonthlyCharges ($)', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Box plot para Tenure por Churn
sns.boxplot(data=df, x='Churn', y='tenure', ax=axes[1])
axes[1].set_title('Distribución de Tenure por Churn', fontweight='bold', fontsize=14)
axes[1].set_xlabel('Churn', fontsize=12)
axes[1].set_ylabel('Tenure (meses)', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Análisis estadístico adicional
from scipy.stats import mannwhitneyu

# Test de Mann-Whitney U para comparar distribuciones
statistic_monthly, p_value_monthly = mannwhitneyu(
    df[df['Churn'] == 'Yes']['MonthlyCharges'], 
    df[df['Churn'] == 'No']['MonthlyCharges'],
    alternative='two-sided'
)

statistic_tenure, p_value_tenure = mannwhitneyu(
    df[df['Churn'] == 'Yes']['tenure'], 
    df[df['Churn'] == 'No']['tenure'],
    alternative='two-sided'
)

print(f"\nPruebas estadísticas (Mann-Whitney U):")
print(f"MonthlyCharges - p-valor: {p_value_monthly:.2e} ({'Significativo' if p_value_monthly < 0.05 else 'No significativo'})")
print(f"Tenure - p-valor: {p_value_tenure:.2e} ({'Significativo' if p_value_tenure < 0.05 else 'No significativo'})")

# RESUMEN FINAL
print("RESUMEN")
print('------------------------')
print(f"1. El cliente típico tiene una permanencia mediana de {tenure_median:.1f} meses")
print(f"2. La distribución de tenure es {tenure_shape}")
print(f"3. La distribución de MonthlyCharges es {monthly_shape}")
print(f"4. Los clientes que hacen churn tienen {churn_no_tenure_median - churn_yes_tenure_median:.1f} meses menos de permanencia")
print(f"5. Los clientes que hacen churn pagan ${churn_yes_monthly_median - churn_no_monthly_median:.2f} más por mes")
print(f"6. Ambas diferencias son estadísticamente significativas")
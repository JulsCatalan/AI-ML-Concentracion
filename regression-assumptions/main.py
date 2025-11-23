
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats


class ResidualsAnalyzer:
    """Clase para análisis completo de residuales y diagnóstico de regresión"""
    
    def __init__(self, X_train, X_test, y_train, y_test, dataset_name="Dataset"):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.dataset_name = dataset_name
        self.model = None
        self.y_pred = None
        self.residuals = None
        
    def fit_model(self):
        """Entrenar modelo de regresión lineal"""
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        self.residuals = self.y_test - self.y_pred
        return self
    
    def calculate_metrics(self):
        """Calcular métricas de rendimiento del modelo"""
        mse = mean_squared_error(self.y_test, self.y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.y_pred)
        r2 = r2_score(self.y_test, self.y_pred)
        
        print(f"\n{'='*60}")
        print(f"Métricas del Modelo: {self.dataset_name}")
        print(f"{'='*60}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}
    
    def plot_residuals_vs_fitted(self, ax=None):
        """Gráfico de residuales vs valores ajustados para detectar patrones"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        
        ax.scatter(self.y_pred, self.residuals, alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Residual = 0')
        ax.set_xlabel('Valores Predichos', fontsize=11)
        ax.set_ylabel('Residuales', fontsize=11)
        ax.set_title(f'Residuales vs Valores Ajustados\n{self.dataset_name}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_homoscedasticity(self, ax=None):
        """Gráfico para verificar homocedasticidad (varianza constante)"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        
        ax.scatter(self.y_pred, np.abs(self.residuals), alpha=0.6, edgecolors='k', linewidth=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
        
        # Línea de tendencia para detectar heterocedasticidad
        z = np.polyfit(self.y_pred, np.abs(self.residuals), 2)
        p = np.poly1d(z)
        x_line = np.linspace(self.y_pred.min(), self.y_pred.max(), 100)
        ax.plot(x_line, p(x_line), 'b-', linewidth=2, alpha=0.7, label='Tendencia')
        
        ax.set_xlabel('Valores Predichos', fontsize=11)
        ax.set_ylabel('|Residuales|', fontsize=11)
        ax.set_title(f'Test de Homocedasticidad\n{self.dataset_name}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_residuals_histogram(self, ax=None):
        """Histograma de residuales para verificar normalidad"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        
        ax.hist(self.residuals, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Superponer curva normal
        mu, std = self.residuals.mean(), self.residuals.std()
        x = np.linspace(self.residuals.min(), self.residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, std), 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={std:.2f})')
        
        ax.set_xlabel('Residuales', fontsize=11)
        ax.set_ylabel('Densidad', fontsize=11)
        ax.set_title(f'Distribución de Residuales\n{self.dataset_name}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_qq(self, ax=None):
        """Gráfico Q-Q para verificar normalidad de residuales"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        
        sm.qqplot(self.residuals, line='s', ax=ax)
        ax.set_xlabel('Cuantiles Teóricos', fontsize=11)
        ax.set_ylabel('Cuantiles de Muestra', fontsize=11)
        ax.set_title(f'Gráfico Q-Q\n{self.dataset_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def plot_actual_vs_predicted(self, ax=None):
        """Gráfico de valores reales vs predichos"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        
        ax.scatter(self.y_test, self.y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
        
        # Línea de referencia perfecta
        min_val = min(self.y_test.min(), self.y_pred.min())
        max_val = max(self.y_test.max(), self.y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
        
        ax.set_xlabel('Valores Reales', fontsize=11)
        ax.set_ylabel('Valores Predichos', fontsize=11)
        ax.set_title(f'Valores Reales vs Predichos\n{self.dataset_name}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def complete_diagnostic_plot(self, save_name=None):
        """Crear panel completo de diagnóstico con 4 gráficos"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Diagnóstico Completo de Regresión: {self.dataset_name}', 
                     fontsize=16, fontweight='bold', y=1.00)
        
        self.plot_residuals_vs_fitted(axes[0, 0])
        self.plot_homoscedasticity(axes[0, 1])
        self.plot_residuals_histogram(axes[1, 0])
        self.plot_qq(axes[1, 1])
        
        plt.tight_layout()
        
        plt.show()
    
    def statistical_tests(self):
        """Realizar tests estadísticos sobre los residuales"""
        print(f"\n{'='*60}")
        print(f"Tests Estadísticos: {self.dataset_name}")
        print(f"{'='*60}")
        
        # Test de normalidad Shapiro-Wilk
        shapiro_stat, shapiro_p = stats.shapiro(self.residuals[:min(5000, len(self.residuals))])
        print(f"\nTest de Normalidad (Shapiro-Wilk):")
        print(f"  Estadístico: {shapiro_stat:.4f}")
        print(f"  P-valor: {shapiro_p:.4f}")
        print(f"  Conclusión: {'Los residuales son normales' if shapiro_p > 0.05 else 'Los residuales NO son normales'} (α=0.05)")
        
        # Test de Jarque-Bera para normalidad
        jb_stat, jb_p = stats.jarque_bera(self.residuals)
        print(f"\nTest de Normalidad (Jarque-Bera):")
        print(f"  Estadístico: {jb_stat:.4f}")
        print(f"  P-valor: {jb_p:.4f}")
        print(f"  Conclusión: {'Los residuales son normales' if jb_p > 0.05 else 'Los residuales NO son normales'} (α=0.05)")
        
        # Durbin-Watson para autocorrelación
        dw = sm.stats.stattools.durbin_watson(self.residuals)
        print(f"\nTest de Durbin-Watson (Autocorrelación):")
        print(f"  Estadístico: {dw:.4f}")
        print(f"  Conclusión: ", end="")
        if dw < 1.5:
            print("Autocorrelación positiva")
        elif dw > 2.5:
            print("Autocorrelación negativa")
        else:
            print("No hay autocorrelación significativa")


def analyze_synthetic_dataset_1():
    """Análisis del primer dataset sintético (homocedastico)"""
    print("\n" + "="*60)
    print("ANÁLISIS 1: DATASET SINTÉTICO HOMOCEDÁSTICO")
    print("="*60)
    
    # Generar dataset con varianza constante
    X, y = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Análisis completo
    analyzer = ResidualsAnalyzer(X_train, X_test, y_train, y_test, "Dataset Homocedástico")
    analyzer.fit_model()
    analyzer.calculate_metrics()
    analyzer.complete_diagnostic_plot('diagnostic_homoscedastic.png')
    analyzer.statistical_tests()
    
    return analyzer


def analyze_synthetic_dataset_2():
    """Análisis del segundo dataset sintético (heterocedastico)"""
    print("\n" + "="*60)
    print("ANÁLISIS 2: DATASET SINTÉTICO HETEROCEDÁSTICO")
    print("="*60)
    
    # Generar dataset con varianza creciente (heterocedasticidad)
    X = np.random.rand(200, 1) * 100
    y = 2 * X.squeeze() + np.random.normal(0, X.squeeze(), 200)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Análisis completo
    analyzer = ResidualsAnalyzer(X_train, X_test, y_train, y_test, "Dataset Heterocedástico")
    analyzer.fit_model()
    analyzer.calculate_metrics()
    analyzer.complete_diagnostic_plot('diagnostic_heteroscedastic.png')
    analyzer.statistical_tests()
    
    return analyzer


def analyze_wine_dataset():
    """Análisis del dataset de vinos"""
    print("\n" + "="*60)
    print("ANÁLISIS 3: DATASET DE CALIDAD DE VINOS")
    print("="*60)
    
    try:
        # Cargar dataset de vinos
        df = pd.read_csv("wine.csv")
        
        print(f"\nDimensiones del dataset: {df.shape}")
        print(f"Variables: {list(df.columns)}")
        
        # Preparar datos
        X = df.drop('quality', axis=1).values
        y = df['quality'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Análisis completo
        analyzer = ResidualsAnalyzer(X_train, X_test, y_train, y_test, "Dataset de Vinos")
        analyzer.fit_model()
        analyzer.calculate_metrics()
        analyzer.complete_diagnostic_plot('diagnostic_wine.png')
        analyzer.statistical_tests()
        
        # Gráfico adicional: importancia de features (coeficientes)
        plot_feature_importance(analyzer.model, df.drop('quality', axis=1).columns)
        
        return analyzer
    
    except FileNotFoundError:
        print("\n⚠️ Archivo 'wine.csv' no encontrado.")
        print("Puedes descargar el dataset de:")
        print("https://archive.ics.uci.edu/ml/datasets/wine+quality")
        return None


def plot_feature_importance(model, feature_names):
    """Graficar importancia de características basada en coeficientes"""
    coefs = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_
    })
    coefs['Abs_Coefficient'] = np.abs(coefs['Coefficient'])
    coefs = coefs.sort_values('Abs_Coefficient', ascending=True)
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in coefs['Coefficient']]
    plt.barh(coefs['Feature'], coefs['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Coeficiente', fontsize=12)
    plt.ylabel('Variable', fontsize=12)
    plt.title('Importancia de Variables (Coeficientes del Modelo)', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def compare_models_summary():
    """Generar resumen comparativo de los modelos analizados"""
    print("\n" + "="*60)
    print("GUÍA DE INTERPRETACIÓN DE DIAGNÓSTICOS")
    print("="*60)
    
    guide = """
    
SUPUESTOS DE REGRESIÓN LINEAL:

1. LINEALIDAD
   • Qué verificar: Relación lineal entre X e Y
   • Gráfico: Residuales vs Valores Ajustados
   • Interpretación: Puntos dispersos aleatoriamente alrededor de 0

2. HOMOCEDASTICIDAD (Varianza Constante)
   • Qué verificar: Varianza constante de residuales
   • Gráfico: |Residuales| vs Valores Ajustados
   • Interpretación: Dispersión uniforme sin patrón de embudo
   • Problema: Si la varianza aumenta/disminuye sistemáticamente

3. NORMALIDAD DE RESIDUALES
   • Qué verificar: Residuales siguen distribución normal
   • Gráficos: Histograma y Q-Q Plot
   • Interpretación (Histograma): Forma de campana
   • Interpretación (Q-Q): Puntos cerca de la línea diagonal
   • Test: Shapiro-Wilk (p > 0.05 = normalidad)

4. INDEPENDENCIA
   • Qué verificar: No autocorrelación de residuales
   • Test: Durbin-Watson (valor ideal ≈ 2)
   • Interpretación: DW < 1.5 = autocorrelación positiva
                     DW > 2.5 = autocorrelación negativa

SOLUCIONES A PROBLEMAS COMUNES:

• Heterocedasticidad → Transformación log, raíz cuadrada, o modelos WLS
• No normalidad → Transformaciones Box-Cox, aumentar n, o usar modelos robustos
• No linealidad → Agregar términos polinomiales o usar modelos no lineales
• Autocorrelación → Incluir rezagos, diferenciación temporal
    """
    
    print(guide)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETO DE RESIDUALES Y SUPUESTOS DE REGRESIÓN")
    print("="*60)
    
    # Análisis de los tres datasets
    analyzer1 = analyze_synthetic_dataset_1()
    analyzer2 = analyze_synthetic_dataset_2()
    analyzer3 = analyze_wine_dataset()
    
    # Guía de interpretación
    compare_models_summary()
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
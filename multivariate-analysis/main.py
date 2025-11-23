import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class LeverageAnalyzer:
    """Clase para análisis de leverage points y detección de observaciones influyentes"""
    
    def __init__(self, X, y, feature_names=None):
        self.X = X
        self.y = y.reshape(-1, 1) if y.ndim == 1 else y
        self.n, self.p = X.shape
        self.feature_names = feature_names or [f"X{i+1}" for i in range(self.p)]
        
        # Inicializar atributos
        self.beta = None
        self.y_hat = None
        self.residuals = None
        self.H = None
        self.leverage = None
        self.avg_leverage = self.p / self.n
        self.threshold = 2 * self.avg_leverage
    
    def fit(self):
        """Calcular coeficientes de regresión y valores ajustados"""
        # Calcular beta usando ecuaciones normales: beta = (X'X)^-1 X'y
        XtX = self.X.T @ self.X
        XtX_inv = np.linalg.inv(XtX)
        self.beta = XtX_inv @ self.X.T @ self.y
        
        # Predicciones y residuales
        self.y_hat = self.X @ self.beta
        self.residuals = self.y - self.y_hat
        
        # Hat Matrix: H = X(X'X)^-1X'
        self.H = self.X @ XtX_inv @ self.X.T
        
        # Leverage values (diagonal de H)
        self.leverage = np.diag(self.H)
        
        return self
    
    def get_summary_table(self):
        """Crear tabla resumen con leverage, predicciones y residuales"""
        summary = pd.DataFrame({
            'Index': np.arange(self.n),
            'Leverage': self.leverage,
            'Y_Actual': self.y.flatten(),
            'Y_Predicted': self.y_hat.flatten(),
            'Residual': self.residuals.flatten()
        })
        
        # Añadir flags de leverage alto
        summary['High_Leverage'] = summary['Leverage'] > self.threshold
        
        return summary
    
    def get_high_leverage_points(self, top_n=10):
        """Obtener los puntos con mayor leverage"""
        summary = self.get_summary_table()
        high_lev = summary.sort_values('Leverage', ascending=False).head(top_n)
        
        print(f"\n{'='*60}")
        print(f"TOP {top_n} LEVERAGE POINTS")
        print(f"{'='*60}")
        print(high_lev.round(4).to_string(index=False))
        print(f"\nTotal de puntos con leverage alto (> {self.threshold:.4f}): "
              f"{(self.leverage > self.threshold).sum()}")
        
        return high_lev
    
    def plot_leverage(self):
        """Gráfico de leverage points"""
        summary = self.get_summary_table()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Subplot 1: Leverage por índice
        ax1 = axes[0]
        normal_idx = summary[~summary['High_Leverage']]['Index']
        high_idx = summary[summary['High_Leverage']]['Index']
        
        ax1.scatter(normal_idx, summary.loc[normal_idx, 'Leverage'], 
                   s=30, alpha=0.6, color='steelblue', label='Normal')
        ax1.scatter(high_idx, summary.loc[high_idx, 'Leverage'], 
                   s=80, alpha=0.8, color='red', edgecolors='black', 
                   linewidth=1, label='High Leverage')
        
        ax1.axhline(self.threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {self.threshold:.3f}')
        ax1.axhline(self.avg_leverage, color='orange', linestyle='--', linewidth=2, 
                   label=f'Average: {self.avg_leverage:.3f}')
        
        ax1.set_xlabel('Observation Index', fontsize=11)
        ax1.set_ylabel('Leverage (hᵢᵢ)', fontsize=11)
        ax1.set_title('Leverage Points Analysis', fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Histograma de leverage
        ax2 = axes[1]
        ax2.hist(self.leverage, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(self.threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {self.threshold:.3f}')
        ax2.axvline(self.avg_leverage, color='orange', linestyle='--', linewidth=2, 
                   label=f'Average: {self.avg_leverage:.3f}')
        
        ax2.set_xlabel('Leverage (hᵢᵢ)', fontsize=11)
        ax2.set_ylabel('Frequency', fontsize=11)
        ax2.set_title('Distribution of Leverage Values', fontsize=13, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals_vs_leverage(self):
        """Gráfico de residuales estandarizados vs leverage"""
        summary = self.get_summary_table()
        
        # Residuales estandarizados
        std_residuals = self.residuals.flatten() / np.std(self.residuals)
        
        plt.figure(figsize=(10, 6))
        
        # Puntos normales
        normal_mask = ~summary['High_Leverage']
        plt.scatter(summary[normal_mask]['Leverage'], 
                   std_residuals[normal_mask],
                   s=30, alpha=0.6, color='steelblue', label='Normal')
        
        # Puntos con alto leverage
        high_mask = summary['High_Leverage']
        plt.scatter(summary[high_mask]['Leverage'], 
                   std_residuals[high_mask],
                   s=100, alpha=0.8, color='red', edgecolors='black', 
                   linewidth=1, label='High Leverage')
        
        # Líneas de referencia
        plt.axvline(self.threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Leverage Threshold: {self.threshold:.3f}')
        plt.axhline(2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        plt.axhline(-2, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, 
                   label='±2 Std Residuals')
        plt.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        
        plt.xlabel('Leverage (hᵢᵢ)', fontsize=11)
        plt.ylabel('Standardized Residuals', fontsize=11)
        plt.title('Residuals vs Leverage\n(Detecting Influential Points)', 
                 fontsize=13, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def calculate_cooks_distance(self):
        """Calcular distancia de Cook para identificar puntos influyentes"""
        # D_i = (e_i^2 / (p * MSE)) * (h_ii / (1 - h_ii)^2)
        mse = np.sum(self.residuals ** 2) / (self.n - self.p)
        
        cooks_d = (self.residuals.flatten() ** 2 / (self.p * mse)) * \
                  (self.leverage / (1 - self.leverage) ** 2)
        
        return cooks_d
    
    def plot_cooks_distance(self):
        """Gráfico de distancia de Cook"""
        cooks_d = self.calculate_cooks_distance()
        threshold_cooks = 4 / self.n  # Threshold común para Cook's D
        
        plt.figure(figsize=(10, 6))
        
        # Barras normales vs influyentes
        colors = ['red' if d > threshold_cooks else 'steelblue' for d in cooks_d]
        plt.bar(range(self.n), cooks_d, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        plt.axhline(threshold_cooks, color='red', linestyle='--', linewidth=2, 
                   label=f"Threshold: {threshold_cooks:.4f}")
        
        plt.xlabel('Observation Index', fontsize=11)
        plt.ylabel("Cook's Distance", fontsize=11)
        plt.title("Cook's Distance\n(Measure of Influence)", fontsize=13, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        
        # Resumen
        influential = cooks_d > threshold_cooks
        print(f"\n{'='*60}")
        print("COOK'S DISTANCE ANALYSIS")
        print(f"{'='*60}")
        print(f"Threshold: {threshold_cooks:.4f}")
        print(f"Influential points (D > threshold): {influential.sum()}")
        if influential.sum() > 0:
            print(f"Indices: {np.where(influential)[0].tolist()}")
    
    def complete_diagnostic(self):
        """Generar diagnóstico completo de leverage e influencia"""
        print(f"\n{'='*60}")
        print("LEVERAGE AND INFLUENCE DIAGNOSTICS")
        print(f"{'='*60}")
        print(f"Number of observations: {self.n}")
        print(f"Number of predictors: {self.p}")
        print(f"Average leverage: {self.avg_leverage:.4f}")
        print(f"High leverage threshold: {self.threshold:.4f}")
        
        # Top leverage points
        self.get_high_leverage_points(top_n=10)
        
        # Gráficos
        self.plot_leverage()
        self.plot_residuals_vs_leverage()
        self.plot_cooks_distance()


def generate_synthetic_data(n=500, seed=42):
    """Generar dataset sintético con características realistas"""
    np.random.seed(seed)
    
    # Variables independientes
    MedInc = np.random.lognormal(mean=2.5, sigma=0.35, size=n)
    HouseAge = np.random.randint(1, 52, size=n)
    Longitude = -124 + np.random.rand(n) * 10
    Latitude = 32 + np.random.rand(n) * 8
    AveRooms = 5 + 0.15*MedInc + 0.03*HouseAge + np.random.rand(n)
    
    # Variable dependiente con relación lineal
    target = (
        0.45*MedInc
        - 0.02*HouseAge
        - 0.3*(Latitude-36)
        - 0.1*(Longitude+119)
        + 0.02*AveRooms
        + np.random.rand(n)*0.5
    )
    
    # Crear DataFrame
    df = pd.DataFrame({
        'MedInc': MedInc,
        'HouseAge': HouseAge,
        'Longitude': Longitude,
        'Latitude': Latitude,
        'AveRooms': AveRooms,
        'Target': target
    })
    
    return df


def analyze_correlations(df):
    """Análisis de correlaciones y covarianzas"""
    print(f"\n{'='*60}")
    print("CORRELATION AND COVARIANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Matriz de covarianza
    cov_matrix = df.cov()
    print("\nCovariance Matrix:")
    print(cov_matrix.round(3))
    
    # Matriz de correlación
    corr_matrix = df.corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(3))
    
    # Visualización de matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Heatmap de correlación
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, ax=axes[0],
                cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Correlation Matrix', fontsize=13, fontweight='bold')
    
    # Heatmap de covarianza
    sns.heatmap(cov_matrix, annot=True, fmt='.1f', cmap='viridis', 
                square=True, ax=axes[1],
                cbar_kws={'label': 'Covariance'})
    axes[1].set_title('Covariance Matrix', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Identificar correlaciones altas con el target
    print(f"\n{'='*60}")
    print("CORRELATIONS WITH TARGET")
    print(f"{'='*60}")
    target_corr = corr_matrix['Target'].drop('Target').sort_values(ascending=False)
    for var, corr in target_corr.items():
        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
        direction = "positive" if corr > 0 else "negative"
        print(f"{var:12s}: {corr:7.3f}  ({strength} {direction})")
    
    return cov_matrix, corr_matrix


def compare_with_sklearn(X, y):
    """Comparar resultados con sklearn LinearRegression"""
    print(f"\n{'='*60}")
    print("COMPARISON WITH SKLEARN")
    print(f"{'='*60}")
    
    # Nuestro método (ecuaciones normales)
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_manual = XtX_inv @ X.T @ y.reshape(-1, 1)
    
    # Sklearn
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    beta_sklearn = model.coef_.reshape(-1, 1)
    
    # Comparación
    comparison = pd.DataFrame({
        'Manual': beta_manual.flatten(),
        'Sklearn': beta_sklearn.flatten(),
        'Difference': (beta_manual - beta_sklearn).flatten()
    })
    
    print("\nCoefficients Comparison:")
    print(comparison.round(6))
    print(f"\nMax difference: {abs(comparison['Difference']).max():.2e}")


def main():
    """Función principal para ejecutar el análisis completo"""
    
    print("\n" + "="*60)
    print("LEVERAGE POINTS AND HAT MATRIX ANALYSIS")
    print("="*60)
    
    # Generar datos
    df = generate_synthetic_data(n=500, seed=42)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    # Análisis de correlaciones
    cov_matrix, corr_matrix = analyze_correlations(df)
    
    # Preparar datos para regresión
    feature_cols = ['MedInc', 'HouseAge', 'Longitude', 'Latitude', 'AveRooms']
    X = df[feature_cols].values
    y = df['Target'].values
    
    # Análisis de leverage
    analyzer = LeverageAnalyzer(X, y, feature_names=feature_cols)
    analyzer.fit()
    
    # Comparación con sklearn
    compare_with_sklearn(X, y)
    
    # Diagnóstico completo
    analyzer.complete_diagnostic()
    
    # Propiedades de Hat Matrix
    print(f"\n{'='*60}")
    print("HAT MATRIX PROPERTIES")
    print(f"{'='*60}")
    print(f"Shape: {analyzer.H.shape}")
    print(f"Trace (sum of leverages): {np.trace(analyzer.H):.4f}")
    print(f"Expected trace (p): {analyzer.p}")
    print(f"Is symmetric: {np.allclose(analyzer.H, analyzer.H.T)}")
    print(f"Is idempotent (H @ H = H): {np.allclose(analyzer.H @ analyzer.H, analyzer.H)}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
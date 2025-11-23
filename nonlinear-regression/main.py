# Modelos de Regresión: Polinomial, Exponencial y Potencia

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import os

# Configuración global para gráficos
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)


class RegressionAnalyzer:
    """Clase para análisis de regresión con múltiples modelos"""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Calcula métricas de evaluación para un modelo"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        self.metrics[model_name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
        return self.metrics[model_name]
    
    def print_metrics(self):
        """Imprime todas las métricas calculadas"""
        print("\n" + "="*60)
        print("RESUMEN DE MÉTRICAS DE LOS MODELOS")
        print("="*60)
        
        for model_name, metrics in self.metrics.items():
            print(f"\n{model_name}:")
            print(f"  MSE:  {metrics['MSE']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE:  {metrics['MAE']:.4f}")
            print(f"  R²:   {metrics['R²']:.4f}")


# =============================================================================
# 1. MODELO POLINOMIAL - Dataset de Rendimiento Académico
# =============================================================================

def polynomial_regression_analysis():
    """Análisis de regresión polinomial con dataset mejorado"""
    
    print("\n" + "="*60)
    print("1. ANÁLISIS DE REGRESIÓN POLINOMIAL")
    print("="*60)
    
    # Dataset más realista con patrón no lineal
    np.random.seed(42)
    study_hours = np.linspace(0, 12, 30)
    
    # Modelo real: rendimiento aumenta hasta un punto óptimo, luego decae (burnout)
    exam_score = 35 + 15*study_hours - 0.8*study_hours**2 + np.random.normal(0, 3, 30)
    exam_score = np.clip(exam_score, 0, 100)
    
    data = pd.DataFrame({
        'Study_Hours': study_hours,
        'Exam_Score': exam_score
    })
    
    X = data[['Study_Hours']].values
    y = data['Exam_Score'].values
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    analyzer = RegressionAnalyzer()
    
    # Modelo Lineal
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    analyzer.evaluate_model(y_test, y_pred_linear, 'Regresión Lineal')
    
    # Modelo Polinomial (grado 2)
    poly2 = PolynomialFeatures(degree=2)
    X_poly2_train = poly2.fit_transform(X_train)
    X_poly2_test = poly2.transform(X_test)
    poly2_model = LinearRegression()
    poly2_model.fit(X_poly2_train, y_train)
    y_pred_poly2 = poly2_model.predict(X_poly2_test)
    analyzer.evaluate_model(y_test, y_pred_poly2, 'Polinomial Grado 2')
    
    # Modelo Polinomial (grado 3)
    poly3 = PolynomialFeatures(degree=3)
    X_poly3_train = poly3.fit_transform(X_train)
    X_poly3_test = poly3.transform(X_test)
    poly3_model = LinearRegression()
    poly3_model.fit(X_poly3_train, y_train)
    y_pred_poly3 = poly3_model.predict(X_poly3_test)
    analyzer.evaluate_model(y_test, y_pred_poly3, 'Polinomial Grado 3')
    
    # Visualización
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Comparación de modelos
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, s=50, label='Datos entrenamiento')
    plt.scatter(X_test, y_test, color='red', alpha=0.6, s=50, label='Datos prueba')
    
    # Líneas de predicción
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    plt.plot(X_range, linear_model.predict(X_range), 
             'g--', linewidth=2, label='Lineal', alpha=0.8)
    plt.plot(X_range, poly2_model.predict(poly2.transform(X_range)), 
             'orange', linewidth=2, label='Polinomial (grado 2)', alpha=0.8)
    plt.plot(X_range, poly3_model.predict(poly3.transform(X_range)), 
             'purple', linewidth=2, label='Polinomial (grado 3)', alpha=0.8)
    
    plt.xlabel('Horas de Estudio', fontsize=12)
    plt.ylabel('Calificación del Examen', fontsize=12)
    plt.title('Comparación de Modelos de Regresión', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Residuales
    plt.subplot(1, 2, 2)
    residuals_linear = y_test - y_pred_linear
    residuals_poly2 = y_test - y_pred_poly2
    residuals_poly3 = y_test - y_pred_poly3
    
    plt.scatter(y_pred_linear, residuals_linear, alpha=0.6, label='Lineal')
    plt.scatter(y_pred_poly2, residuals_poly2, alpha=0.6, label='Polinomial 2')
    plt.scatter(y_pred_poly3, residuals_poly3, alpha=0.6, label='Polinomial 3')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=2)
    plt.xlabel('Valores Predichos', fontsize=12)
    plt.ylabel('Residuales', fontsize=12)
    plt.title('Análisis de Residuales', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Ecuaciones
    print("\nEcuaciones de los modelos:")
    print(f"Lineal: y = {linear_model.intercept_:.2f} + {linear_model.coef_[0]:.2f}x")
    
    coefs2 = poly2_model.coef_
    print(f"Polinomial (grado 2): y = {poly2_model.intercept_:.2f} + "
          f"{coefs2[1]:.2f}x + {coefs2[2]:.2f}x²")
    
    coefs3 = poly3_model.coef_
    print(f"Polinomial (grado 3): y = {poly3_model.intercept_:.2f} + "
          f"{coefs3[1]:.2f}x + {coefs3[2]:.2f}x² + {coefs3[3]:.2f}x³")
    
    analyzer.print_metrics()
    
    return analyzer


# =============================================================================
# 2. MODELO EXPONENCIAL - Crecimiento de Usuarios
# =============================================================================

def exponential_regression_analysis():
    """Análisis de regresión exponencial con dataset de crecimiento viral"""
    
    print("\n" + "="*60)
    print("2. ANÁLISIS DE REGRESIÓN EXPONENCIAL")
    print("="*60)
    
    # Dataset más extenso con crecimiento exponencial
    np.random.seed(42)
    days = np.arange(0, 60, 2)
    
    # Modelo real: crecimiento exponencial con ruido
    users = 50 * np.exp(0.08 * days) + np.random.normal(0, 20, len(days))
    users = np.maximum(users, 0)
    
    data = pd.DataFrame({
        'Days': days,
        'Active_Users': users
    })
    
    X = data[['Days']].values
    y = data['Active_Users'].values
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    analyzer = RegressionAnalyzer()
    
    # Modelo Lineal
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    analyzer.evaluate_model(y_test, y_pred_linear, 'Regresión Lineal')
    
    # Modelo Exponencial
    # log(y) = a + bx => y = exp(a) * exp(bx)
    y_train_log = np.log(y_train + 1)  # +1 para evitar log(0)
    exp_model = LinearRegression()
    exp_model.fit(X_train, y_train_log)
    y_pred_exp_log = exp_model.predict(X_test)
    y_pred_exp = np.exp(y_pred_exp_log) - 1
    analyzer.evaluate_model(y_test, y_pred_exp, 'Regresión Exponencial')
    
    # Modelo Polinomial (grado 2) para comparación
    poly = PolynomialFeatures(degree=2)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    poly_model = LinearRegression()
    poly_model.fit(X_poly_train, y_train)
    y_pred_poly = poly_model.predict(X_poly_test)
    analyzer.evaluate_model(y_test, y_pred_poly, 'Regresión Polinomial')
    
    # Visualización
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Datos y modelos
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.6, s=50, label='Datos entrenamiento')
    plt.scatter(X_test, y_test, color='red', alpha=0.6, s=50, label='Datos prueba')
    
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    plt.plot(X_range, linear_model.predict(X_range), 
             'g--', linewidth=2, label='Lineal', alpha=0.8)
    
    y_exp_range = np.exp(exp_model.predict(X_range)) - 1
    plt.plot(X_range, y_exp_range, 
             'orange', linewidth=2, label='Exponencial', alpha=0.8)
    
    plt.plot(X_range, poly_model.predict(poly.transform(X_range)), 
             'purple', linewidth=2, label='Polinomial', alpha=0.8)
    
    plt.xlabel('Días desde el lanzamiento', fontsize=12)
    plt.ylabel('Usuarios Activos', fontsize=12)
    plt.title('Crecimiento de Usuarios: Comparación de Modelos', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Escala logarítmica
    plt.subplot(1, 2, 2)
    plt.scatter(X_train, np.log(y_train + 1), color='blue', alpha=0.6, s=50, label='Log(Datos entrenamiento)')
    plt.scatter(X_test, np.log(y_test + 1), color='red', alpha=0.6, s=50, label='Log(Datos prueba)')
    plt.plot(X_range, exp_model.predict(X_range), 
             'orange', linewidth=2, label='Ajuste exponencial (escala log)', alpha=0.8)
    
    plt.xlabel('Días desde el lanzamiento', fontsize=12)
    plt.ylabel('Log(Usuarios Activos)', fontsize=12)
    plt.title('Visualización en Escala Logarítmica', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Ecuación exponencial
    a = exp_model.intercept_
    b = exp_model.coef_[0]
    c = np.exp(a)
    
    print("\nEcuaciones de los modelos:")
    print(f"Lineal: y = {linear_model.intercept_:.2f} + {linear_model.coef_[0]:.2f}x")
    print(f"Exponencial: y = {c:.2f} * e^({b:.4f}x)")
    
    # Predicción futura
    future_days = np.array([[70], [80], [90]])
    future_pred = np.exp(exp_model.predict(future_days)) - 1
    
    print("\nPredicciones futuras (modelo exponencial):")
    for day, pred in zip(future_days.flatten(), future_pred):
        print(f"  Día {day}: {pred:.0f} usuarios")
    
    analyzer.print_metrics()
    
    return analyzer


# =============================================================================
# 3. MODELO DE POTENCIA - Engagement en Redes Sociales
# =============================================================================

def power_regression_analysis():
    """Análisis de regresión potencial para métricas de redes sociales"""
    
    print("\n" + "="*60)
    print("3. ANÁLISIS DE REGRESIÓN DE POTENCIA")
    print("="*60)
    
    # Dataset más realista con múltiples métricas
    np.random.seed(42)
    followers = np.logspace(2, 4.5, 40)  # De 100 a ~31,000 seguidores
    
    # Modelos de potencia con diferentes exponentes
    likes = 0.5 * followers**1.2 + np.random.normal(0, followers**1.2 * 0.1, len(followers))
    shares = 0.08 * followers**1.4 + np.random.normal(0, followers**1.4 * 0.08, len(followers))
    comments = 0.02 * followers**1.5 + np.random.normal(0, followers**1.5 * 0.05, len(followers))
    
    data = pd.DataFrame({
        'Followers': followers,
        'Likes': np.maximum(likes, 0),
        'Shares': np.maximum(shares, 0),
        'Comments': np.maximum(comments, 0)
    })
    
    # Análisis para cada métrica
    metrics = ['Likes', 'Shares', 'Comments']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    analyzer = RegressionAnalyzer()
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        X = data[['Followers']].values
        y = data[metric].values
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # Transformación log-log: log(y) = log(a) + b*log(x)
        X_train_log = np.log(X_train)
        X_test_log = np.log(X_test)
        y_train_log = np.log(y_train + 1)
        y_test_log = np.log(y_test + 1)
        
        # Modelo de potencia
        power_model = LinearRegression()
        power_model.fit(X_train_log, y_train_log)
        
        # Parámetros
        b = power_model.coef_[0]
        log_a = power_model.intercept_
        a = np.exp(log_a)
        
        # Predicciones
        y_pred_log = power_model.predict(X_test_log)
        y_pred = np.exp(y_pred_log) - 1
        
        # Evaluar
        analyzer.evaluate_model(y_test, y_pred, f'Modelo de Potencia - {metric}')
        
        # Visualización
        ax = axes[idx]
        ax.scatter(X_train, y_train, color=color, alpha=0.5, s=30, label='Entrenamiento')
        ax.scatter(X_test, y_test, color='red', alpha=0.7, s=30, label='Prueba')
        
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_range = a * (X_range ** b)
        ax.plot(X_range, y_range, color='black', linewidth=2.5, 
                label=f'y = {a:.2f} * x^{b:.3f}', alpha=0.8)
        
        ax.set_xlabel('Seguidores', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} vs Seguidores\nR² = {r2_score(y_test, y_pred):.3f}', 
                     fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        print(f"\nModelo de Potencia para {metric}:")
        print(f"  Ecuación: y = {a:.3f} * x^{b:.3f}")
        print(f"  R² = {r2_score(y_test, y_pred):.3f}")
    
    plt.tight_layout()
    plt.show()
    
    analyzer.print_metrics()
    
    return analyzer


# =============================================================================
# COMPARACIÓN FINAL DE TODOS LOS MODELOS
# =============================================================================

def compare_all_models():
    """Genera una comparación visual de todos los modelos"""
    
    print("\n" + "="*60)
    print("COMPARACIÓN GENERAL DE MODELOS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Resumen de cuándo usar cada modelo
    summary_text = """
GUÍA DE SELECCIÓN DE MODELOS:

1. REGRESIÓN POLINOMIAL
   • Uso: Relaciones no lineales con puntos de inflexión
   • Ejemplo: Rendimiento vs esfuerzo (ley de rendimientos decrecientes)
   • Ventaja: Captura curvas y cambios de dirección
   
2. REGRESIÓN EXPONENCIAL
   • Uso: Crecimiento o decaimiento acelerado
   • Ejemplo: Crecimiento viral, decaimiento radiactivo
   • Ventaja: Modeliza crecimiento compuesto
   
3. REGRESIÓN DE POTENCIA
   • Uso: Relaciones escalables (ley de potencia)
   • Ejemplo: Engagement en redes sociales
   • Ventaja: Captura efectos de escala
   
4. REGRESIÓN LINEAL
   • Uso: Relaciones proporcionales simples
   • Ejemplo: Precio vs cantidad, distancia vs tiempo
   • Ventaja: Simple, interpretable, rápida
    """
    
    axes[0, 0].text(0.1, 0.5, summary_text, fontsize=10, 
                    verticalalignment='center', family='monospace')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Guía de Modelos de Regresión', 
                         fontsize=14, fontweight='bold', pad=20)
    
    # Ejemplos visuales simplificados
    x = np.linspace(0, 10, 100)
    
    # Polinomial
    axes[0, 1].plot(x, 2 + 3*x - 0.3*x**2, 'b-', linewidth=2)
    axes[0, 1].set_title('Patrón Polinomial', fontweight='bold')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Exponencial
    axes[1, 0].plot(x, 2 * np.exp(0.3*x), 'r-', linewidth=2)
    axes[1, 0].set_title('Patrón Exponencial', fontweight='bold')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Potencia
    x_power = np.linspace(0.1, 10, 100)
    axes[1, 1].plot(x_power, 2 * x_power**1.5, 'g-', linewidth=2)
    axes[1, 1].set_title('Patrón de Potencia', fontweight='bold')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ANÁLISIS AVANZADO DE MODELOS DE REGRESIÓN")
    print("="*60)
    
    # Ejecutar análisis
    poly_analyzer = polynomial_regression_analysis()
    exp_analyzer = exponential_regression_analysis()
    power_analyzer = power_regression_analysis()
    
    # Comparación final
    compare_all_models()
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
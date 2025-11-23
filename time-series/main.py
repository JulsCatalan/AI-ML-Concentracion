import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


class TimeSeriesAnalyzer:
    """Clase para análisis completo de series temporales"""
    
    def __init__(self, data, date_col=None, value_col=None):
        if isinstance(data, pd.DataFrame):
            if date_col and value_col:
                self.df = data.set_index(date_col)[[value_col]]
                self.series = self.df[value_col]
            else:
                self.series = data.iloc[:, 0]
        else:
            self.series = data
        
        self.name = self.series.name or 'Value'
    
    def plot_series(self, title=None):
        """Gráfico de la serie temporal"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.series, color='steelblue', linewidth=2)
        plt.title(title or f'{self.name} - Time Series', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=11)
        plt.ylabel(self.name, fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_by_year(self, title=None):
        """Gráfico separado por años"""
        years = self.series.index.year.unique()
        
        plt.figure(figsize=(12, 6))
        for year in years:
            year_data = self.series[self.series.index.year == year]
            plt.plot(year_data.index.month, year_data.values, label=year, alpha=0.7)
        
        plt.title(title or f'{self.name} by Year', fontsize=14, fontweight='bold')
        plt.xlabel('Month', fontsize=11)
        plt.ylabel(self.name, fontsize=11)
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_seasonal_subseries(self, title=None):
        """Gráfico de subseries estacionales"""
        df_temp = pd.DataFrame({
            'value': self.series.values,
            'year': self.series.index.year,
            'month': self.series.index.month
        })
        
        pivot = df_temp.pivot_table(values='value', index='month', columns='year', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot.plot(ax=ax, legend=False, alpha=0.7)
        ax.set_title(title or f'{self.name} - Seasonal Subseries', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=11)
        ax.set_ylabel(self.name, fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_distribution(self, title=None):
        """Histograma y gráfico de densidad"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histograma
        axes[0].hist(self.series.dropna(), bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].set_title('Histogram', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(self.name, fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Densidad
        self.series.dropna().plot(kind='kde', ax=axes[1], color='steelblue', linewidth=2)
        axes[1].set_title('Density Plot', fontsize=12, fontweight='bold')
        axes[1].set_xlabel(self.name, fontsize=11)
        axes[1].set_ylabel('Density', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def adf_test(self):
        """Test de Dickey-Fuller Aumentado para estacionariedad"""
        result = adfuller(self.series.dropna(), autolag='AIC')
        return {
            'test_stat': result[0],
            'pvalue': result[1],
            'lags': result[2],
            'nobs': result[3],
            'crit_values': result[4]
        }
    
    def kpss_test(self, regression='c'):
        """Test KPSS para estacionariedad"""
        try:
            stat, pval, lags, crit = kpss(self.series.dropna(), regression=regression, nlags='auto')
            return {
                'test_stat': stat,
                'pvalue': pval,
                'lags': lags,
                'crit_values': crit
            }
        except:
            return None
    
    def stationarity_tests(self):
        """Ejecutar tests de estacionariedad"""
        print(f"\n{'='*60}")
        print(f"STATIONARITY TESTS: {self.name}")
        print(f"{'='*60}")
        
        # ADF Test
        adf = self.adf_test()
        print(f"\nAugmented Dickey-Fuller Test:")
        print(f"  Test Statistic: {adf['test_stat']:.4f}")
        print(f"  P-value: {adf['pvalue']:.4f}")
        print(f"  Critical Values:")
        for key, value in adf['crit_values'].items():
            print(f"    {key}: {value:.4f}")
        
        if adf['pvalue'] < 0.05:
            print(f"  → Reject H0: Series is STATIONARY (ADF)")
        else:
            print(f"  → Fail to reject H0: Series is NON-STATIONARY (ADF)")
        
        # KPSS Test
        kpss_result = self.kpss_test()
        if kpss_result:
            print(f"\nKPSS Test:")
            print(f"  Test Statistic: {kpss_result['test_stat']:.4f}")
            print(f"  P-value: {kpss_result['pvalue']:.4f}")
            
            if kpss_result['pvalue'] < 0.05:
                print(f"  → Reject H0: Series is NON-STATIONARY (KPSS)")
            else:
                print(f"  → Fail to reject H0: Series is STATIONARY (KPSS)")
    
    def plot_rolling_stats(self, window=12, title=None):
        """Gráfico de media y varianza móviles"""
        rolling_mean = self.series.rolling(window=window).mean()
        rolling_std = self.series.rolling(window=window).std()
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Media móvil
        axes[0].plot(self.series, label='Original', alpha=0.7, color='steelblue')
        axes[0].plot(rolling_mean, label=f'Rolling Mean (window={window})', 
                    color='red', linewidth=2)
        axes[0].set_title(title or f'{self.name} - Rolling Mean', fontsize=12, fontweight='bold')
        axes[0].set_ylabel(self.name, fontsize=11)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Desviación estándar móvil
        axes[1].plot(rolling_std, label=f'Rolling Std (window={window})', 
                    color='orange', linewidth=2)
        axes[1].set_title(f'{self.name} - Rolling Standard Deviation', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time', fontsize=11)
        axes[1].set_ylabel('Std Dev', fontsize=11)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def decompose(self, model='additive', period=12):
        """Descomposición de serie temporal"""
        decomposition = seasonal_decompose(self.series.dropna(), model=model, period=period)
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Serie original
        decomposition.observed.plot(ax=axes[0], color='steelblue')
        axes[0].set_title('Original Series', fontsize=12, fontweight='bold')
        axes[0].set_ylabel(self.name, fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Tendencia
        decomposition.trend.plot(ax=axes[1], color='red')
        axes[1].set_title('Trend', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Trend', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        # Estacionalidad
        decomposition.seasonal.plot(ax=axes[2], color='green')
        axes[2].set_title('Seasonal', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Seasonal', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        # Residuos
        decomposition.resid.plot(ax=axes[3], color='purple')
        axes[3].set_title('Residuals', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('Time', fontsize=11)
        axes[3].set_ylabel('Residuals', fontsize=11)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return decomposition
    
    def plot_acf_pacf(self, lags=40):
        """Gráficos de autocorrelación y autocorrelación parcial"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ACF
        plot_acf(self.series.dropna(), lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # PACF
        plot_pacf(self.series.dropna(), lags=lags, ax=axes[1])
        axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def fit_ar(self, lags=1):
        """Ajustar modelo AutoRegresivo"""
        model = AutoReg(self.series.dropna(), lags=lags)
        fitted = model.fit()
        
        print(f"\n{'='*60}")
        print(f"AR({lags}) MODEL")
        print(f"{'='*60}")
        print(fitted.summary())
        
        return fitted
    
    def fit_ma(self, q=1):
        """Ajustar modelo de Media Móvil"""
        model = ARIMA(self.series.dropna(), order=(0, 0, q))
        fitted = model.fit()
        
        print(f"\n{'='*60}")
        print(f"MA({q}) MODEL")
        print(f"{'='*60}")
        print(fitted.summary())
        
        return fitted
    
    def fit_arima(self, order=(1, 1, 1)):
        """Ajustar modelo ARIMA"""
        model = ARIMA(self.series.dropna(), order=order)
        fitted = model.fit()
        
        print(f"\n{'='*60}")
        print(f"ARIMA{order} MODEL")
        print(f"{'='*60}")
        print(fitted.summary())
        
        # Gráfico de valores ajustados
        plt.figure(figsize=(12, 6))
        plt.plot(self.series, label='Original', color='steelblue', linewidth=2)
        plt.plot(fitted.fittedvalues, label='Fitted', color='red', linewidth=2, alpha=0.7)
        plt.title(f'ARIMA{order} - Fitted Values', fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=11)
        plt.ylabel(self.name, fontsize=11)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return fitted
    
    def complete_analysis(self):
        """Análisis completo de la serie temporal"""
        print(f"\n{'='*60}")
        print(f"COMPLETE TIME SERIES ANALYSIS: {self.name}")
        print(f"{'='*60}")
        
        self.plot_series()
        self.plot_distribution()
        self.stationarity_tests()
        self.plot_rolling_stats()
        self.decompose()
        self.plot_acf_pacf()


def load_airline_passengers():
    """Cargar dataset de pasajeros de aerolínea"""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
    df.rename(columns={'Passengers': 'Passengers'}, inplace=True)
    return df


def generate_synthetic_series(n=240):
    """Generar series temporales sintéticas"""
    t = np.arange(n)
    
    # Serie estacionaria (ruido blanco)
    white_noise = pd.Series(np.random.normal(0, 1, n), name='White Noise')
    
    # Serie con tendencia
    trend = 0.02 * t
    trended = pd.Series(trend + np.random.normal(0, 1, n), name='Trended')
    
    # Serie con tendencia y estacionalidad
    seasonal = 2 * np.sin(2 * np.pi * t / 12)
    seasonal_trend = pd.Series(trend + seasonal + np.random.normal(0, 1, n), 
                              name='Seasonal + Trend')
    
    return white_noise, trended, seasonal_trend


def analyze_airline_passengers():
    """Análisis del dataset de pasajeros de aerolínea"""
    print("\n" + "="*60)
    print("AIRLINE PASSENGERS ANALYSIS")
    print("="*60)
    
    df = load_airline_passengers()
    analyzer = TimeSeriesAnalyzer(df['Passengers'])
    
    analyzer.plot_series('Airline Passengers - Monthly Count (1949-1960)')
    analyzer.plot_by_year('Airline Passengers by Year')
    analyzer.plot_seasonal_subseries('Airline Passengers - Seasonal Pattern')
    analyzer.plot_distribution('Airline Passengers - Distribution')


def analyze_synthetic_series():
    """Análisis de series sintéticas"""
    print("\n" + "="*60)
    print("SYNTHETIC TIME SERIES ANALYSIS")
    print("="*60)
    
    white_noise, trended, seasonal_trend = generate_synthetic_series()
    
    # Ruido blanco
    print("\n--- WHITE NOISE (STATIONARY) ---")
    wn_analyzer = TimeSeriesAnalyzer(white_noise)
    wn_analyzer.plot_series('White Noise')
    wn_analyzer.stationarity_tests()
    wn_analyzer.plot_rolling_stats(window=24, title='White Noise')
    
    # Serie con tendencia
    print("\n--- TRENDED SERIES (NON-STATIONARY) ---")
    trend_analyzer = TimeSeriesAnalyzer(trended)
    trend_analyzer.plot_series('Trended Series')
    trend_analyzer.stationarity_tests()
    trend_analyzer.plot_rolling_stats(window=24, title='Trended Series')
    
    # Serie con estacionalidad y tendencia
    print("\n--- SEASONAL + TREND (NON-STATIONARY) ---")
    st_analyzer = TimeSeriesAnalyzer(seasonal_trend)
    st_analyzer.plot_series('Seasonal + Trend Series')
    st_analyzer.stationarity_tests()
    st_analyzer.plot_rolling_stats(window=24, title='Seasonal + Trend')


def arima_modeling():
    """Modelado ARIMA del dataset de pasajeros"""
    print("\n" + "="*60)
    print("ARIMA MODELING - AIRLINE PASSENGERS")
    print("="*60)
    
    df = load_airline_passengers()
    analyzer = TimeSeriesAnalyzer(df['Passengers'])
    
    # Descomposición
    print("\n--- DECOMPOSITION ---")
    analyzer.decompose(model='additive', period=12)
    
    # ACF y PACF
    print("\n--- ACF and PACF ---")
    analyzer.plot_acf_pacf(lags=40)
    
    # Ajustar modelos
    print("\n--- FITTING MODELS ---")
    
    # AR(1)
    ar_model = analyzer.fit_ar(lags=1)
    
    # MA(1)
    ma_model = analyzer.fit_ma(q=1)
    
    # ARIMA(1,1,1)
    arima_model = analyzer.fit_arima(order=(1, 1, 1))
    
    return arima_model


def main():
    """Función principal"""
    print("\n" + "="*60)
    print("TIME SERIES ANALYSIS - COMPLETE WORKFLOW")
    print("="*60)
    
    # Análisis de pasajeros de aerolínea
    analyze_airline_passengers()
    
    # Análisis de series sintéticas
    analyze_synthetic_series()
    
    # Modelado ARIMA
    arima_modeling()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
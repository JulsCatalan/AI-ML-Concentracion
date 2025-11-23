import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping

# Configuración
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
np.random.seed(42)
tf.random.set_seed(42)


class TimeSeriesForecaster:
    """Clase para forecasting de series temporales con deep learning"""
    
    def __init__(self, data, window=29, lag=1, test_size=0.2):
        self.data = data
        self.window = window
        self.lag = lag
        self.test_size = test_size
        
        # Inicializar atributos
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler_x = None
        self.scaler_y = None
        
        # Modelos
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        
    def make_supervised_data(self, series):
        """Convertir serie temporal a problema supervisado"""
        X, y = [], []
        for i in range(self.window, len(series) - self.lag):
            X.append(series[i - self.window:i + 1])
            y.append(series[i + self.lag])
        return np.array(X), np.array(y)
    
    def prepare_data(self):
        """Preparar datos: supervisado, split y escalado"""
        # Crear datos supervisados
        X, y = self.make_supervised_data(self.data)
        
        print(f"\n{'='*60}")
        print("DATA PREPARATION")
        print(f"{'='*60}")
        print(f"Original series length: {len(self.data)}")
        print(f"Window size: {self.window}")
        print(f"Lag: {self.lag}")
        print(f"Supervised X shape: {X.shape}")
        print(f"Supervised y shape: {y.shape}")
        
        # Split temporal
        cut = int(len(X) * (1 - self.test_size))
        self.X_train, self.X_test = X[:cut], X[cut:]
        self.y_train, self.y_test = y[:cut], y[cut:]
        
        print(f"\nTrain samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        
        # Escalado
        self.scaler_x = StandardScaler().fit(self.X_train)
        self.scaler_y = StandardScaler().fit(self.y_train.reshape(-1, 1))
        
        return self
    
    def get_scaled_data(self, reshape_3d=False, subseq=None):
        """Obtener datos escalados, opcionalmente en 3D o 4D"""
        X_train_s = self.scaler_x.transform(self.X_train)
        X_test_s = self.scaler_x.transform(self.X_test)
        y_train_s = self.scaler_y.transform(self.y_train.reshape(-1, 1)).ravel()
        y_test_s = self.scaler_y.transform(self.y_test.reshape(-1, 1)).ravel()
        
        if not reshape_3d:
            return X_train_s, X_test_s, y_train_s, y_test_s
        
        # Reshape para CNN/LSTM (3D)
        X_train_3d = X_train_s.reshape((len(X_train_s), self.window + 1, 1))
        X_test_3d = X_test_s.reshape((len(X_test_s), self.window + 1, 1))
        
        if subseq is None:
            return X_train_3d, X_test_3d, y_train_s, y_test_s
        
        # Reshape para CNN-LSTM (4D)
        steps = (self.window + 1) // subseq
        X_train_4d = X_train_3d.reshape((len(X_train_3d), subseq, steps, 1))
        X_test_4d = X_test_3d.reshape((len(X_test_3d), subseq, steps, 1))
        
        return X_train_4d, X_test_4d, y_train_s, y_test_s
    
    def build_cnn(self, filters=32, kernel_size=3, pool_size=2, dense_units=32):
        """Construir modelo CNN 1D"""
        model = Sequential([
            Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', 
                   input_shape=(self.window + 1, 1)),
            MaxPooling1D(pool_size=pool_size),
            Flatten(),
            Dense(dense_units, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_lstm(self, units=32):
        """Construir modelo LSTM"""
        model = Sequential([
            LSTM(units, activation='tanh', input_shape=(self.window + 1, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def build_cnn_lstm(self, subseq=5, filters=32, kernel_size=3, lstm_units=32):
        """Construir modelo híbrido CNN-LSTM"""
        steps = (self.window + 1) // subseq
        
        model = Sequential([
            TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'),
                          input_shape=(subseq, steps, 1)),
            TimeDistributed(MaxPooling1D(2)),
            TimeDistributed(Flatten()),
            LSTM(lstm_units, activation='tanh'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def train_model(self, model_name, model, X_train, y_train, epochs=50, batch_size=32, 
                   validation_split=0.1, verbose=1):
        """Entrenar modelo con early stopping"""
        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        
        print(f"\n{'='*60}")
        print(f"TRAINING {model_name}")
        print(f"{'='*60}")
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        self.models[model_name] = model
        return history
    
    def predict_and_evaluate(self, model_name, model, X_test):
        """Hacer predicciones y calcular métricas"""
        # Predicción en escala normalizada
        y_pred_s = model.predict(X_test, verbose=0).ravel()
        
        # Invertir escala
        y_pred = self.scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
        y_true = self.y_test
        
        # Calcular métricas
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Guardar resultados
        self.predictions[model_name] = y_pred
        self.metrics[model_name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
        print(f"\n{model_name} Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        return y_pred
    
    def plot_predictions(self, model_name=None, last_n=150):
        """Graficar predicciones vs valores reales"""
        if model_name is None:
            model_names = list(self.predictions.keys())
        else:
            model_names = [model_name]
        
        n_models = len(model_names)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        y_true = self.y_test
        k = min(last_n, len(y_true))
        
        for ax, name in zip(axes, model_names):
            y_pred = self.predictions[name]
            
            ax.plot(y_true[-k:], label='Actual', linewidth=2, color='steelblue')
            ax.plot(y_pred[-k:], label='Predicted', linewidth=2, 
                   color='red', alpha=0.7, linestyle='--')
            
            rmse = self.metrics[name]['RMSE']
            ax.set_title(f'{name} - Predictions (RMSE: {rmse:.2f})', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Step', fontsize=11)
            ax.set_ylabel('Value', fontsize=11)
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self):
        """Comparar métricas de todos los modelos"""
        if not self.metrics:
            print("No models trained yet!")
            return
        
        models = list(self.metrics.keys())
        rmse_values = [self.metrics[m]['RMSE'] for m in models]
        mae_values = [self.metrics[m]['MAE'] for m in models]
        r2_values = [self.metrics[m]['R²'] for m in models]
        
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # RMSE
        axes[0].bar(models, rmse_values, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_title('RMSE Comparison', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('RMSE', fontsize=11)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # MAE
        axes[1].bar(models, mae_values, color='green', alpha=0.7, edgecolor='black')
        axes[1].set_title('MAE Comparison', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('MAE', fontsize=11)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # R²
        axes[2].bar(models, r2_values, color='orange', alpha=0.7, edgecolor='black')
        axes[2].set_title('R² Comparison', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('R²', fontsize=11)
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        # Tabla resumen
        print(f"\n{'='*60}")
        print("METRICS SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
        print(f"{'-'*60}")
        for model in models:
            metrics = self.metrics[model]
            print(f"{model:<15} {metrics['RMSE']:>10.4f} {metrics['MAE']:>10.4f} {metrics['R²']:>10.4f}")
    
    def plot_training_history(self, history, model_name):
        """Graficar pérdida durante el entrenamiento"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in history.history:
            axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_title(f'{model_name} - Training Loss', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss (MSE)', fontsize=11)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE
        if 'mae' in history.history:
            axes[1].plot(history.history['mae'], label='Train MAE', linewidth=2)
            if 'val_mae' in history.history:
                axes[1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
            axes[1].set_title(f'{model_name} - Training MAE', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Epoch', fontsize=11)
            axes[1].set_ylabel('MAE', fontsize=11)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def load_demand_data(filepath='train.csv', store=1, item=1):
    """Cargar dataset de demanda y filtrar"""
    print(f"\n{'='*60}")
    print("LOADING DEMAND FORECASTING DATA")
    print(f"{'='*60}")
    
    df = pd.read_csv(filepath, parse_dates=['date'])
    
    # Filtrar
    df = df.query(f"store == {store} and item == {item}").copy()
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Store: {store}, Item: {item}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total samples: {len(df)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    return df


def train_all_models(forecaster, epochs=50, batch_size=32):
    """Entrenar CNN, LSTM y CNN-LSTM"""
    
    # CNN Model
    print("\n" + "="*60)
    print("1. CNN MODEL")
    print("="*60)
    
    X_train_3d, X_test_3d, y_train_s, y_test_s = forecaster.get_scaled_data(reshape_3d=True)
    
    cnn_model = forecaster.build_cnn(filters=32, kernel_size=3, dense_units=32)
    print("\nCNN Architecture:")
    cnn_model.summary()
    
    history_cnn = forecaster.train_model(
        'CNN', cnn_model, X_train_3d, y_train_s,
        epochs=epochs, batch_size=batch_size, verbose=0
    )
    
    forecaster.predict_and_evaluate('CNN', cnn_model, X_test_3d)
    forecaster.plot_training_history(history_cnn, 'CNN')
    
    # LSTM Model
    print("\n" + "="*60)
    print("2. LSTM MODEL")
    print("="*60)
    
    lstm_model = forecaster.build_lstm(units=32)
    print("\nLSTM Architecture:")
    lstm_model.summary()
    
    history_lstm = forecaster.train_model(
        'LSTM', lstm_model, X_train_3d, y_train_s,
        epochs=epochs, batch_size=batch_size, verbose=0
    )
    
    forecaster.predict_and_evaluate('LSTM', lstm_model, X_test_3d)
    forecaster.plot_training_history(history_lstm, 'LSTM')
    
    # CNN-LSTM Model
    print("\n" + "="*60)
    print("3. CNN-LSTM HYBRID MODEL")
    print("="*60)
    
    subseq = 5
    if (forecaster.window + 1) % subseq != 0:
        print(f"Warning: window+1={forecaster.window+1} not divisible by {subseq}")
        subseq = 6
    
    X_train_4d, X_test_4d, y_train_s, y_test_s = forecaster.get_scaled_data(
        reshape_3d=True, subseq=subseq
    )
    
    cnn_lstm_model = forecaster.build_cnn_lstm(subseq=subseq, filters=32, lstm_units=32)
    print("\nCNN-LSTM Architecture:")
    cnn_lstm_model.summary()
    
    history_cnn_lstm = forecaster.train_model(
        'CNN-LSTM', cnn_lstm_model, X_train_4d, y_train_s,
        epochs=epochs, batch_size=batch_size, verbose=0
    )
    
    forecaster.predict_and_evaluate('CNN-LSTM', cnn_lstm_model, X_test_4d)
    forecaster.plot_training_history(history_cnn_lstm, 'CNN-LSTM')


def main():
    """Función principal"""
    
    print("\n" + "="*60)
    print("DEMAND FORECASTING WITH DEEP LEARNING")
    print("="*60)
    
    # Cargar datos
    df = load_demand_data('train.csv', store=1, item=1)
    
    # Crear forecaster
    forecaster = TimeSeriesForecaster(
        data=df['sales'].values,
        window=29,
        lag=1,
        test_size=0.2
    )
    
    # Preparar datos
    forecaster.prepare_data()
    
    # Entrenar todos los modelos
    train_all_models(forecaster, epochs=50, batch_size=32)
    
    # Comparar resultados
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    forecaster.plot_metrics_comparison()
    forecaster.plot_predictions(last_n=150)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from IPython.display import Audio, display

def get_sample_rate(frequency, duration, sr=5000):
    data = librosa.tone(frequency, sr=sr, duration=duration)
    nyquist_freq = sr / 2
    return data, sr, nyquist_freq


# 1️⃣ Cargar sonido de trompeta de ejemplo
archivo_trompeta = librosa.example('trumpet')
tono_original, sr = librosa.load(archivo_trompeta) 
t = np.arange(len(tono_original)) / sr

# 2️⃣ Reducir la frecuencia de muestreo a la mitad (downsampling)
tono_reducido = tono_original[::2]
t_reducido = t[::2]
sr_reducido = sr // 2

# 3️⃣ Reconstruir la señal usando interpolación lineal
interpolador = interp1d(t_reducido, tono_reducido, kind='linear')
tono_reconstruido = interpolador(t)

# 4️⃣ Graficar las señales
plt.figure(figsize=(12, 6))
plt.plot(t, tono_original, label='Original', alpha=0.7)
plt.plot(t_reducido, tono_reducido, 'o', label='Reducido', markersize=3)
plt.plot(t, tono_reconstruido, '--', label='Reconstruido', alpha=0.7)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Señal de trompeta: Original, Reducida y Reconstruida')
plt.legend()
plt.show()

# # 5️⃣ Reproducir audio
# print("🔊 Sonido original:")
# display(Audio(tono_original, rate=sr))

# print("🔊 Sonido reducido (frecuencia de muestreo a la mitad):")
# display(Audio(tono_reducido, rate=sr_reducido))

# print("🔊 Sonido reconstruido:")
# display(Audio(tono_reconstruido, rate=sr))

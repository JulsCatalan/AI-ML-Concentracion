# 1. Importar librerías
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio

# 2. Cargar archivos de audio
data_blinding, sr_blinding = librosa.load("./sounds/blinding_lights.wav")
data_takeonme, sr_takeonme = librosa.load("./sounds/take_on_me.wav")

# 3. Reproducir audios
print("Blinding Lights")
Audio(data_blinding, rate=sr_blinding)

print("Take On Me")
Audio(data_takeonme, rate=sr_takeonme)

# 4. Visualizar ondas completas
plt.figure(figsize=(12, 4))
librosa.display.waveshow(data_blinding, sr=sr_blinding)
plt.title("Blinding Lights")
plt.show()

plt.figure(figsize=(12, 4))
librosa.display.waveshow(data_blinding, sr=sr_takeonme)
plt.title("Take On Me")
plt.show()

plt.figure(figsize=(12, 4))
plt.title("Blinding Lights (fragmento 1000:20000)")
librosa.display.waveshow(data_blinding[1000:20000], sr=sr_blinding)
plt.show()


plt.figure(figsize=(12, 4))
plt.title("Take On Me (fragmento 1000:20000)")
librosa.display.waveshow(data_takeonme[1000:200000], sr=sr_takeonme)
plt.show()

# 5. Reducir duración (fragmento de 60s)
data_blinding_short, sr_blinding_short = librosa.load("./sounds/blinding_lights.wav", duration=60)
data_takeonme_short, sr_takeonme_short = librosa.load("./sounds/take_on_me.wav", duration=60)

plt.figure(figsize=(12, 4))
librosa.display.waveshow(data_blinding_short, sr=sr_blinding_short)
plt.title("Blinding Lights (60s)")
plt.show()

plt.figure(figsize=(12, 4))
librosa.display.waveshow(data_takeonme_short, sr=sr_takeonme_short)
plt.title("Take On Me (60s)")
plt.show()

# 6. Calcular amplitudes máximas y mínimas
print(f"Blinding Lights - Max amplitude: {np.max(data_blinding_short)}, Min amplitude: {np.min(data_blinding_short)}")
print(f"Take On Me - Max amplitude: {np.max(data_takeonme_short)}, Min amplitude: {np.min(data_takeonme_short)}")

# 7. Comparar dos sonidos en la misma gráfica
plt.figure(figsize=(12, 4))
librosa.display.waveshow(data_blinding_short, sr=sr_blinding_short, color='r', alpha=0.5, label="Blinding Lights")
librosa.display.waveshow(data_takeonme_short, sr=sr_takeonme_short, color='b', alpha=0.5, label="Take On Me")
plt.title("Comparación Blinding Lights vs Take On Me")
plt.show()

# 8. Reproducir mezcla de sonidos
Audio(data_blinding_short + data_takeonme_short, rate=sr_blinding_short)

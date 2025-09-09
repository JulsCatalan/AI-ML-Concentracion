import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def calculate_zero_crossing_rate(signal, frame_length, hop_length):

    n_frames = 1 + (len(signal) - frame_length) // hop_length
    zcr = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frame = signal[start:end]
        zero_crossings = np.sum(np.abs(np.diff(np.sign(frame))) > 0)
        zcr[i] = zero_crossings / frame_length
    
    return zcr

def plot_audio_and_zcr(signal, sample_rate, zcr, hop_length):
    time_signal = np.arange(len(signal)) / sample_rate
    time_zcr = np.arange(len(zcr)) * hop_length / sample_rate
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(time_signal, signal, 'b', linewidth=0.5)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Original Audio Signal')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_zcr, zcr, 'r', linewidth=0.5)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Zero Crossing Rate')
    ax2.set_title('Zero Crossing Rate Over Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

filename = 'audio.wav'
sample_rate, audio_signal = wavfile.read(filename)

if audio_signal.dtype == np.int16:
    audio_signal = audio_signal / 32768.0
elif audio_signal.dtype == np.int32:
    audio_signal = audio_signal / 2147483648.0

frame_length = 2048
hop_length = 512

print(f"Sample Rate: {sample_rate} Hz")
print(f"Duration: {len(audio_signal)/sample_rate:.2f} seconds")
print(f"Total Samples: {len(audio_signal)}")

zcr = calculate_zero_crossing_rate(audio_signal, frame_length, hop_length)

print(f"\nZCR Statistics:")
print(f"  Mean ZCR: {np.mean(zcr):.4f}")
print(f"  Max ZCR: {np.max(zcr):.4f}")
print(f"  Min ZCR: {np.min(zcr):.4f}")
print(f"  Std Dev: {np.std(zcr):.4f}")

fig = plot_audio_and_zcr(audio_signal, sample_rate, zcr, hop_length)
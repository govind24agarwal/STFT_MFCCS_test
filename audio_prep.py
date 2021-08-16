import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

FIG_SIZE = (15, 10)

file = "birds.wav"

# load audio file with librosa
signal, sr = librosa.load(file, sr=22050)

# display waveform
plt.figure(figsize=FIG_SIZE)
librosa.display.waveplot(signal, sr, alpha=0.4)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")

# show plot
plt.show()

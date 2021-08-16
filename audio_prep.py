import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

FIG_SIZE = (15, 10)

file = "birds.wav"

# load audio file with librosa
signal, sr = librosa.load(file, sr=22050)


"""
===========================================
Waveform
===========================================
"""
# display waveform
plt.figure(figsize=FIG_SIZE)
librosa.display.waveplot(signal, sr, alpha=0.4)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Waveform")


"""
===========================================
FFT (power spectrum)
===========================================
"""
# perform Fourier Transform
fft = np.fft.fft(signal)
# Get magnitude from complex variables
spectrum = np.abs(fft)
# Createfrequency  variable
f = np.linspace(0, sr, len(spectrum))
# Getuseful half of frequency variable and frequency spectrum
left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]
# Plotting spectrum
plt.figure(figsize=FIG_SIZE)
plt.plot(left_f, left_spectrum, alpha=0.4)
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Power spectrum")

# show plot
plt.show()

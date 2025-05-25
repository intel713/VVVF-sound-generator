import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

sample_rate = 192000
duration = 20
f = 100
svm = False

def signal_freq(t):
    x = t / duration
    return f * x

def signal_amp(t):
    x = f * (t / duration)
    a1 = 1.374 * (x / 65)
    a2 = x / 67
    a3 = np.full_like(x, 1.1)
    
    return np.where(x < 60, a1,
           np.where(x < 67, a2, a3))

def carrier_freq(t):
    x = f * (t / duration)
    c1 = np.full_like(x, 1050)
    c2 = np.full_like(x, 1050 - 350 * ((x - 23) / 26))
    c3 = np.full_like(x, 700 + 1100 * ((x - 49) / 11))
    c4 = 3 * f_signal
    
    return np.where(x < 23, c1,
           np.where(x < 49, c2,
           np.where(x < 60, c3, c4)))


t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

f_signal = signal_freq(t)
a_signal = signal_amp(t)
f_carrier = carrier_freq(t)

phase_U = 2 * np.pi * np.cumsum(f_signal) / sample_rate
phase_V = phase_U - 2 * np.pi / 3

U = a_signal * (np.sin(phase_U) + (svm / (4 * np.pi)) * np.arcsin(np.sin(3 * phase_U)))
V = a_signal * (np.sin(phase_V) + (svm / (4 * np.pi)) * np.arcsin(np.sin(3 * phase_U)))

carrier_phase = 2 * np.pi * np.cumsum(f_carrier) / sample_rate
carrier = (-2 / np.pi) * np.arcsin(np.sin(carrier_phase))

pwm_U = (U > carrier).astype(np.float32) * 2 - 1
pwm_V = (V > carrier).astype(np.float32) * 2 - 1

line_UV = pwm_U - pwm_V
line_UV /= np.max(np.abs(line_UV))\

write("VVVF_sound.wav", sample_rate, (line_UV * 32767).astype(np.int16))

sd.play(line_UV, sample_rate)
sd.wait()

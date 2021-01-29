# My imports
from amt_models.features.cqt import CQT
from amt_models.features.vqt import VQT
from amt_models.features.melspec import MelSpec

# Regular imports
from librosa.display import specshow

import matplotlib.pyplot as plt
import numpy as np
import librosa

y, sr = librosa.load(librosa.util.example_audio_file())

times = np.arange(len(y)) / sr

#plt.rcParams.update({'font.size': 15})
plt.plot(times, y, color='k')
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

cqt_ = CQT(sample_rate=sr,
           hop_length=512,
           n_bins=(8 * 24),
           bins_per_octave=24,
           decibels=True)
cqt = cqt_.process_audio(y)[0]
cqt = 80 * (cqt - 1)

plt.figure()
specshow(cqt,
         sr=sr,
         hop_length=512,
         bins_per_octave=24,
         x_axis='time',
         y_axis='cqt_hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q')
plt.show()

vqt_ = VQT(sample_rate=sr,
           hop_length=512,
           n_bins=(8 * 24),
           bins_per_octave=24,
           decibels=True)
vqt = vqt_.process_audio(y)[0]
vqt = 80 * (vqt - 1)

plt.figure()
specshow(vqt,
         sr=sr,
         hop_length=512,
         bins_per_octave=24,
         x_axis='time',
         y_axis='cqt_hz')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Variable-Q (gamma = {vqt_.gamma:.2f})')
plt.show()

mel_ = MelSpec(sample_rate=sr,
               hop_length=512,
               n_mels=229)
mel = mel_.process_audio(y)[0]
mel = 80 * (mel - 1)

plt.figure()
specshow(mel,
         sr=sr,
         hop_length=512,
         x_axis='time',
         y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Mel Spectrogram')
plt.show()

spec = np.abs(librosa.stft(y, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=np.max)

plt.figure()
specshow(spec,
         sr=sr,
         hop_length=512,
         x_axis='time',
         y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title(f'Power Spectrogram')
plt.show()

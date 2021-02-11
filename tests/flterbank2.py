# My imports
#from amt_models.tools.constants import *

# Regular imports
import matplotlib.pyplot as plt
import numpy as np
import librosa
import torch
import os

device = 1
device = device = torch.device(f'cuda:{device}'if torch.cuda.is_available() else 'cpu')

gen_expr_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'amt_models', 'generated', 'experiments')
model_path = os.path.join(gen_expr_dir, 'OnsetsFrames_GuitarSet_LHVQT_vd_new_params', 'models', 'fold-0', 'model-2500.pt')
model = torch.load(model_path, map_location=device)

# Extract the filterbank from the model
fb_module = torch.nn.Sequential(*list(model.feat_ext.fb.tfs.children()))[0]

n_fft = 2 ** 12

# Choose a filter
filt_num = 79
# Index a complex filter from the filterbank
filt = fb_module.get_comp_weights()[filt_num]

# Do the FFT and take the magnitude of the response
freq_resp = np.abs(np.fft.fft(filt, n=n_fft))

# Convert the frequency respone from amplitude to decibel
#freq_resp = librosa.amplitude_to_db(freq_resp, ref=np.max)

# Get the frequencies corresponding to the FFT indices
freqs = np.fft.fftfreq(n_fft, (1 / fb_module.fs))

# Re-order the FFT and frequencies so they go from most negative to most positive
freq_resp = np.roll(freq_resp, freq_resp.size // 2)
freqs = np.roll(freqs, freqs.size // 2)

# Plot the FFT
plt.plot(freqs, freq_resp)
#plt.ylim([0, nyquist])
plt.title(f'Frequency Response for Filter {filt_num}')
#plt.ylabel('Amplitude (dB)')
plt.ylabel('Amplitude')
plt.xlabel('Frequency')
plt.show()

# My imports
from amt_models.tools.constants import *

# Regular imports
import matplotlib.pyplot as plt
import librosa
import torch
import os

def visualize(model, i=None):
    vis_dir = os.path.join(GEN_VISL_DIR, 'OnsetsFrames_GuitarSet_LHVQT_4')

    # TODO - overwrites fold-wise

    if i is not None:
        vis_dir = os.path.join(vis_dir, f'checkpoint-{i}')

    model.feat_ext.fb.plot_time_weights(vis_dir)
    model.feat_ext.fb.plot_freq_weights(vis_dir)

model_path = os.path.join(GEN_EXPR_DIR, 'OnsetsFrames_GuitarSet_LHVQT_4', 'models', 'fold-5', 'model-3000.pt')
model = torch.load(model_path)

visualize(model, 3000)

"""
filterbank = model.feat_ext.fb

audio, sr = librosa.load(librosa.util.example_audio_file())

audio = torch.Tensor(audio).to(model.device)
audio = audio.unsqueeze(0).unsqueeze(0)
rand_vqt = filterbank(audio).cpu().detach().numpy()[0][0]
rand_vqt = 80 * (rand_vqt - 1)

plt.figure()
plt.imshow(rand_vqt, vmin=0, vmax=1)
plt.gca().invert_yaxis()
plt.colorbar(format='%+2.0f dB')
plt.title('Random Filterbank')
plt.show()
"""

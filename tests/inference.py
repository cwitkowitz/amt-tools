# My imports
from amt_models.tools.conversion import to_multi, to_single
from amt_models.tools.instrument import GuitarProfile
from amt_models.tools.constants import *

from amt_models.pipeline.visualize import plot_pianoroll

from amt_models.features import VQT

# Regular imports
import numpy as np
import torch
import os

model_path = os.path.join(GEN_EXPR_DIR, 'OnsetsFrames_GuitarSet_VQT', 'models', 'fold-2', 'model-3000.pt')
model = torch.load(model_path)

sample_path = os.path.join(GEN_DATA_DIR, 'GuitarSet', 'gt', '02_BN1-129-Eb_comp.npz')
sample_data = dict(np.load(sample_path))

fs = sample_data['fs'].item()
audio = sample_data['audio']
tabs_ref = sample_data['pitch']

vqt_ = VQT(sample_rate=fs,
           hop_length=512,
           n_bins=(8 * 24),
           bins_per_octave=24)
vqt = vqt_.process_audio(audio)

# Initialize the default guitar profile
profile = GuitarProfile()

multipitch_ref = to_multi(tabs_ref, profile)
pianoroll_ref = to_single(multipitch_ref, profile)

vqt = torch.Tensor(vqt).to(model.device)
vqt = vqt.unsqueeze(0)
vqt = vqt.transpose(-1, -2)
results = model(vqt)
# TODO - this is pretty wonky - should make an inference function for a single sample
results = model.post_proc({'preds' : results})

tabs_est = results['pitch'][0].cpu().detach().numpy()
multipitch_est = to_multi(tabs_est, profile)
pianoroll_est = to_single(multipitch_est, profile)

ax = plot_pianoroll(pianoroll_ref)

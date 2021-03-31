# My imports
import amt_models.tools as tools

from tools.visualize import visualize_multi_pitch

from amt_models.features import MelSpec

# Regular imports
from matplotlib import rcParams

import matplotlib.pyplot as plt
import torch
import os

model_path = os.path.join(tools.DEFAULT_EXPERIMENTS_DIR,
                          'OnsetsFrames2_MAESTRO_V1_MelSpec',
                          'models', 'model-1000.pt')
model = torch.load(model_path)
model.change_device(1)
model.eval()

sample_path = os.path.join(tools.DEFAULT_FEATURES_GT_DIR,
                           'MAPS', 'ground_truth',
                           'MAPS_MUS-bor_ps6_ENSTDkCl.npz')
reference = tools.load_unpack_npz(sample_path)

frame_start, frame_end = 800, 1000

audio = reference[tools.KEY_AUDIO]

dim_in = 229
hop_length = 512
sample_rate = 16000

mel_ = MelSpec(sample_rate=sample_rate,
               hop_length=hop_length,
               n_mels=dim_in,
               htk=True)
mel = mel_.process_audio(audio)

# Initialize the default guitar profile
profile = tools.PianoProfile()

mel = torch.Tensor(mel).to(model.device)
mel = mel.unsqueeze(0)
mel = mel.transpose(-1, -2)

with torch.no_grad():
    predictions = model(mel)
    # TODO - this is pretty wonky - should make an inference function for a single sample
    predictions = model.post_proc({tools.KEY_OUTPUT : predictions})

predictions = tools.track_to_cpu(predictions)

reference = tools.slice_track(reference, frame_start, frame_end, skip=['fs', 'audio', 'notes'])
predictions = tools.slice_track(predictions, frame_start, frame_end)

figsize = rcParams['figure.figsize']
figsize = [2 * figsize[0], figsize[1]]

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=figsize)

visualize_multi_pitch(reference[tools.KEY_ONSETS], axs[0, 0])
axs[0, 0].set_title('Reference Onsets')

visualize_multi_pitch(predictions[tools.KEY_ONSETS], axs[1, 0])
axs[1, 0].set_title('Estimated Onsets')

visualize_multi_pitch(reference[tools.KEY_MULTIPITCH], axs[0, 1])
axs[0, 1].set_title('Reference Multi Pitch')

visualize_multi_pitch(predictions[tools.KEY_MULTIPITCH], axs[1, 1])
axs[1, 1].set_title('Estimated Multi Pitch')

visualize_multi_pitch(reference[tools.KEY_OFFSETS], axs[0, 2])
axs[0, 2].set_title('Reference Offsets')

visualize_multi_pitch(predictions[tools.KEY_OFFSETS], axs[1, 2])
axs[1, 2].set_title('Estimated Offsets')

fig.suptitle(f'Frames {frame_start} - {frame_end}')
fig.tight_layout()

plt.show()

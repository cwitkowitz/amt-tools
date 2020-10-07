# My imports
from features.common import *

# Regular imports
from lhvqt.lhvqt_ds import LHVQT_DS as _LHVQT
from lhvqt.lvqt_orig import LVQT

import librosa
import torch


class LHVQT(FeatureModule):
    def __init__(self, sample_rate=44100, harmonics=[0.5, 1, 2, 3, 4, 5], hop_length=512,
                 fmin=None, n_bins=84, bins_per_octave=12, gamma=0, random=False, max_p=1):

        super().__init__(sample_rate, hop_length, False)

        self.lhvqt = _LHVQT(fmin=fmin,
                            harmonics=harmonics,
                            lvqt=LVQT,
                            fs=self.sample_rate,
                            hop_length=self.hop_length,
                            n_bins=n_bins,
                            bins_per_octave=bins_per_octave,
                            gamma=gamma,
                            random=random,
                            max_p=max_p)

    def get_expected_frames(self, audio):
        # TODO - one frame more than librosa in compare.py? - related to padding for extra frame or filter lengths?
        audio = torch.from_numpy(audio)
        num_frames_all = self.lhvqt.get_expected_frames(audio)
        num_frames = int(np.mean(num_frames_all))
        assert np.sum(np.array(num_frames_all) - num_frames) == 0
        return num_frames

    def get_sample_range(self, num_frames):
        # TODO - different behavior if padding vs. not padding for extra frame
        # sample_range = np.arange(1, self.hop_length + 1) + (num_frames - 2) * self.hop_length
        sample_range = np.arange(0, self.hop_length) + (num_frames - 1) * self.hop_length
        sample_range = sample_range[sample_range > 0]
        return sample_range

    def process_audio(self, audio):
        return None

    def to_decibels(self, feats):
        return None

    def get_times(self, audio):
        num_frames = self.get_expected_frames(audio)
        frame_idcs = np.arange(num_frames + 1)
        # TODO - is there a centering factor?
        times = librosa.frames_to_time(frames=frame_idcs,
                                       sr=self.sample_rate,
                                       hop_length=self.hop_length)
        return times

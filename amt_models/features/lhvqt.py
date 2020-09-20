# My imports
from features.common import *

# Regular imports
from lhvqt.lhvqt_orig import LHVQT as _LHVQT

import librosa


class LHVQT(FeatureModule):
    def __init__(self, sample_rate=44100, harmonics=[0.5, 1, 2, 3, 4, 5], hop_length=512,
                 fmin=None, n_bins=84, bins_per_octave=12, gamma=0, scale=True, norm_length=True,
                 random=False, max_p=1):

        super().__init__(sample_rate, hop_length, False)

        self.lhvqt = _LHVQT(fs=self.sample_rate,
                            harmonics=harmonics,
                            hop_length=self.hop_length,
                            fmin=fmin,
                            n_bins=n_bins,
                            bins_per_octave=bins_per_octave,
                            gamma=gamma,
                            scale=scale,
                            norm_length=norm_length,
                            random=random,
                            max_p=max_p)

    def get_expected_frames(self, audio):
        num_frames = (audio.shape[-1] - 1) // self.hop_length + 1
        return num_frames

    def get_sample_range(self, num_frames):
        sample_range = np.arange(1, self.hop_length + 1) + (num_frames - 1) * self.hop_length
        return sample_range

    def process_audio(self, audio):
        return None

    def to_decibels(self, feats):
        return NotImplementedError

    def get_times(self, audio):
        num_frames = self.get_expected_frames(audio)
        frame_idcs = np.arange(num_frames + 1)
        # TODO - is there a centering factor?
        times = librosa.frames_to_time(frames=frame_idcs,
                                       sr=self.sample_rate,
                                       hop_length=self.hop_length)
        return times

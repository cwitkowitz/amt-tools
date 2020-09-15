# My imports
from features.common import *

# Regular imports
import numpy as np
import librosa


class MelSpec(FeatureModule):
    def __init__(self, sample_rate=16000, n_mels=229, n_fft=2048,
                 hop_length=512, htk=False, norm=None, decibels=True):
        super().__init__(sample_rate, hop_length, decibels)

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.htk = htk
        self.norm = norm

    def get_expected_frames(self, audio):
        num_frames = 1 + len(audio) // self.hop_length

        return num_frames

    def get_sample_range(self, num_frames):
        max_samples = num_frames * self.hop_length - 1
        min_samples = max(1, max_samples - self.hop_length + 1)
        sample_range = np.arange(min_samples, max_samples + 1)
        return sample_range

    def process_audio(self, audio):
        mel = librosa.feature.melspectrogram(y=audio,
                                             sr=self.sample_rate,
                                             n_mels=self.n_mels,
                                             n_fft=self.n_fft,
                                             hop_length=self.hop_length,
                                             htk=self.htk,
                                             norm=self.norm)

        mel = super().post_proc(mel)

        return mel

    def to_decibels(self, feats):
        feats = librosa.core.power_to_db(feats, ref=np.max)
        return feats

    def get_times(self, audio):
        num_frames = self.get_expected_frames(audio)
        frame_idcs = np.arange(num_frames + 1)
        times = librosa.frames_to_time(frames=frame_idcs,
                                       sr=self.sample_rate,
                                       hop_length=self.hop_length,
                                       n_fft=self.n_fft)
        return times

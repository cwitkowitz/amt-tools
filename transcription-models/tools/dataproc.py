# My imports
from .constants import *
from .utils import *

# Regular imports
from librosa.core.constantq import __early_downsample_count as early_downsample_count

import numpy as np
import librosa

# TODO - build data proc module on top which is general - i.e. whole/local cqt or filterbank learning module

class CQT:
    def __init__(self, hop_length=512, fmin=None, n_bins=84, bins_per_octave=12):
        self.fs = SAMPLE_RATE
        self.hop_length = hop_length

        self.fmin = fmin
        if self.fmin is None:
            # C1 by default
            self.fmin = librosa.note_to_hz('C1')

        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave

    # TODO - can this be abstracted from librosa?
    def get_num_octaves(self):
        # How many octaves are we dealing with?
        n_octaves = int(np.ceil(float(self.n_bins) / self.bins_per_octave))
        return n_octaves

    # TODO - can this be abstracted from librosa?
    def get_early_ds_count(self):
        n_octaves = self.get_num_octaves()

        # First thing, get the freqs of the top octave
        freqs = librosa.cqt_frequencies(self.n_bins, self.fmin, bins_per_octave=self.bins_per_octave)[-self.bins_per_octave:]

        fmax_t = np.max(freqs)

        # Determine required resampling quality
        window = 'hann'
        Q = 1.0 / (2.0 ** (1. / self.bins_per_octave) - 1)
        filter_cutoff = fmax_t * (1 + 0.5 * librosa.filters.window_bandwidth(window) / Q)
        nyquist = SAMPLE_RATE / 2.0

        early_ds_count = early_downsample_count(nyquist, filter_cutoff, self.hop_length, n_octaves)
        return early_ds_count

    def get_expected_frames(self, audio):
        n_octaves = self.get_num_octaves()
        early_ds_count = self.get_early_ds_count()

        k = early_ds_count + n_octaves - 1
        k = np.arange(early_ds_count, k + 1)
        sig_lens = np.ceil(len(audio) / (2**k))
        hop_lens = self.hop_length // (2**k)
        num_hops = sig_lens // hop_lens
        num_frames = int(min(num_hops + 1))

        return num_frames

    def get_sample_range(self, num_frames):
        early_ds_count = self.get_early_ds_count()

        early_ds_factor = 2**early_ds_count
        max_samples = ((num_frames * self.hop_length // early_ds_factor) - 1) * early_ds_factor
        min_samples = max(1, max_samples - self.hop_length + 1)
        sample_range = np.arange(min_samples, max_samples + 1)
        return sample_range

    def process_audio(self, audio):
        cqt = librosa.cqt(audio, self.fs, self.hop_length, self.fmin, self.n_bins, self.bins_per_octave)

        cqt_mag = np.abs(cqt)

        # TODO - allow bypass
        cqt_log_db = 1 + librosa.core.amplitude_to_db(cqt_mag, ref=np.max) / 80

        # Add a channel dimension
        cqt_log_db = np.expand_dims(cqt_log_db, axis=0)

        return cqt_log_db

    @staticmethod
    def features_name():
        return 'cqt'

class MelSpec:
    def __init__(self, n_mels=229, n_fft=2048, hop_length=512, htk=False, norm=None):
        self.fs = SAMPLE_RATE
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
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
        mel = librosa.feature.melspectrogram(audio, self.fs, n_mels=self.n_mels,
                                             n_fft=self.n_fft, hop_length=self.hop_length,
                                             htk=self.htk, norm=self.norm)

        # TODO - allow bypass and abstract this to a parent class
        mel_log_db = 1 + librosa.core.power_to_db(mel, ref=np.max) / 80

        # Add a channel dimension
        mel_log_db = np.expand_dims(mel_log_db, axis=0)

        return mel_log_db

    @staticmethod
    def features_name():
        return 'melspec'

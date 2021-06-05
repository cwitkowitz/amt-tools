# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import FeatureModule

# Regular imports
import numpy as np
import librosa

# TODO - can use n_fft in get_times() if wanted to (parameterize offset?)


class MelSpec(FeatureModule):
    """
    Implements a Mel Spectrogram wrapper.
    """
    def __init__(self, sample_rate=16000, hop_length=512, decibels=True,
                 n_mels=229, n_fft=2048, win_length=None, htk=False):
        """
        Initialize parameters for the Mel Spectrogram.

        Parameters
        ----------
        See FeatureModule class for others...
        n_mels : int
          Number of bins (filters) in Mel spectrogram
        n_fft : int
          Length of the FFT window in spectrogram calculation
        win_length : int
          Number of samples to use for each frame;
          Must be less than or equal to n_fft;
          Defaults to n_fft if unspecified
        htk : bool
          Whether to use HTK formula instead of Slaney
        """

        super().__init__(sample_rate, hop_length, 1, decibels)

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.win_length = win_length
        self.htk = htk

    def process_audio(self, audio):
        """
        Get the Mel Spectrogram features for a piece of audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        feats : ndarray
          Post-processed features
        """

        # Calculate the Mel Spectrogram using librosa
        mel = librosa.feature.melspectrogram(y=audio,
                                             sr=self.sample_rate,
                                             n_mels=self.n_mels,
                                             n_fft=self.n_fft,
                                             hop_length=self.hop_length,
                                             win_length=self.win_length,
                                             htk=self.htk)

        # Post-process the Mel Spectrogram
        mel = super().post_proc(mel)

        return mel

    def to_decibels(self, feats):
        """
        Convert features to decibels (dB) units.

        Parameters
        ----------
        feats : ndarray
          Calculated power features

        Returns
        ----------
        feats : ndarray
          Calculated features in decibels
        """

        # Simply use the appropriate librosa function
        feats = librosa.core.power_to_db(feats, ref=np.max)

        return feats

# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import FeatureModule

# Regular imports
import numpy as np
import librosa

# TODO - can use n_fft in get_times() if wanted to (parameterize offset?)


class STFT(FeatureModule):
    """
    Implements a Spectrogram wrapper.
    """
    def __init__(self, sample_rate=16000, hop_length=512, decibels=True,
                 n_fft=2048, win_length=None):
        """
        Initialize parameters for the Mel Spectrogram.

        Parameters
        ----------
        See FeatureModule class for others...
        n_fft : int
          Length of the FFT window in spectrogram calculation
        win_length : int
          Number of samples to use for each frame;
          Must be less than or equal to n_fft;
          Defaults to n_fft if unspecified
        """

        super().__init__(sample_rate, hop_length, 1, decibels)

        self.n_fft = n_fft
        self.win_length = win_length

    def process_audio(self, audio):
        """
        Get the spectrogram features for a piece of audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        feats : ndarray
          Post-processed features
        """

        # Calculate the spectrogram using librosa
        spec = librosa.stft(y=audio,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length)
        # Take the magnitude of the spectrogram
        spec = np.abs(spec)

        # Post-process the Spectrogram
        spec = super().post_proc(spec)

        return spec

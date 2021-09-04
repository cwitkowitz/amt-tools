# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .stft import STFT

# Regular imports
import numpy as np
import librosa


class MelSpec(STFT):
    """
    Implements a Mel Spectrogram wrapper.
    """
    def __init__(self, sample_rate=16000, hop_length=512, decibels=True,
                 n_mels=229, n_fft=2048, win_length=None, center=True,
                 htk=False):
        """
        Initialize parameters for the Mel Spectrogram.

        Parameters
        ----------
        See STFT class for others...
        n_mels : int
          Number of bins (filters) in Mel spectrogram
        htk : bool
          Whether to use HTK formula instead of Slaney
        """

        super().__init__(sample_rate, hop_length, decibels, n_fft, win_length, center)

        self.n_mels = n_mels
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

        # Pad the audio if it is necessary to do so
        audio = super()._pad_audio(audio)

        # Calculate the Mel Spectrogram using librosa
        mel = librosa.feature.melspectrogram(y=audio,
                                             sr=self.sample_rate,
                                             n_mels=self.n_mels,
                                             n_fft=self.n_fft,
                                             hop_length=self.hop_length,
                                             win_length=self.win_length,
                                             center=self.center,
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

        # Only use maximum as reference if there is more than one frame
        ref = 1.0 if feats.shape[-1] == 1 else np.max

        # Simply use the appropriate librosa function
        feats = librosa.core.power_to_db(feats, ref=ref)

        return feats

    def get_feature_size(self):
        """
        Helper function to access dimensionality of features.

        Returns
        ----------
        feature_size : int
          Dimensionality along feature axis
        """

        feature_size = self.n_mels

        return feature_size

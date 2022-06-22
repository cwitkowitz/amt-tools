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

        super().__init__(sample_rate=sample_rate,
                         hop_length=hop_length,
                         decibels=decibels,
                         win_length=win_length,
                         center=center,
                         n_fft=n_fft)

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

        if audio.shape[-1] == 0:
            # Handle case of empty audio array
            return np.zeros((1, self.n_mels, 0))

        if not self.center:
            # Pad the audio to fill in a final frame
            audio = self.frame_pad(audio)

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

        # Simply use the appropriate librosa function
        feats = librosa.core.power_to_db(feats, ref=np.max)

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

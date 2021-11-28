# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .waveform import WaveformWrapper

# Regular imports
import numpy as np
import librosa


class STFT(WaveformWrapper):
    """
    Implements a Spectrogram wrapper.
    """
    def __init__(self, sample_rate=16000, hop_length=512, decibels=True,
                 win_length=None, center=True, n_fft=2048):
        """
        Initialize parameters for the Mel Spectrogram.

        Parameters
        ----------
        See WaveformWrapper class for others...
        n_fft : int
          Length of the FFT window in spectrogram calculation
        """

        super().__init__(sample_rate, hop_length, decibels, win_length, center)

        self.n_fft = n_fft

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

        # Pad the audio if it is necessary to do so
        audio = self._pad_audio(audio)

        # Calculate the spectrogram using librosa
        spec = librosa.stft(y=audio,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.win_length,
                            center=self.center)
        # Take the magnitude of the spectrogram
        spec = np.abs(spec)

        # Post-process the Spectrogram
        spec = super().post_proc(spec)

        return spec

    def get_feature_size(self):
        """
        Helper function to access dimensionality of features.

        Returns
        ----------
        feature_size : int
          Dimensionality along feature axis
        """

        feature_size = self.n_fft // 2 + 1

        return feature_size

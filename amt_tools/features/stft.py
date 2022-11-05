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
        win_length : int
          Number of samples to use for each frame;
          Must be less than or equal to n_fft;
          Defaults to n_fft if unspecified
        n_fft : int
          Length of the FFT window in spectrogram calculation
        """

        self.n_fft = n_fft

        if win_length is None:
            win_length = self.n_fft

        super().__init__(sample_rate=sample_rate,
                         hop_length=hop_length,
                         decibels=decibels,
                         win_length=win_length,
                         center=center)

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

        if audio.shape[-1] == 0:
            # Handle case of empty audio array
            return np.zeros((1, self.n_fft, 0))

        if not self.center:
            # Pad the audio to fill in a final frame
            audio = self.frame_pad(audio)

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

# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .waveform import WaveformWrapper

# Regular imports
from librosa.core import amplitude_to_db

import numpy as np


class SignalPower(WaveformWrapper):
    """
    Computes signal power at the frame-level.
    """
    def __init__(self, sample_rate=44100, hop_length=512, decibels=True, win_length=None, center=True):
        """
        Initialize parameters for computing signal power.

        Parameters
        ----------
        See WaveformWrapper class...
        """

        super().__init__(sample_rate=sample_rate,
                         hop_length=hop_length,
                         decibels=decibels,
                         win_length=win_length,
                         center=center)

    def process_audio(self, audio):
        """
        Get the signal power for each frame of a piece of audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        powers : ndarray
          Frame-level signal powers
        """

        # Split the audio into frames
        audio_frames = super().process_audio(audio)

        # Compute frame-level signal powers
        powers = np.sum(audio_frames ** 2, axis=-2) / self.win_length

        if self.decibels:
            # Convert to Decibels using the maximum
            # power (among this signal) as the reference
            powers = amplitude_to_db(powers, ref=np.max)

        return powers

    def get_feature_size(self):
        """
        Helper function to access dimensionality of features.

        Returns
        ----------
        feature_size : int
          Dimensionality along feature axis
        """

        # Simply one value (power) per frame
        feature_size = 1

        return feature_size

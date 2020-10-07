# My imports
# None of my imports used

# Regular imports
from abc import abstractmethod

import numpy as np

# TODO - FeatureCombo (stacking features)


class FeatureModule(object):
    """
    Implements a generic music feature extraction module.
    """

    def __init__(self, sample_rate, hop_length, decibels=True):
        """
        Initialize parameters common to all feature extraction modules.

        Parameters
        ----------
        sample_rate : int or float
          Assumed sampling rate for all audio
        hop_length : int or float
          Number of samples between feature frames
        decibels : bool
          Convert features to decibel (dB) units
        """

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.decibels = decibels

    @abstractmethod
    def get_expected_frames(self, audio):
        """
        Determine the number of frames the module will return
        for a piece of audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio
        """

        return NotImplementedError

    @abstractmethod
    def get_sample_range(self, num_frames):
        """
        Determine the range of audio samples which will produce
        features with a given number of frames

        Parameters
        ----------
        num_frames : int
          Number of frames for sample-range query
        """

        return NotImplementedError

    @abstractmethod
    def process_audio(self, audio):
        """
        Get features for a piece of audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio
        """

        return NotImplementedError

    @abstractmethod
    def to_decibels(self, feats):
        """
        Convert features to decibels (dB) units.

        Parameters
        ----------
        feats : ndarray
          Calculated features
        """

        return NotImplementedError

    def post_proc(self, feats):
        """
        Perform post-processing steps.

        Parameters
        ----------
        feats : ndarray
          Calculated features

        Returns
        ----------
        feats : ndarray
          Post-processed features
        """

        if self.decibels:
            # Convert to decibels (dB)
            feats = self.to_decibels(feats)

            # Assuming range of -80 to 0 dB, scale between 0 and 1
            feats = feats / 80
            feats = feats + 1
        else:
            # TODO - should anything be done here?
            pass

        # Add a channel dimension
        feats = np.expand_dims(feats, axis=0)

        return feats

    @abstractmethod
    def get_times(self, audio):
        """
        Determine the time, in seconds, associated with each sample
        or frame of audio.

        Parameters
        ----------
        audio: ndarray
          Mono-channel audio
        """

        return NotImplementedError

    @classmethod
    def features_name(cls):
        """
        Retrieve an appropriate tag, the class name, for the module.

        Returns
        ----------
        tag : str
          Name of the child class calling the function
        """

        return cls.__name__

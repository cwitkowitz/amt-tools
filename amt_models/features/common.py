# My imports
# None of my imports used

# Regular imports
from abc import abstractmethod

# TODO - feature stacking - FeatureCombo class
# TODO - put framify here - another param (frame_span?)
# TODO - add filterbank learning module


class FeatureModule:
    """
    Implements a generic music feature extraction module.
    """

    def __init__(self, sample_rate, hop_length):
        """
        Initialize parameters common to all feature extraction modules.

        Parameters
        ----------
        sample_rate : int or float
          Assumed sampling rate for all audio
        hop_length : int or float
          Number of samples between feature frames
        """

        self.sample_rate = sample_rate
        self.hop_length = hop_length

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
        audio: ndarray
          Mono-channel audio
        """

        return NotImplementedError

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

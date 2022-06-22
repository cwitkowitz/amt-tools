# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .. import tools

# Regular imports
from abc import abstractmethod

import numpy as np
import librosa

# TODO - take squared modulus of some of these?


class FeatureModule(object):
    """
    Implements a generic music feature extraction module wrapper.
    """

    def __init__(self, sample_rate, hop_length, num_channels, decibels=True):
        """
        Initialize parameters common to all feature extraction modules.

        Parameters
        ----------
        sample_rate : int or float
          Presumed sampling rate for all audio
        hop_length : int or float
          Number of samples between feature frames
        num_channels : int
          Number of independent feature channels
        decibels : bool
          Convert features to decibel (dB) units
        """

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.num_channels = num_channels
        self.decibels = decibels

    def get_expected_frames(self, audio):
        """
        Determine the number of frames the module will return
        for a piece of audio.

        This is the default behavior. It can be overridden.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        num_frames : int
          Number of frames expected
        """

        # Default the number of frames
        num_frames = 0

        if audio.shape[-1] != 0:
            # Simply the number of hops plus one
            num_frames = 1 + len(audio) // self.hop_length

        return num_frames

    def get_sample_range(self, num_frames):
        """
        Determine the range of audio samples which will produce features
        with a given number of frames.

        This is the default behavior. It can be overridden.

        Parameters
        ----------
        num_frames : int
          Number of frames for sample-range query

        Returns
        ----------
        sample_range : ndarray
          Valid audio signal lengths to obtain queried number of frames
        """

        # Default the sample range
        sample_range = np.array([0])

        if num_frames > 0:
            # Calculate the boundaries
            max_samples = num_frames * self.hop_length - 1
            min_samples = max(1, max_samples - self.hop_length + 1)

            # Construct an array ranging between the minimum and maximum number of samples
            sample_range = np.arange(min_samples, max_samples + 1)

        return sample_range

    def get_num_samples_required(self):
        """
        Determine the number of samples required to extract one full frame of features.

        Returns
        ----------
        num_samples_required : int
          Number of samples
        """

        # Maximum number of samples which still produces one frame
        num_samples_required = self.get_sample_range(1)[-1]

        return num_samples_required

    @staticmethod
    def divisor_pad(audio, divisor):
        """
        Pad audio such that it is divisible by the specified divisor.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio
        divisor : int
          Number by which the amount of audio samples should be divisible

        Returns
        ----------
        audio : ndarray
          Padded audio
        """

        # Determine how many samples would be needed such that the audio is evenly divisible
        pad_amt = divisor - (audio.shape[-1] % divisor)

        if pad_amt > 0 and pad_amt != divisor:
            # Pad the audio for divisibility
            audio = np.append(audio, np.zeros(pad_amt).astype(tools.FLOAT32), axis=-1)

        return audio

    def frame_pad(self, audio):
        """
        Pad the audio to fill out the final frame.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        audio : ndarray
          Padded audio
        """

        # We need at least this many samples
        divisor = self.get_num_samples_required()

        if audio.shape[-1] > divisor:
            # If above is satisfied, just pad for one extra hop
            divisor = self.hop_length

        # Pad the audio so it is divisible by the divisor
        audio = self.divisor_pad(audio, divisor)

        return audio

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

    def to_decibels(self, feats):
        """
        Convert features to decibels (dB) units.

        This is the default behavior. It can be overridden.

        Parameters
        ----------
        feats : ndarray
          Calculated amplitude features

        Returns
        ----------
        feats : ndarray
          Calculated features in decibels
        """

        # Simply use the appropriate librosa function
        feats = librosa.core.amplitude_to_db(feats, ref=np.max)

        return feats

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

            # TODO - make additional variable for 0/1 scaling
            # Assuming range of -80 to 0 dB, scale between 0 and 1
            feats = feats / 80
            feats = feats + 1

        # Add a channel dimension
        feats = np.expand_dims(feats, axis=0)

        return feats

    def get_times(self, audio):
        """
        Determine the time, in seconds, associated with each frame.

        This is the default behavior. It can be overridden.

        Parameters
        ----------
        audio: ndarray
          Mono-channel audio

        Returns
        ----------
        times : ndarray
          Time in seconds of each frame
        """

        # Determine the number of frames we will get
        num_frames = self.get_expected_frames(audio)

        frame_idcs = np.arange(num_frames)
        # Obtain the time of the sample at each hop
        times = librosa.frames_to_time(frames=frame_idcs,
                                       sr=self.sample_rate,
                                       hop_length=self.hop_length)

        return times

    def get_sample_rate(self):
        """
        Helper function to access sampling rate.

        Returns
        ----------
        sample_rate : int or float
          Presumed sampling rate for all audio
        """

        sample_rate = self.sample_rate

        return sample_rate

    def get_hop_length(self):
        """
        Helper function to access hop length.

        Returns
        ----------
        hop_length : int or float
          Number of samples between feature frames
        """

        hop_length = self.hop_length

        return hop_length

    def get_num_channels(self):
        """
        Helper function to access number of feature channels.

        Returns
        ----------
        num_channels : int
          Number of independent feature channels
        """

        num_channels = self.num_channels

        return num_channels

    @abstractmethod
    def get_feature_size(self):
        """
        Helper function to access dimensionality of features.
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

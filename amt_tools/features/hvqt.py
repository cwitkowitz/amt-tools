# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import FeatureModule
from .vqt import VQT

# Regular imports
import numpy as np
import librosa


class HVQT(FeatureModule):
    """
    Implements a Harmonic Variable-Q Transform wrapper.
    """
    def __init__(self, sample_rate=22050, hop_length=512, decibels=True,
                 fmin=None, harmonics=None, n_bins=84, bins_per_octave=12,
                 gamma=None):
        """
        Initialize parameters for the HVQT.

        Parameters
        ----------
        See FeatureModule & VQT class for others...
        fmin : float
          Center frequency of lowest filter in lowest harmonic
        harmonics : list of int or float
          Harmonics transforms to take, relative to transform at fmin
        """

        # Default the lowest center frequency to the note C1
        if fmin is None:
            # C1 by default
            fmin = librosa.note_to_hz('C1')
        self.fmin = fmin

        # Default the harmonics to those used in the DeepSalience paper
        if harmonics is None:
            harmonics = [0.5, 1, 2, 3, 4, 5]
        harmonics.sort()
        self.harmonics = harmonics

        super().__init__(sample_rate, hop_length, len(self.harmonics), decibels)

        modules = []
        # Construct a list of VQT modules for the harmonic transform
        for h in self.harmonics:
            # Center frequency for the harmonic
            fmin_h = h * fmin
            # Add a module for this harmonic's VQT
            modules += [VQT(sample_rate=sample_rate,
                            hop_length=hop_length,
                            decibels=decibels,
                            fmin=fmin_h,
                            n_bins=n_bins,
                            bins_per_octave=bins_per_octave,
                            gamma=gamma)]
        self.modules = modules

    def get_expected_frames(self, audio):
        """
        Determine the number of frames the module will return
        for a piece of audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        num_frames : int
          Number of frames expected
        """

        # Determine the expected number of frames for each harmonic
        num_frames = [module.get_expected_frames(audio)
                      for module in self.modules]

        # Take the minimum of these frame counts
        num_frames = min(num_frames)

        return num_frames

    def get_sample_range(self, num_frames):
        """
        Determine the range of audio samples which will produce
        features with a given number of frames. This will inevitably
        fall onto the highest harmonic's sample range.

        Parameters
        ----------
        num_frames : int
          Number of frames for sample-range query

        Returns
        ----------
        sample_range : ndarray
          Valid audio signal lengths to obtain queried number of frames
        """

        # Get the sample range of the highest harmonic's VQT
        sample_range = self.modules[-1].get_sample_range(num_frames)

        return sample_range

    def process_audio(self, audio):
        """
        Get the VQT features stacked across harmonics for a piece of audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        feats : ndarray
          Post-processed features
        """

        # Determine the frame cutoff (highest harmonic's output)
        num_frames = self.get_expected_frames(audio)

        feats = []
        # Take the VQT at each harmonic
        for module in self.modules:
            feats += [module.process_audio(audio)[..., :num_frames]]

        # Stack the features along the
        feats = np.concatenate(feats, axis=0)

        return feats

    def to_decibels(self, feats):
        """
        Convert features to decibels (dB) units.
        This will be taken care of in the lower-level VQT modules.

        Parameters
        ----------
        feats : ndarray
          Calculated amplitude features
        """

        return NotImplementedError

    def get_times(self, audio, at_start=False):
        """
        Determine the time, in seconds, associated with each frame.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio
        at_start : bool
          Whether time is associated with beginning of frame instead of center

        Returns
        ----------
        times : ndarray
          Time in seconds of each frame
        """

        # Use the times of the lowest harmonic's transform, trimmed to the expected frames
        times = self.modules[0].get_times(audio, at_start)[:self.get_expected_frames(audio)]

        return times

    def get_feature_size(self):
        """
        Helper function to access dimensionality of features.

        Returns
        ----------
        feature_size : int
          Dimensionality along feature axis
        """

        # Obtain the feature size from the lowest harmonic
        feature_size = self.modules[0].get_feature_size()

        return feature_size

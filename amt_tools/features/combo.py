# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import FeatureModule

# Regular imports
import numpy as np

# TODO - redundant saving - maybe it should be a function of feature extraction module and not dataset
# TODO - concatenate (feature dimension) option
#      - if not selected, try to concat on channel dimension or put in list if not possible


class FeatureCombo(FeatureModule):
    """
    Implements a wrapper for a combination of multiple feature extraction modules.
    """
    def __init__(self, modules):
        """
        Initialize parameters for the feature combination.

        Parameters
        ----------
        modules : list of FeatureModules
          Post-initialization feature extraction modules
        """

        self.modules = modules

    def get_expected_frames(self, audio):
        """
        Determine the number of frames we expect from provided audio.

        Parameters
        ----------
        audio: ndarray
          Mono-channel audio

        Returns
        ----------
        num_frames : int
          Number of frames which will be generated for given audio
        """

        # Gather expected counts from inner modules
        num_frames = [module.get_expected_frames(audio)
                      for module in self.modules]

        # TODO - equality constraint depend on how I deal with following TODOs
        # Make sure the expected frame count of each module is the same
        assert len(set(num_frames)) == 1
        # Collapse the expected frame counts
        num_frames = num_frames[0]

        return num_frames

    def get_sample_range(self, num_frames):
        """
        Determine the range of audio samples which will produce features
        with a given number of frames.

        Parameters
        ----------
        num_frames : int
          Number of frames for sample-range query

        Returns
        ----------
        sample_range : ndarray
          Valid audio signal lengths to obtain queried number of frames
        """

        sample_range = None

        # Loop through feature extraction modules
        for module in self.modules:
            if sample_range is None:
                # On first iteration, start with the sample range of the current module
                sample_range = module.get_sample_range(num_frames)
            else:
                # On subsequent iterations, remove any non-overlapping sample counts
                sample_range = np.intersect1d(sample_range, module.get_sample_range(num_frames))

        return sample_range

    def process_audio(self, audio):
        """
        Get the features for a piece of audio.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        feats : ndarray
          Post-processed features
        """

        # TODO - more sophisticated stacking/padding may be required - not if I return a list and let user figure it out

        feats = []
        # Gather features from inner modules
        for module in self.modules:
            mod_feats = module.process_audio(audio)
            # Add to the list if fixed features were calculated
            if mod_feats is not None:
                feats += [mod_feats]

        # If the list is still empty, all modules produced None
        if len(feats) == 0:
            # Collapse to a single None
            feats = None

        if feats is not None:
            # Stack the features along the channel dimension
            # TODO - this will break if dimensionality mismatch
            # TODO - I should just return the list if I can't concatenate
            feats = np.concatenate(feats, axis=0)

        return feats

    def get_times(self, audio):
        """
        Determine the time, in seconds, associated with frame.

        Parameters
        ----------
        audio: ndarray
          Mono-channel audio

        Returns
        ----------
        times : ndarray
          Time in seconds of each frame
        """

        # Gather times from inner modules
        times = np.array([module.get_times(audio) for module in self.modules])

        # Obtain the expected number of frames for the audio
        num_frames = self.get_expected_frames(audio)

        # Check if the time arrays are all the same
        #if len(np.unique(times).size) == 1:
        # If so, collapse into one time array
        times = times[0]

        return times

    def get_sample_rate(self):
        """
        Gather sample rates from inner modules, making sure they are equal.

        Returns
        ----------
        sample_rate : int or float
          Presumed sampling rate for all audio
        """

        # TODO - remove equality constraint - return min/max? - or only collapse if size(unique) == 1 like above

        sample_rate = [module.get_sample_rate() for module in self.modules]
        # Make sure the sample_rate of each module is the same
        assert len(set(sample_rate)) == 1
        # Collapse the sample rates
        sample_rate = sample_rate[0]

        return sample_rate

    def get_hop_length(self):
        """
        Gather hop lengths from inner modules, making sure they are equal.

        Returns
        ----------
        hop_length : int or float
          Number of samples between feature frames
        """

        # TODO - remove equality constraint - return min/max? - or only collapse if size(unique) == 1 like above

        hop_length = [module.get_hop_length() for module in self.modules]
        # Make sure the hop length of each module is the same
        assert len(set(hop_length)) == 1
        # Collapse the hop lengths
        hop_length = hop_length[0]

        return hop_length

    def get_num_channels(self):
        """
        Sum number of feature channels from inner modules.

        Returns
        ----------
        num_channels : int
          Number of independent feature channels
        """

        num_channels = sum([module.get_num_channels() for module in self.modules])

        return num_channels

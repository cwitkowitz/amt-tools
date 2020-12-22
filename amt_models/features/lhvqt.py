# My imports
from features.common import *

# Regular imports
from lhvqt.lhvqt_comb import LHVQT_COMB as lc_type
from lhvqt.lhvqt import LHVQT as _LHVQT

# TODO - different get_sample_range() behavior if padding vs. not padding for extra frame


class LHVQT(FeatureModule):
    """
    Implements a Harmonic Variable-Q Transform wrapper.
    """
    def __init__(self, sample_rate=44100, hop_length=512, decibels=True, lhvqt=None, lvqt=None,
                 fmin=None, harmonics=None, n_bins=84, bins_per_octave=12, gamma=0, random=False,
                 max_p=1, batch_norm=True):
        """
        Initialize parameters for the HVQT.

        Parameters
        ----------
        See FeatureModule and chosen LHVQT & LVQT class for others...
        lhvqt : type
          Class definition of chosen harmonic-level module
        lvqt : type
          Class definition of chosen lower-level module
        """

        # Default the class definition for the harmonic-level
        if lhvqt is None:
            # Original LVQT module
            lhvqt = _LHVQT

        self.lhvqt = lhvqt(fmin=fmin,
                           harmonics=harmonics,
                           lvqt=lvqt,
                           fs=sample_rate,
                           hop_length=hop_length,
                           n_bins=n_bins,
                           bins_per_octave=bins_per_octave,
                           gamma=gamma,
                           random=random,
                           max_p=max_p,
                           to_db=decibels,
                           batch_norm=batch_norm)

        # TODO - this is kinda sloppy
        if isinstance(self.lhvqt, lc_type):
            num_channels = 1
        else:
            num_channels = len(self.lhvqt.harmonics)

        super().__init__(sample_rate, hop_length, num_channels, decibels)

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

        # Use the function of the harmonic-level module
        num_frames = self.lhvqt.get_expected_frames(audio)

        return num_frames

    def process_audio(self, audio):
        """
        Return None to indicate that the audio are the true features
        to be processed by this module within an MIR model.

        Parameters
        ----------
        audio : ndarray
          Mono-channel audio

        Returns
        ----------
        None
        """

        return None

# My imports
from .common import FeatureModule

# Regular imports
from lhvqt.lhvqt_comb import LHVQT_COMB as lc_type
from lhvqt.lhvqt import LHVQT as _LHVQT

# TODO - get sample_range() is invalid (for non-divisible max_p) due to change in LVQT.get_expected_frames()


class LHVQT(FeatureModule):
    """
    Implements a Harmonic Variable-Q Transform wrapper.
    """
    def __init__(self, sample_rate=44100, hop_length=512, decibels=True, lhvqt=None, lvqt=None,
                 fmin=None, harmonics=None, n_bins=84, bins_per_octave=12, gamma=None, max_p=1,
                 random=False, update=True, batch_norm=True, var_drop=True):
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
                           max_p=max_p,
                           random=random,
                           update=update,
                           to_db=decibels,
                           db_to_prob=True,
                           batch_norm=batch_norm,
                           var_drop=var_drop)

        # If using harmonic comb variant of LHVQT, everything collapses to one channel
        num_channels = 1 if isinstance(self.lhvqt, lc_type) else len(self.lhvqt.harmonics)

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

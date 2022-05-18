# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .hvqt import HVQT


class HCQT(HVQT):
    """
    A simple wrapper (for convenience) for a Harmonic Constant-Q Transform,
    """
    def __init__(self, sample_rate=22050, hop_length=512, decibels=True,
                 fmin=None, harmonics=None, n_bins=84, bins_per_octave=12):
        """
        Initialize parameters for the HCQT.

        Parameters
        ----------
        See HVQT class...
        """

        super().__init__(sample_rate, hop_length, decibels, fmin, harmonics, n_bins, bins_per_octave, gamma=0)

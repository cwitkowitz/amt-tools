# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .vqt import VQT


class CQT(VQT):
    """
    A simple wrapper (for convenience) for a Constant-Q Transform,
    which is a special case of the Variable-Q Transform.
    """
    def __init__(self, sample_rate=22050, hop_length=512, decibels=True,
                 fmin=None, n_bins=84, bins_per_octave=12):
        """
        Initialize parameters for the CQT.

        Parameters
        ----------
        See VQT class...
        """

        super().__init__(sample_rate, hop_length, decibels, fmin, n_bins, bins_per_octave, gamma=0)

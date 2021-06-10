# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from . import constants

# Regular imports
from abc import abstractmethod

import numpy as np
import librosa

# TODO - some sort of combo class for multi-instrument transcription?


class InstrumentProfile(object):
    """
    Implements a generic instrument profile.
    """

    def __init__(self, low, high):
        """
        Initialize the common properties among instrument profiles.

        Parameters
        ----------
        low : int
          Lowest midi note playable on the instrument
        high : int
          Highest midi note playable on the instrument
        """

        self.low = low
        self.high = high

    def get_midi_range(self):
        """
        Obtain the instrument's range of MIDI pitches.

        Returns
        ----------
        pitch_range : ndarray
          Ascending array of pitches playable on the instrument
        """

        # Create an ascending array from the lowest pitch to the highest pitch
        pitch_range = np.arange(self.low, self.high + 1)

        return pitch_range

    def get_range_len(self):
        """
        Determine how many discrete pitches can be played by the instrument.

        Returns
        ----------
        range_len : int
          Number of pitches instrument supports
        """

        # Just count the size of the MIDI range
        range_len = self.get_midi_range().size

        return range_len


class PianoProfile(InstrumentProfile):
    """
    Implements a basic piano profile.
    """

    def __init__(self, low=None, high=None):
        """
        Initialize the profile and establish parameter defaults in function signature.

        Parameters
        ----------
        See InstrumentProfile class...
        """

        # Set defaults if parameters are not specified
        if low is None:
            low = constants.DEFAULT_PIANO_LOWEST_PITCH
        if high is None:
            high = constants.DEFAULT_PIANO_HIGHEST_PITCH

        super().__init__(low, high)


class TablatureProfile(InstrumentProfile):
    """
    Implements a generic instrument profile for instruments
    with multiple degrees of freedom (e.g. strings).
    """

    def __init__(self, tuning, num_pitches):
        """
        Initialize the common properties among tablature profiles.

        Parameters
        ----------
        tuning : list of str
          Name of lowest note playable on each degree of freedom
        num_pitches : int
          Number of pitches playable on each degree of freedom
        """

        self.tuning = tuning
        self.num_pitches = num_pitches

        # Convert the named tuning to respective MIDI pitches
        midi_tuning = self.get_midi_tuning()

        # Determine the pitch boundaries of the instrument
        low, high = midi_tuning[0], midi_tuning[-1] - 1 + self.num_pitches

        super().__init__(low, high)

    def get_num_dofs(self):
        """
        Determine how many degrees of freedom (e.g. strings) are present on the instrument.

        Returns
        ----------
        num_dofs : int
          Number of degrees of freedom instrument supports
        """

        # This is intrinsically defined by the
        # amount of entries in the specified tuning
        num_dofs = len(self.tuning)

        return num_dofs

    def get_midi_tuning(self):
        """
        Determine the instruments tuning in MIDI.

        Returns
        ----------
        midi_tuning : list of int
          MIDI pitch of lowest note playable on each degree of freedom
        """

        # Convert the named pitches to MIDI
        midi_tuning = librosa.note_to_midi(self.tuning)

        return midi_tuning


class GuitarProfile(TablatureProfile):
    """
    Implements a basic guitar profile.
    """

    def __init__(self, tuning=None, num_frets=None):
        """
        Initialize the profile and establish parameter defaults in function signature.

        Parameters
        ----------
        tuning : list of str
          Name of lowest note playable on each degree of freedom
        num_frets : int
          Number of frets on guitar (or at least the number used)
        """

        # Set defaults if parameters are not specified
        if tuning is None:
            tuning = constants.DEFAULT_GUITAR_TUNING
        if num_frets is None:
            num_frets = constants.DEFAULT_GUITAR_NUM_FRETS

        # Plus one for open string
        num_pitches = num_frets + 1

        super().__init__(tuning, num_pitches)

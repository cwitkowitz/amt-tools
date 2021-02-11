# My imports
# None of my imports used

# Regular imports
from abc import abstractmethod

import numpy as np
import librosa

# TODO - turn this into classes.py and add estimator + evaluator?
# TODO - tablature profile


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
        pitch_range = np.arange(self.low, self.high + 1)
        return pitch_range

    def get_range_len(self):
        range_len = self.get_midi_range().size
        return range_len

    @abstractmethod
    def get_multi_range(self):
        return NotImplementedError


class PianoProfile(InstrumentProfile):
    def __init__(self, low=21, high=108):
        super().__init__(low, high)

    def get_multi_range(self):
        multi_range = np.expand_dims(self.get_midi_range(), axis=0)
        return multi_range


class GuitarProfile(InstrumentProfile):
    def __init__(self, tuning=None, num_frets=19):
        if tuning is None:
            tuning = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']

        self.tuning = tuning
        self.num_frets = num_frets

        self.num_strings = len(self.tuning)

        midi_tuning = self.get_midi_tuning()
        low = midi_tuning[0].item()
        high = (midi_tuning[-1] + num_frets).item()
        super().__init__(low, high)

    def get_midi_tuning(self):
        midi_tuning = np.array([librosa.note_to_midi(self.tuning)]).T
        return midi_tuning

    def get_multi_range(self):
        multi_range = self.get_midi_tuning()
        multi_range = np.tile(multi_range, (1, self.num_frets + 1))
        semitones = np.arange(0, self.num_frets + 1)
        multi_range = multi_range + semitones
        return multi_range

# TODO - combo for multiple instrument transcription

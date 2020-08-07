# My imports
# None of my imports used

# Regular imports
from abc import abstractmethod

# TODO - build data proc module on top which is general - i.e. whole/local cqt or filterbank learning module


class FeatureModule:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    @abstractmethod
    def get_expected_frames(self, audio):
        return NotImplementedError

    @abstractmethod
    def get_sample_range(self, num_frames):
        return NotImplementedError

    @abstractmethod
    def process_audio(self, audio):
        return NotImplementedError

    @classmethod
    def features_name(cls):
        return cls.__name__

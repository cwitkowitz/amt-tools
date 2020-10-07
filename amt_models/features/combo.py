# My imports
from features.common import *

# Regular imports
# No imports used

import numpy as np


class Combo(FeatureModule):
    def __init__(self, modules):

        self.modules = modules

    def get_expected_frames(self, audio):
        num_frames = []
        for module in self.modules:
            num_frames += [module.get_expected_frames(audio)]

        return num_frames

    def get_sample_range(self, num_frames):
        sample_range = None
        for module in self.modules:
            if sample_range is None:
                sample_range = module.get_sample_range(num_frames)
            else:
                sample_range = np.intersect1d(sample_range, module.get_sample_range(num_frames))

        return sample_range

    def process_audio(self, audio):
        feats = []
        for module in self.modules:
            feats += [module.process_audio(audio)]

        # TODO - this will break if dimensionality mismatch
        feats = np.concatenate(feats, axis=0)

        return feats

    @abstractmethod
    def get_times(self, audio):
        times = None
        for module in self.modules:
            if times is None:
                times = module.get_times(audio)
            else:
                assert (times == module.get_times(audio)).all()

        return NotImplementedError

# My imports
from features.common import *

# Regular imports
# No imports used

import numpy as np

# TODO - redundant saving


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
        max_frames = max(self.get_expected_frames(audio))
        min_samples = min(self.get_sample_range(max_frames))
        # TODO - if sample range is empty, just pad lower with frames of zeros
        padding = min_samples - audio.shape[-1]
        if padding > 0:
            shape = tuple(audio.shape[:-1]) + tuple([padding])
            audio = np.concatenate((audio, np.zeros(shape)), axis=-1)

        feats = []
        for module in self.modules:
            mod_feats = module.process_audio(audio)
            if mod_feats is not None:
                feats += [mod_feats]

        if len(feats) == 0:
            feats = None

        # TODO - this will break if dimensionality mismatch
        feats = np.concatenate(feats, axis=0)

        return feats

    @abstractmethod
    def get_times(self, audio):
        times = None
        #for module in self.modules:
        #    if times is None:
        #        times = module.get_times(audio)
        #    else:
        #        # TODO - this seems strict, but it also makes sense
        #        assert (times == module.get_times(audio)).all()
        # TODO - this doesn't seem like a good permanent solution
        times = self.modules[np.argmax(self.get_expected_frames(audio))].get_times(audio)

        return times

    def get_sample_rate(self):
        sample_rate = [module.get_sample_rate() for module in self.modules]
        # TODO - is this always valid...? i think so
        assert len(set(sample_rate)) == 1
        sample_rate = sample_rate[0]
        return sample_rate

    def get_hop_length(self):
        hop_length = [module.get_hop_length() for module in self.modules]
        # TODO - is this always valid...? i think so
        assert len(set(hop_length)) == 1
        hop_length = hop_length[0]
        return hop_length

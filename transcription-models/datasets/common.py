# My imports
from features.cqt import CQT

from tools.utils import *

# Regular imports
from torch.utils.data import Dataset
from abc import abstractmethod
from random import randint
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import shutil
import os

# TODO - MusicNet already implemented - add an easy ref and maybe some functions to make it compatible
# TODO - validate methods for each data entry - such as tabs, notes, frames, etc.
# TODO - ComboDataset


class TranscriptionDataset(Dataset):
    def __init__(self, base_dir, splits, hop_length, sample_rate, data_proc, frame_length, split_notes, reset_data, seed):
        self.base_dir = base_dir
        if self.base_dir is None:
            self.base_dir = os.path.join(HOME, 'Desktop', 'Datasets', self.dataset_name())

        # Check if the dataset exists in memory
        if not os.path.isdir(self.base_dir):
            # Download the dataset if it is missing
            self.download(self.base_dir)

        self.splits = splits
        if self.splits is None:
            self.splits = self.available_splits()

        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.data_proc = data_proc
        if self.data_proc is None:
            self.data_proc = CQT()

        self.frame_length = frame_length

        if self.frame_length is None:
            # Transcribe whole tracks at a time
            self.seq_length = None
        else:
            self.seq_length = max(self.data_proc.get_sample_range(self.frame_length))

        self.reset_data = reset_data

        if os.path.exists(self.get_gt_dir()) and self.reset_data:
            shutil.rmtree(self.get_gt_dir())
        os.makedirs(self.get_gt_dir(), exist_ok=True)

        if os.path.exists(self.get_feats_dir()) and self.reset_data:
            shutil.rmtree(self.get_feats_dir())
        os.makedirs(self.get_feats_dir(), exist_ok=True)

        # TODO - do something with the seed - training seed and common seed?

        # Load the paths of the audio tracks
        self.tracks = []

        for split in self.splits:
            self.tracks += self.get_tracks(split)

        self.data = {}

        # Loading ground truth
        for track in tqdm(self.tracks):
            self.data[track] = self.load(track)

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        track = self.tracks[index]

        data = deepcopy(self.data[track])

        feats_path = self.get_feats_dir(track)

        if os.path.exists(feats_path):
            # TODO - feats_name = self.data_proc.features_name()?
            feats_dict = np.load(feats_path)
            feats = feats_dict['feats']
            times = feats_dict['times']
        else:
            # TODO - invoke data_proc only once unless fb learn
            feats = self.data_proc.process_audio(data['audio'])
            times = self.data_proc.get_times(data['audio'])
            np.savez(feats_path, feats=feats, times=times)

        data['feats'] = feats
        data['times'] = times

        if self.frame_length is not None:
            # TODO - how much will it hurt to pad with zeros instead of actual previous/later frames? - TabCNN

            # TODO - note splitting here for sample start - check Google's code
            """
            note_intervals = SAMPLE_RATE * data['notes'][:, :2]

            valid_start = False
            while not valid_start:
            """
            data = self.slice_track(data)
            data.pop('notes')

        # TODO - make this a func - def conv_batch('') - fold into batching function
        #if 'tabs' in data.keys():
        #    data['tabs'] = data['tabs'].astype('float32')
        if 'frames' in data.keys():
            data['frames'] = data['frames'].astype('float32')
        if 'onsets' in data.keys():
            data['onsets'] = data['onsets'].astype('float32')

        return data

    def slice_track(self, data, sample_start=None, seq_length=None, snap_to_frame=False):
        track_id = data['track']

        if seq_length is None:
            if self.seq_length is not None:
                seq_length = self.seq_length
            else:
                return data

        if sample_start is None:
            # This well mess up deterministic behavior if called in validation loop
            sample_start = randint(0, len(data['audio']) - seq_length)

        frame_start = sample_start // self.hop_length
        frame_end = frame_start + self.frame_length

        if snap_to_frame:
            sample_start = frame_start * self.hop_length

        sample_end = sample_start + seq_length

        data['audio'] = data['audio'][sample_start: sample_end]
        data['feats'] = data['feats'][:, :, frame_start: frame_end]
        data['times'] = data['times'][frame_start : frame_end + 1]

        if 'tabs' in data.keys():
            data['tabs'] = data['tabs'][:, frame_start: frame_end]
        if 'frames' in data.keys():
            data['frames'] = data['frames'][:, frame_start: frame_end]
        if 'onsets' in data.keys():
            data['onsets'] = data['onsets'][:, frame_start: frame_end]

        if 'notes' in data:
            # Notes is popped off dictionary in get_item()
            notes = data['notes']
        else:
            notes = self.data[track_id]['notes']

        sec_start = sample_start / self.sample_rate
        sec_stop = sample_end / self.sample_rate
        notes = notes[notes[:, 0] > sec_start]
        # TODO - make the stop time the min between the max time and the actually time
        notes = notes[notes[:, 0] < sec_stop]

        notes[:, 0] = notes[:, 0] - sec_start
        notes[:, 1] = notes[:, 1] - sec_start

        data['notes'] = notes

        return data

    @abstractmethod
    def get_tracks(self, split):
        return NotImplementedError

    @abstractmethod
    def load(self, track):
        # Default data to None (not existing)
        data = None

        # Determine the expected path to the track's data
        gt_path = self.get_gt_dir(track)

        # Check if an entry for the data exists
        if os.path.exists(gt_path):
            # Load the data if it exists
            data = dict(np.load(gt_path))

        # If the data was not previously generated
        if data is None:
            # Initialize a new dictionary
            data = {}

        # Add the track ID to the dictionary
        data['track'] = track
        # TODO - can add sample_rate and hop_length if necessary

        return data

    def get_gt_dir(self, track=None):
        path = os.path.join(GEN_DATA_DIR, self.dataset_name(), 'gt')
        if track is not None:
            path = os.path.join(path, track + '.npz')
        return path

    def get_feats_dir(self, track=None):
        path = os.path.join(GEN_DATA_DIR, self.dataset_name(), self.data_proc.features_name())
        if track is not None:
            path = os.path.join(path, track + '.npz')
        return path

    @staticmethod
    @abstractmethod
    def available_splits():
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def dataset_name():
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def download(save_dir):
        return NotImplementedError

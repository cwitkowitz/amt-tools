# My imports
from tools.constants import *
from tools.dataproc import *
from tools.utils import *

# Regular imports
from mir_eval.io import load_valued_intervals
from torch.utils.data import Dataset
from abc import abstractmethod
from random import randint
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import mirdata
import shutil
import os

# TODO - ComboDataset


class TranscriptionDataset(Dataset):
    def __init__(self, base_dir, splits, hop_length, data_proc, frame_length, split_notes, reset_data, seed):

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
            # TODO - different key for each module?
            feats = np.load(feats_path)['feats']
        else:
            # TODO - invoke data_proc only once unless fb learn
            feats = self.data_proc.process_audio(data['audio'])
            np.savez(feats_path, feats=feats)

        data['feats'] = feats

        if self.frame_length is not None:
            # TODO - how much will it hurt to pad with zeros instead of actual previous/later frames?

            # TODO - note splitting here for sample start - check Google's code
            """
            note_intervals = SAMPLE_RATE * data['notes'][:, :2]

            valid_start = False
            while not valid_start:
            """
            sample_start = randint(0, len(data['audio']) - self.seq_length)

            frame_start = sample_start // self.hop_length
            frame_end = frame_start + self.frame_length

            # TODO - quantize at even frames or start where sampled?
            #sample_start = frame_start * self.hop_length
            sample_end = sample_start + self.seq_length

            data['audio'] = data['audio'][sample_start : sample_end]
            data['tabs'] = data['tabs'][:, :, frame_start : frame_end]
            data['feats'] = feats[:, :, frame_start : frame_end]
            data.pop('notes')

        return data

    @abstractmethod
    def get_tracks(self, split):
        return NotImplementedError

    @abstractmethod
    def load(self, track):
        data = None
        gt_path = self.get_gt_dir(track)
        if os.path.exists(gt_path):
            data = dict(np.load(gt_path))
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

# TODO - MusicNet already implemented - add an easy ref and maybe some functions to make it compatible

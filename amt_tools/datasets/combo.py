# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import TranscriptionDataset

# Regular imports
import numpy as np
import mirdata
import shutil
import os

# TODO - ComboDataset
# TODO - clean this up and verify


class DatasetCombo(TranscriptionDataset):
    """
    Implements the combination of multiple datasets.
    """

    def __init__(self, datasets, splits):
        """
        """

        self.datasets = datasets
        self.splits = splits

        self.tracks = []
        for split in self.splits:
            self.tracks += self.get_tracks(split)

    def get_tracks(self, split):
        """
        """

        tracks = []
        for dataset in self.datasets:
            if split in dataset.available_splits():
                tracks += dataset.get_tracks(split)

        return tracks

    def load(self, track):
        """
        """

        for dataset in self.datasets:
            if track in dataset.tracks:
                data = dataset.load(track)
                break

        return data

    def available_splits(self):
        """
        """

        splits = []
        for dataset in self.datasets:
            splits += dataset.available_splits()

        return splits

    @staticmethod
    def download(save_dir):
        """
        Each constituent dataset will be downloaded,
        if necessary, at the time of it's initialization.

        Parameters
        ----------
        save_dir : string
          Directory in which to save the contents of GuitarSet
        """

        return NotImplementedError

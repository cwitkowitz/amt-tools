# My imports
from features.cqt import CQT

from tools.conversion import *
from tools.utils import *

# Regular imports
from torch.utils.data import Dataset
from abc import abstractmethod
from random import randint
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import mir_eval
import librosa
import shutil
import os

# TODO - MusicNet already implemented - add an easy ref and maybe some functions to make it compatible
# TODO - validate methods for each data entry - such as tabs, notes, frames, etc.
# TODO - ComboDataset


class TranscriptionDataset(Dataset):
    """
    Implements a generic music transcription dataset.
    """

    def __init__(self, base_dir, splits, hop_length, sample_rate, data_proc, num_frames, split_notes, reset_data, seed):
        """
        Initialize parameters common to all datasets as fields and instantiate
        as a PyTorch Dataset.

        Parameters
        ----------
        base_dir : string
          Path to the directory containing the dataset
        splits : list of strings
          Names of tracks to include in this partition
        hop_length : int
          Number of samples between frames
        sample_rate : int or float
          Number of samples per second of audio
        data_proc : FeatureModel (features/common.py)
          Feature extraction model to use for the dataset
        num_frames : int
          Number of frames per data sample
        split_notes : bool
          TODO - better description when I have code for this
          Flag to avoiding cutting samples in between notes
        reset_data : bool
          Flag to re-generate extracted features and ground truth data if it already exists
        seed : int
          TODO - this is currently unused - I should use it instead of assuming a seed has been set
          The seed for random number generation
        """

        self.base_dir = base_dir
        # Select a default base directory path if none was provided
        if self.base_dir is None:
            self.base_dir = os.path.join(HOME, 'Desktop', 'Datasets', self.dataset_name())

        # Check if the dataset exists in memory
        if not os.path.isdir(self.base_dir):
            # Download the dataset if it is missing
            self.download(self.base_dir)

        self.splits = splits
        # Choose all available dataset splits if none were provided
        if self.splits is None:
            self.splits = self.available_splits()

        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.data_proc = data_proc
        # Default the feature extraction to a plain CQT if none was provided
        if self.data_proc is None:
            self.data_proc = CQT()

        self.num_frames = num_frames
        # Determine the number of samples per data point
        if self.num_frames is None:
            # Transcribe whole tracks at a time (all samples)
            self.seq_length = None
        else:
            # The maximum number of samples which give the number of frames
            self.seq_length = max(self.data_proc.get_sample_range(self.num_frames))

        self.reset_data = reset_data
        # Remove any saved ground-truth for the dataset if reset_data is selected
        if os.path.exists(self.get_gt_dir()) and self.reset_data:
            shutil.rmtree(self.get_gt_dir())
        # Make sure a directory exists for saving and loading ground-truth
        os.makedirs(self.get_gt_dir(), exist_ok=True)

        # Remove any saved features for the tracks in the dataset if reset_data is selected
        if os.path.exists(self.get_feats_dir()) and self.reset_data:
            shutil.rmtree(self.get_feats_dir())
        # Make sure a directory exists for saving and loading features
        os.makedirs(self.get_feats_dir(), exist_ok=True)

        self.tracks = []
        # Aggregate all the track names from the selected splits
        for split in self.splits:
            self.tracks += self.get_tracks(split)

        self.data = {}
        # Load the ground-truth for each track into RAM
        for track in tqdm(self.tracks):
            self.data[track] = self.load(track)
            assert self.validate_track(self.data[track])

    def __len__(self):
        """
        Defines the notion of length for the dataset - used by PyTorch Dataset class.

        Returns
        ----------
        length : int
          Number of tracks in the dataset partition
        """

        length = len(self.tracks)
        return length

    def __getitem__(self, index):
        """
        Retrieve the (potentially randomly sliced) item associated with the selected index.

        Parameters
        ----------
        index : int
          Index of sampled track

        Returns
        ----------
        data : dict
          Dictionary containing the features and ground-truth data for the sampled track
        """

        # Get the name of the track
        track_id = self.tracks[index]
        # Slice the track's features and ground-truth
        data = self.get_track_data(track_id)
        # Remove the notes, as they cannot be batched
        data.pop('notes')

        return data

    # TODO - I probably need to decouple this sh
    def get_track_data(self, track_id, sample_start=None, seq_length=None, snap_to_frame=True):
        """
        Get the features and ground truth for a track within a time interval.

        Parameters
        ----------
        track_id : string
          Name of track data to fetch
        sample_start : int
          Sample with which to begin the slice
        seq_length : int
          Number of samples to take for the slice
        snap_to_frame : bool
          Whether to begin exactly on frame boundaries or loose samples

        Returns
        ----------
        data : dict
          Dictionary with each entry sliced for the random or provided interval
        """

        # Copy the track's ground-truth data into a local dictionary
        data = deepcopy(self.data[track_id])

        # Determine the path to the track's features
        feats_path = self.get_feats_dir(track_id)

        # Check if the features already exist
        # TODO - may need to modify when filterbank learning is possible
        if os.path.exists(feats_path):
            # If so, load the features
            feats_dict = np.load(feats_path)
            feats = feats_dict['feats']
            times = feats_dict['times']
        else:
            # If not, calculate the features and save them
            feats = self.data_proc.process_audio(data['audio'])
            # TODO - is there any point to saving times?
            times = self.data_proc.get_times(data['audio'])
            np.savez(feats_path, feats=feats, times=times)

        # Add the features to the data dictionary
        data['feats'] = feats
        data['times'] = times

        # TODO - note splitting occurs in here - check Google's code for procedure

        # Check to see if a specific sequence length was given
        if seq_length is None:
            # If not, and this Dataset object has a sequence length, use it
            if self.seq_length is not None:
                seq_length = self.seq_length
            # Otherwise, we assume the whole track is desired and perform no further actions
            else:
                return data

        # If a specific starting sample was not provided, sample one randomly
        if sample_start is None:
            # TODO - boolean for training vs. testing to determing which RNG to call
            sample_start = randint(0, len(data['audio']) - seq_length)

        # Determine the frames contained in this slice
        frame_start = sample_start // self.hop_length
        frame_end = frame_start + self.num_frames

        # Snap the sample_start to the nearest frame boundary if snap_to_frame is selected
        if snap_to_frame:
            sample_start = frame_start * self.hop_length

        # Calculate the last sample included in the slice
        sample_end = sample_start + seq_length

        # Slice the features
        data['audio'] = data['audio'][sample_start : sample_end]
        data['feats'] = data['feats'][:, :, frame_start: frame_end]
        data['times'] = data['times'][frame_start : frame_end + 1]

        # Slice the ground-truth
        if 'tabs' in data.keys():
            data['tabs'] = data['tabs'][:, frame_start: frame_end]
        if 'pianoroll' in data.keys():
            data['pianoroll'] = data['pianoroll'][:, frame_start: frame_end]
        if 'onsets' in data.keys():
            data['onsets'] = data['onsets'][:, frame_start: frame_end]

        # Determine if there is note ground-truth and get it
        if 'notes' in data.keys():
            # Notes entry has not been popped or never existed
            notes = data['notes']
        elif 'notes' in self.data[data['track']].keys():
            # Notes entry has been popped and must be added again
            notes = self.data[data['track']]['notes']
        else:
            notes = None

        # Slice the ground-truth notes
        if notes is not None:
            # TODO - potentially make this a function - get_notes_in_range()
            # Determine the time in seconds of the boundary samples
            sec_start = sample_start / self.sample_rate
            sec_stop = sample_end / self.sample_rate

            # Remove notes with onsets before the slice start time
            notes = notes[notes[:, 0] > sec_start]
            # Remove notes with onsets after the slice stop time
            notes = notes[notes[:, 0] < sec_stop]
            # Clip offsets at the slice stop time
            notes[:, 1] = np.minimum(notes[:, 1], sec_stop)

            # Offset the note intervals by the slice start time
            notes[:, 0] = notes[:, 0] - sec_start
            notes[:, 1] = notes[:, 1] - sec_start
            data['notes'] = notes

        # Convert all numpy arrays in the data dictionary to float32
        data = track_to_dtype(data, dtype='float32')

        return data

    def validate_track(self, data):
        """
        Get the features and ground truth for a track within a time interval.

        Parameters
        ----------
        data : dict
          Dictionary containing the features and ground-truth for a track

        Returns
        ----------
        valid : bool
          Whether or not the track data is valid
        """

        valid = True

        # TODO - validate each field in some way

        if 'notes' in data.keys():
            # Convert the dictionary representation into standard representation
            pitches, intervals = arr_to_note_groups(data['notes'])

            # Validate the intervals
            valid = valid and librosa.util.valid_intervals(intervals)

            # Validate the pitches - should be in Hz
            try:
                mir_eval.util.validate_frequencies(pitches, 5000, 20)
            except ValueError:
                valid = False

        return valid

    @abstractmethod
    def get_tracks(self, split):
        """
        Get the tracks associated with a dataset partition.

        Parameters
        ----------
        split : string
          Name of the partition from which to fetch tracks
        """

        return NotImplementedError

    @abstractmethod
    def load(self, track):
        """
        Get the ground truth for a track if it has already been saved. If the
        ground-truth does not exist yet, initialize a new dictionary to hold it.

        Parameters
        ----------
        track : string
          Name of the track to load

        Returns
        ----------
        data : dict
          Dictionary with ground-truth for the track
        """

        # Default data to None (not existing)
        data = None

        # Determine the expected path to the track's data
        gt_path = self.get_gt_dir(track)

        # Check if an entry for the data exists
        if os.path.exists(gt_path):
            # Load the ground-truth if it exists
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
        """
        Get the path for the ground-truth directory or a track's ground-truth.

        Parameters
        ----------
        track : string or None
          Optionally, append a track to the directory for the track's ground-truth path

        Returns
        ----------
        path : string
          Path to the ground-truth directory or a specific track's ground-truth.
        """

        # Get the path to the ground truth directory
        path = os.path.join(GEN_DATA_DIR, self.dataset_name(), 'gt')

        # Add the track name and the .npz extension if a track was provided
        if track is not None:
            path = os.path.join(path, track + '.npz')

        return path

    def get_feats_dir(self, track=None):
        """
        Get the path for the features directory or a track's features.

        Parameters
        ----------
        track : string or None
          Optionally, append a track to the directory for the track's features path

        Returns
        ----------
        path : string
          Path to the features directory or a specific track's features.
        """

        # Get the path to the features directory
        path = os.path.join(GEN_DATA_DIR, self.dataset_name(), self.data_proc.features_name())

        # Add the track name and the .npz extension if a track was provided
        if track is not None:
            path = os.path.join(path, track + '.npz')

        return path

    @staticmethod
    @abstractmethod
    def available_splits():
        """
        Get the supported partitions for the dataset.
        """

        return NotImplementedError

    @staticmethod
    @abstractmethod
    def dataset_name():
        """
        Get an appropriate name for the dataset.
        """

        return NotImplementedError

    @staticmethod
    @abstractmethod
    def download(save_dir):
        """
        Download the dataset to disk if possible.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the dataset
        """

        return NotImplementedError

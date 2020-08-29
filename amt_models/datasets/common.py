# My imports
from features.cqt import CQT

from tools.conversion import *
from tools.utils import *

# Regular imports
from torch.utils.data import Dataset
from abc import abstractmethod
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import shutil
import os

# TODO - MusicNet already implemented - add an easy ref and maybe some functions to make it compatible or redo
# TODO - implement functions for extending based on sustain pedal - any other bells or whistles
# TODO - ComboDataset


class TranscriptionDataset(Dataset):
    """
    Implements a generic music transcription dataset.
    """

    def __init__(self, base_dir, splits, hop_length, sample_rate, data_proc,
                 num_frames, split_notes, reset_data, store_data, save_data, seed):
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
        store_data : bool
          Flag to load data from memory or calculate each time instead of storing within RAM
        save_data : bool
          Flag to save data to memory after calculating once
        seed : int
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

        self.data_proc = data_proc
        # Default the feature extraction to a plain CQT if none was provided
        if self.data_proc is None:
            self.data_proc = CQT()

        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # This is redundant - but it's good to have in both places
        # Make sure there is agreement between dataset and feature module
        assert self.hop_length == self.data_proc.hop_length
        assert self.sample_rate == self.data_proc.sample_rate

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

        self.store_data = store_data
        self.save_data = save_data

        # Initialize a random number generator for the dataset
        self.rng = np.random.RandomState(seed)

        self.tracks = []
        # Aggregate all the track names from the selected splits
        for split in self.splits:
            self.tracks += self.get_tracks(split)

        # Load the ground-truth for each track into RAM
        if self.store_data:
            self.data = {}
            for track in tqdm(self.tracks):
                self.data[track] = self.load(track)

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

    def get_track_feats(self, data):
        if isinstance(data, dict):
            # Copy the track's data into a local dictionary
            data = deepcopy(data)
        else:
            # We assume a track name was given
            data = {'track' : data}

        track = data['track']

        # Determine the path to the track's features
        feats_path = self.get_feats_dir(track)

        # Check if the features already exist
        # TODO - may need to modify when filterbank learning is possible - actually, save_data should take care of it
        if os.path.exists(feats_path):
            # If so, load the features
            feats_dict = np.load(feats_path)
            feats = feats_dict['feats']
        else:
            # If not, calculate the features
            feats = self.data_proc.process_audio(data['audio'])

            if self.save_data:
                # Save the features to memory
                os.makedirs(os.path.dirname(feats_path), exist_ok=True)
                np.savez(feats_path, feats=feats)

        # It is faster to just calculate the frame times instead of loading
        times = self.data_proc.get_times(data['audio'])

        # Add the features to the data dictionary
        data['feats'] = feats
        data['times'] = times

        if self.store_data:
            # Add the features to the data dictionary in RAM
            self.data[track]['feats'] = feats
            self.data[track]['times'] = times

        return data

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

        if self.store_data:
            # Copy the track's ground-truth data into a local dictionary
            data = deepcopy(self.data[track_id])
        else:
            data = self.load(track_id)

        if 'feats' not in data.keys():
            data = self.get_track_feats(data)

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
            sample_start = self.rng.randint(0, len(data['audio']) - seq_length)

        # Determine the frames contained in this slice
        frame_start = sample_start // self.hop_length
        frame_end = frame_start + self.num_frames

        # Snap the sample_start to the nearest frame boundary if snap_to_frame is selected
        if snap_to_frame:
            sample_start = frame_start * self.hop_length

        # Calculate the last sample included in the slice
        sample_end = sample_start + seq_length

        # Slice the audio
        data['audio'] = data['audio'][..., sample_start : sample_end]

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
            # Determine the time in seconds of the boundary samples
            sec_start = sample_start / self.sample_rate
            sec_stop = sample_end / self.sample_rate

            # Remove notes with onsets before the slice start time
            notes = notes[notes[:, 0] > sec_start]
            # Remove notes with onsets after the slice stop time
            notes = notes[notes[:, 0] < sec_stop]
            # Clip offsets at the slice stop time
            notes[:, 1] = np.minimum(notes[:, 1], sec_stop)

            data['notes'] = notes

        # Determine which entries remain
        keys = list(data.keys())
        keys.remove('audio')
        keys.remove('notes')

        # Slice remaining entries
        for key in keys:
            if isinstance(data[key], np.ndarray):
                if key == 'times':
                    data[key] = data[key][..., frame_start : frame_end + 1]
                else:
                    data[key] = data[key][..., frame_start : frame_end]

        # Convert all numpy arrays in the data dictionary to float32
        data = track_to_dtype(data, dtype='float32')

        assert self.validate_track(data)

        return data

    @staticmethod
    def validate_track(data):
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

        keys = list(data.keys())

        if 'audio' in keys:
            valid = valid and (len(data['audio'].shape) == 1)

        if 'pitch' in keys:
            pitch = data['pitch']

            if not valid_activations(pitch):
                valid = False

        if 'feats' in keys:
            num_frames = data['feats'].shape[-1]

            # Make sure features have channel, num_feats, and num_frames dimension
            valid = valid and (len(data['feats'].shape) == 3)

            valid = valid and ('times' in keys)
            if 'times' in keys:
                valid = valid and (len(data['times'].shape) == 1)
                valid = valid and (data['times'].size == num_frames + 1)

        if 'notes' in keys:
            # Convert the dictionary representation into standard representation
            pitches, intervals = arr_to_note_groups(data['notes'])
            valid = valid and valid_notes(pitches, intervals)

        for key in keys:
            entry = data[key]
            if isinstance(entry, np.ndarray):
                valid = valid and (entry.dtype == 'float32')

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

    @classmethod
    def dataset_name(cls):
        """
        Retrieve an appropriate tag, the class name, for the dataset.

        Returns
        ----------
        tag : str
          Name of the child class calling the function
        """

        tag = cls.__name__

        return tag

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

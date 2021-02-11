# My imports
from amt_models.features import MelSpec

import amt_models.tools as tools

# Regular imports
from torch.utils.data import Dataset
from abc import abstractmethod
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import shutil
import os

# TODO - MusicNet
# TODO - Note splitting
# TODO - other bells/whistles in OF/TabCNN
# TODO - optionally download datasets in flac to save memory
# TODO - possible integration with mirdata


class TranscriptionDataset(Dataset):
    """
    Implements a generic music transcription dataset.
    """

    def __init__(self, base_dir, splits, hop_length, sample_rate, data_proc, profile,
                 num_frames, split_notes, reset_data, store_data, save_loc, seed):
        """
        Initialize parameters common to all datasets as fields and instantiate
        as a PyTorch Dataset.

        Parameters
        ----------
        base_dir : string
          Path to the directory containing the dataset, e.g. '~/Desktop/Dataset'
        splits : list of strings
          Names of tracks to include in this partition
        hop_length : int
          Number of samples between frames
        sample_rate : int or float
          Number of samples per second of audio
        data_proc : FeatureModel (features/common.py)
          Feature extraction model to use for the dataset
        profile : InstrumentProfile (tools/instrument.py)
          Instructions for organizing data and ground-truth
        num_frames : int
          Number of frames per data sample
        split_notes : bool
          Flag to avoiding cutting samples in between notes
        reset_data : bool
          Flag to reset extracted features and ground truth data if they already exists
        store_data : bool
          Flag to store data in RAM instead of loading ground truth and calculating features each time
        save_loc : string
          Doubles as:
            Flag to save data to memory after calculating once (always loaded afterwards)
            Location for saving and loading pre-organized ground-truth and calculated features
        seed : int
          The seed for random number generation
        """

        # Select a default base directory path if none was provided
        if base_dir is None:
            base_dir = os.path.join(tools.HOME, 'Desktop', 'Datasets', self.dataset_name())
        self.base_dir = base_dir

        # Check if the dataset exists in memory
        if not os.path.isdir(self.base_dir):
            # Attempt to download the dataset if it is missing and if a procedure exists
            self.download(self.base_dir)

        # Choose all available dataset splits if none were provided
        if splits is None:
            splits = self.available_splits()
        self.splits = splits

        # Keep track of hop length and sampling rate
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        # Default the feature extraction to a plain Mel Spectrogram if none was provided
        if data_proc is None:
            data_proc = MelSpec(hop_length=self.hop_length,
                                sample_rate=self.sample_rate)
        self.data_proc = data_proc

        # Default the instrument profile to a standard piano if none was provided
        if profile is None:
            profile = PianoProfile()
        self.profile = profile

        # Determine the number of samples per data point (seq_length)
        if num_frames is None:
            # Transcribe whole tracks at a time (all samples)
            self.seq_length = None
        else:
            # Take maximum number of samples which produce desired number of frames
            self.seq_length = max(self.data_proc.get_sample_range(num_frames))
        # Set the number of frames for each sample
        self.num_frames = num_frames

        # Set the storing and saving parameters
        self.store_data = store_data
        self.save_loc = save_loc

        self.reset_data = reset_data
        # Remove any saved ground-truth for the dataset if reset_data is selected
        if os.path.exists(self.get_gt_dir()) and self.reset_data:
            shutil.rmtree(self.get_gt_dir())
        # Make sure the directory for saving and loading ground-truth exists
        os.makedirs(self.get_gt_dir(), exist_ok=True)

        # Remove any saved features for the dataset if reset_data is selected
        if os.path.exists(self.get_feats_dir()) and self.reset_data:
            shutil.rmtree(self.get_feats_dir())
        # Make sure the directory for saving and loading features exists
        os.makedirs(self.get_feats_dir(), exist_ok=True)

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

    def calculate_feats(self, data):
        """
        Get the features for a track within the dataset.

        Parameters
        ----------
        data : dict or string
          if dict: Track data containing at least the track name,
          if str : ID of a track within dataset

        Returns
        ----------
        data : dict
          Dictionary containing the features and any pre-existing data for the chosen track
        """

        # Determine what type of argument was given
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
        if os.path.exists(feats_path):
            # If so, load the features
            feats_dict = np.load(feats_path, allow_pickle=True)
            feats = feats_dict['feats']
            feats = feats.item() if feats.size == 1 else feats

            # Load supporting hyper-parameters
            fs = feats_dict['fs'].item()
            hop_length = feats_dict['hop_length'].item()

        else:
            # If not, calculate the features
            feats = self.data_proc.process_audio(data['audio'])

            # Fetch the hyper-parameters of the feature module
            fs = self.data_proc.get_sample_rate()
            hop_length = self.data_proc.get_hop_length()

            if self.save_loc is not None:
                # Save the features to memory
                os.makedirs(os.path.dirname(feats_path), exist_ok=True)
                np.savez(feats_path, feats=feats, fs=fs, hop_length=hop_length)

        # Make sure there is agreement between dataset and features
        assert self.sample_rate == fs
        assert self.hop_length == hop_length

        # Calculate the frame times every time (faster than saving/loading)
        times = self.data_proc.get_times(data['audio'])

        # Check if fixed features were provided
        if feats is not None:
            # Add the features to the data dictionary
            data['feats'] = feats
        data['times'] = times

        if self.store_data:
            # Check if fixed features were provided
            if feats is not None:
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
            # Load the track's ground-truth
            data = self.load(track_id)

        # TODO - for now, this cannot be done, as it requires knowledge of original profile
        # # Determine which keys exist before calculating features
        # keys = list(data.keys())
        #
        # # Loop through possible ground-truth time frequency representations
        # for tfr_key in ['pitch', 'onsets', 'offsets']:
        #     # If the tfr exists in the ground-truth
        #     if tfr_key in keys:
        #         # Make sure it adheres to the chosen instrument profile
        #         data[tfr_key] = self.profile.to(data[tfr_key])

        if 'feats' not in data.keys():
            # Calculate the features and add to the dictionary
            data.update(self.calculate_feats(data))

        # Determine which keys exist after calculating features
        keys = list(data.keys())

        # Check to see if a specific sequence length was given
        if seq_length is None:
            # If not, and this Dataset object has a sequence length, use it
            if self.seq_length is not None:
                seq_length = self.seq_length
            # Otherwise, we assume the whole track is desired and perform no further actions
            else:
                # Convert all numpy arrays in the data dictionary to float32
                data = track_to_dtype(data, dtype='float32')
                # Validate all of the track data before returning
                assert self.validate_track(data, self.profile)
                return data

        # If a specific starting sample was not provided, sample one randomly
        if sample_start is None:
            sample_start = self.rng.randint(0, len(data['audio']) - seq_length)

        # Determine the frames contained in this slice
        frame_start = sample_start // self.hop_length
        frame_end = frame_start + self.num_frames

        if snap_to_frame:
            # Snap the sample_start to the left-most frame boundary
            sample_start = frame_start * self.hop_length

        # Calculate the last sample included in the slice
        sample_end = sample_start + seq_length

        # Slice the audio
        data['audio'] = data['audio'][..., sample_start : sample_end]

        # Determine if there is note ground-truth and, if so, get it
        if 'notes' in keys:
            # Notes entry has not been popped by a dataloader
            notes = data['notes']
        elif 'notes' in self.data[track_id].keys():
            # Notes entry has been popped and must be added again
            notes = self.data[track_id]['notes']
        else:
            # Notes entry never existed
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

        # Remove keys already processed
        keys.remove('audio')
        keys.remove('notes')
        keys.remove('fs')

        # Slice remaining entries (all assumed to be NumPy arrays)
        for key in keys:
            if isinstance(data[key], np.ndarray):
                if key == 'times':
                    # Leave in the extra entry to mark the ending time
                    data[key] = data[key][..., frame_start : frame_end + 1]
                else:
                    # Normal frame slicing protocol
                    data[key] = data[key][..., frame_start : frame_end]

        # TODO - repeat code of returning whole track above
        # Convert all numpy arrays in the data dictionary to float32
        data = track_to_dtype(data, dtype='float32')
        # Validate all of the track data before returning
        assert self.validate_track(data, self.profile)

        return data

    @staticmethod
    def validate_track(data, profile):
        """
        Make sure ground-truth and features adhere to standards.

        Parameters
        ----------
        data : dict
          Dictionary containing the features and ground-truth for a track
        profile : InstrumentProfile (tools/instrument.py)
          Instructions for organizing data and ground-truth

        Returns
        ----------
        valid : bool
          Whether or not the track data is valid
        """

        # This true has to pass through all checks
        valid = True

        # Get all of the entries in the track data
        keys = list(data.keys())

        if 'audio' in keys:
            # Audio should be 1-D
            valid = valid and (len(data['audio'].shape) == 1)

        if 'pitch' in keys:
            # Make sure the pitch data follows a supported format
            valid = valid and valid_activations(data['pitch'], profile)

        if 'feats' in keys:
            # Make sure features have channel, num_feats, and num_frames dimension (3-D)
            valid = valid and (len(data['feats'].shape) == 3)
            # Make sure there are times corresponding to features
            valid = valid and ('times' in keys)
            if 'times' in keys:
                # Times should be 1-D
                valid = valid and (len(data['times'].shape) == 1)
                # Dimensionality must agree (time has an extra entry)
                valid = valid and (data['times'].size == data['feats'].shape[-1] + 1)

        if 'notes' in keys:
            # Convert the presumed array into standard representation
            pitches, intervals = batched_notes_to_notes(data['notes'])
            # Make sure the notes are valid
            valid = valid and valid_notes(pitches, intervals)

        # Make sure all NumPy arrays consist of 32-bit floats
        for entry in data.values():
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
        Get the ground truth for a track. If it has already been saved, load it.
        If the ground-truth does not exist yet, initialize a new dictionary to hold it.

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

            # Make sure there is agreement between dataset and saved data
            assert self.sample_rate == data['fs'].item()

        if data is None:
            # Initialize a new dictionary if there is no saved data
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
          Append a track to the directory for the track's ground-truth path

        Returns
        ----------
        path : string
          Path to the ground-truth directory or a specific track's ground-truth
        """

        # Get the path to the ground truth directory
        path = os.path.join(self.save_loc, self.dataset_name(), 'gt')

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
          Append a track to the directory for the track's features path

        Returns
        ----------
        path : string
          Path to the features directory or a specific track's features
        """

        # Get the path to the features directory
        path = os.path.join(self.save_loc, self.dataset_name(), self.data_proc.features_name())

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
        Download the dataset to disk if possible. This is to be extended by a child class.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the dataset
        """

        # If the directory already exists, remove it
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)

        # Create the base directory
        os.makedirs(save_dir)

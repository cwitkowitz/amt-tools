# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from ..features import STFT
from .. import tools

# Regular imports
from torch.utils.data import Dataset
from abc import abstractmethod
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import warnings
import shutil
import os

# TODO - more datasets - MusicNet, Slakh, Drums
# TODO - optionally download datasets in flac to save memory
# TODO - possible integration with mirdata
# TODO - optionally avoud splitting notes...
#      - see magenta/models/onsets_frames_transcription/audio_label_data_utils.py
# TODO - multipitch can overlap on last/first frame of notes
#        - seems correct based on note start/end times but could be problematic

# TODO - get notes function?


class TranscriptionDataset(Dataset):
    """
    Implements a generic music transcription dataset.
    """

    def __init__(self, base_dir, splits, hop_length, sample_rate, data_proc, profile, num_frames,
                 audio_norm, split_notes, reset_data, store_data, save_data, save_loc, seed):
        """
        Initialize parameters common to all datasets as fields and instantiate
        as a PyTorch Dataset.

        Parameters
        ----------
        base_dir : string
          Path to the directory containing the dataset, e.g. '~/Desktop/Datasets/Dataset'
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
        audio_norm : float or None
          Type of normalization to perform when loading audio
          -1 - root-mean-square
          See librosa for others...
            - None case is handled here
        split_notes : bool
          Flag to avoiding cutting samples in between notes
        reset_data : bool
          Flag to reset extracted features and ground truth data if they already exists
        store_data : bool
          Flag to store data in RAM instead of loading ground truth and calculating features each time
        save_data : bool
          Flag to save ground-truth and features to memory after calculating once (always loaded afterwards)
        save_loc : string
          Location for saving and loading pre-organized ground-truth and calculated features
        seed : int
          The seed for random number generation
        """

        # Select a default base directory path if none was provided
        if base_dir is None:
            base_dir = os.path.join(tools.DEFAULT_DATASETS_DIR, self.dataset_name())
        self.base_dir = base_dir

        # Check if the dataset exists in memory
        if not os.path.isdir(self.base_dir):
            warnings.warn(f'Could not find dataset at specified path \'{self.base_dir}\''
                          '. Attempting to download...', category=RuntimeWarning)
            # Attempt to download the dataset if it is missing and if a procedure exists
            self.download(self.base_dir)

        # Choose all available dataset splits if none were provided
        if splits is None:
            splits = self.available_splits()
        self.splits = splits

        # Keep track of hop length and sampling rate
        self.hop_length = hop_length
        # TODO - None sampling rate -> fs extracted from load()?
        self.sample_rate = sample_rate

        # Default the feature extraction to a plain Mel Spectrogram if none was provided
        if data_proc is None:
            # TODO - should keep as None so that, e.g., pure tablature data won't do this step
            # TODO - or create placeholders such as Waveform() or Frames()
            data_proc = STFT(hop_length=self.hop_length,
                             sample_rate=self.sample_rate)
        self.data_proc = data_proc

        # Default the instrument profile to a standard piano if none was provided
        if profile is None:
            profile = tools.PianoProfile()
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

        self.audio_norm = audio_norm

        # Set the storing and saving parameters
        self.store_data = store_data
        self.save_data = save_data
        if save_loc is None:
            save_loc = tools.DEFAULT_FEATURES_GT_DIR
        self.save_loc = save_loc

        # TODO - shouldn't be any reason to save this
        self.reset_data = reset_data
        # Remove any saved ground-truth for the dataset if reset_data is selected
        if os.path.exists(self.get_gt_dir()) and self.reset_data:
            shutil.rmtree(self.get_gt_dir())
        if self.save_data:
            # Make sure the directory for saving and loading ground-truth exists
            os.makedirs(self.get_gt_dir(), exist_ok=True)

        # Remove any saved features for the dataset if reset_data is selected
        if os.path.exists(self.get_feats_dir()) and self.reset_data:
            shutil.rmtree(self.get_feats_dir())
        if self.save_data:
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

        # Convert all numpy arrays in the data dictionary to float32
        data = tools.dict_to_dtype(data, dtype=tools.FLOAT32)

        # Remove any notes, as they cannot be batched
        if tools.query_dict(data, tools.KEY_NOTES):
            data.pop(tools.KEY_NOTES)

        # Remove any pitch lists, as they cannot be batched
        if tools.query_dict(data, tools.KEY_PITCHLIST):
            data.pop(tools.KEY_PITCHLIST)

        # Remove sampling rate - it can cause problems if it is not an ndarray. Sample rate
        # should be able to be inferred from the dataset object, if no warnings are thrown
        if tools.query_dict(data, tools.KEY_FS):
            data.pop(tools.KEY_FS)

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
            data = {tools.KEY_TRACK : data}

        track = data[tools.KEY_TRACK]

        # Determine the path to the track's features
        feats_path = self.get_feats_dir(track)

        # Check if the features already exist
        if self.save_data and os.path.exists(feats_path):
            # If so, load the features
            feats_dict = tools.load_dict_npz(feats_path)
            feats = feats_dict[tools.KEY_FEATS]
            feats = feats.item() if feats.size == 1 else feats

            # Load supporting hyper-parameters
            fs = feats_dict[tools.KEY_FS].item()
            hop_length = feats_dict[tools.KEY_HOP].item()
        else:
            # If not, calculate the features
            feats = self.data_proc.process_audio(data[tools.KEY_AUDIO])

            # Fetch the hyper-parameters of the feature module
            fs = self.data_proc.get_sample_rate()
            hop_length = self.data_proc.get_hop_length()

            if self.save_data:
                # Get the appropriate path for saving the features
                os.makedirs(os.path.dirname(feats_path), exist_ok=True)
                # Save the features to memory as a NumPy zip file
                tools.save_dict_npz(feats_path, {tools.KEY_FS : fs,
                                                 tools.KEY_HOP : hop_length,
                                                 tools.KEY_FEATS : feats})

        # Make sure there is agreement between dataset and features
        if self.sample_rate != fs or self.hop_length != hop_length:
            warnings.warn('Loaded features\' sampling rate or hop length ' +
                          'differs from expected.', category=RuntimeWarning)

        if tools.query_dict(data, tools.KEY_TIMES):
            # Use the times that were already provided
            times = data[tools.KEY_TIMES]
        else:
            # Calculate the frame times (faster than saving/loading)
            times = self.data_proc.get_times(data[tools.KEY_AUDIO])
            # Add the times to the data dictionary
            data[tools.KEY_TIMES] = times

        # Check if fixed features were provided
        if feats is not None:
            # Add the features to the data dictionary
            data[tools.KEY_FEATS] = feats

        if self.store_data:
            # Check if fixed features were provided
            if feats is not None:
                # Add the features to the data dictionary in RAM
                self.data[track][tools.KEY_FEATS] = feats
                # TODO - a lot of memory could be saved by throwing
                #        away audio after features are computed
            self.data[track][tools.KEY_TIMES] = times

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

        if tools.KEY_FEATS not in data.keys():
            # Calculate the features and add to the dictionary
            data.update(self.calculate_feats(data))

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
            sample_start = self.rng.randint(0, len(data[tools.KEY_AUDIO]) - seq_length)

        # Determine the frames contained in this slice
        frame_start = sample_start // self.hop_length
        frame_end = frame_start + self.num_frames

        if snap_to_frame:
            # Snap the sample_start to the left-most frame boundary
            sample_start = frame_start * self.hop_length

        # Calculate the last sample included in the slice
        sample_end = sample_start + seq_length

        # Slice the audio
        data[tools.KEY_AUDIO] = data[tools.KEY_AUDIO][..., sample_start : sample_end]

        # Determine the time in seconds of the boundary samples
        sec_start = sample_start / self.sample_rate
        sec_stop = sample_end / self.sample_rate

        if tools.query_dict(data, tools.KEY_NOTES):
            if isinstance(data[tools.KEY_NOTES], dict):
                # TODO - assumes stack consists of standard note groups
                # Extract the stacked notes and convert them to batched representations
                temp_stacked_notes = tools.apply_func_stacked_representation(data[tools.KEY_NOTES],
                                                                             tools.notes_to_batched_notes)
                # Perform time slicing w.r.t. the batched notes along each slice of the stack
                temp_stacked_notes = tools.apply_func_stacked_representation(temp_stacked_notes,
                                                                             tools.slice_batched_notes,
                                                                             start_time=sec_start,
                                                                             stop_time=sec_stop)
                # Convert back to standard note groups and update the dictionary
                data[tools.KEY_NOTES] = tools.apply_func_stacked_representation(temp_stacked_notes,
                                                                                tools.batched_notes_to_notes)
            else:
                # Slice the ground-truth notes if they exist in the ground-truth
                data[tools.KEY_NOTES] = tools.slice_batched_notes(data[tools.KEY_NOTES], sec_start, sec_stop)

        if tools.query_dict(data, tools.KEY_PITCHLIST):
            if isinstance(data[tools.KEY_PITCHLIST], dict):
                # Slice ground-truth pitch list by slice if exists in the ground-truth
                data[tools.KEY_PITCHLIST] = tools.apply_func_stacked_representation(data[tools.KEY_PITCHLIST],
                                                                                    tools.slice_pitch_list,
                                                                                    start_time=sec_start,
                                                                                    stop_time=sec_stop)
            else:
                # Slice ground-truth pitch list if exists in the ground-truth
                data[tools.KEY_PITCHLIST] = tools.slice_pitch_list(*data[tools.KEY_PITCHLIST], sec_start, sec_stop)

        # Define list of entries to skip during slicing process
        skipped_keys = [tools.KEY_AUDIO, tools.KEY_FS, tools.KEY_NOTES, tools.KEY_PITCHLIST]
        # Slice the remaining dictionary entries
        data = tools.slice_track(data, frame_start, frame_end, skipped_keys)

        return data

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
        if self.save_data and os.path.exists(gt_path):
            # Load and unpack the data
            data = tools.load_dict_npz(gt_path)

            # Make sure there is agreement between dataset and saved data
            if self.sample_rate != data[tools.KEY_FS].item():
                warnings.warn('Loaded track\'s sampling rate differs from expected.', category=RuntimeWarning)

        if data is None:
            # Initialize a new dictionary if there is no saved data
            data = {}
        else:
            if tools.query_dict(data, tools.KEY_NOTES) and data[tools.KEY_NOTES].dtype == object:
                # Unpack the (stacked) notes (which will be in save-friendly format)
                data[tools.KEY_NOTES] = tools.unpack_stacked_representation(data[tools.KEY_NOTES])
            if tools.query_dict(data, tools.KEY_PITCHLIST) and data[tools.KEY_PITCHLIST].dtype == object:
                # TODO - assumes pitch list with type object is always a stacked representation
                # Unpack the (stacked) pitch list (which will be in save-friendly format)
                data[tools.KEY_PITCHLIST] = tools.unpack_stacked_representation(data[tools.KEY_PITCHLIST])

        # Add the track ID to the dictionary
        data[tools.KEY_TRACK] = track

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
        path = os.path.join(self.save_loc, self.dataset_name(), tools.GROUND_TRUTH_DIR)

        # Add the track name and the .npz extension if a track was provided
        if track is not None:
            path = os.path.join(path, f'{track}.{tools.NPZ_EXT}')

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
            path = os.path.join(path, f'{track}.{tools.NPZ_EXT}')

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

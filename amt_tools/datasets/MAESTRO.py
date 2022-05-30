# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import TranscriptionDataset
from .MAPS import MAPS
from .. import tools

# Regular imports
import pandas as pd
import os


class _MAESTRO(MAPS):
    """
    Implements either version of the MAESTRO piano transcription dataset
    (https://magenta.tensorflow.org/datasets/maestro).
    """

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=16000, data_proc=None,
                 profile=None, num_frames=None, audio_norm=-1, split_notes=False, reset_data=False,
                 store_data=False, save_data=True, save_loc=None, seed=0):
        """
        Initialize the dataset and establish parameter defaults in function signature.

        Parameters
        ----------
        See TranscriptionDataset class...
        """

        super().__init__(base_dir, splits, hop_length, sample_rate, data_proc, profile, num_frames,
                         audio_norm, split_notes, reset_data, store_data, save_data, save_loc, seed)

    def get_tracks(self, split):
        """
        Get the tracks associated with a dataset partition.

        Parameters
        ----------
        split : string
          Name of the partition from which to fetch tracks

        Returns
        ----------
        tracks : list of strings
          Names of tracks within the given partition
        """

        # Obtain the name of the csv file for this dataset version
        csv_file = [file for file in os.listdir(self.base_dir) if '.csv' in file][0]
        # Load all of the tabulated data from the csv file
        csv_data = pd.read_csv(os.path.join(self.base_dir, csv_file))

        # Obtain a list of the track splits
        associations = list(csv_data['split'])
        # Obtain a list of the track names (including the year directory)
        tracks = list(csv_data['audio_filename'])
        # Reconstruct the list of track names using only tracks contained in the provided split
        tracks = [tracks[i] for i in range(len(tracks)) if associations[i] == split]
        # Remove the file extensions from each track
        tracks = [os.path.splitext(track)[0] for track in tracks]
        # Sort all of the tracks alphabetically
        tracks.sort()

        return tracks

    def remove_overlapping(self, splits):
        """
        Remove any tracks contained in the given splits from
        the initial dataset partition, should they exist.

        Parameters
        ----------
        splits : list of strings
          Splits to check for repeat tracks
        """

        # TODO
        return NotImplementedError

    def get_track_dir(self, track):
        """
        Parent class MAPS has and uses this method. It is overridden here
        to squash the MAPS functionality if the function were called from
        a MAESTRO instantiation - which there is no need to do anyway.

        Parameters
        ----------
        track : string
          MAESRO track name
        """

        return NotImplementedError

    def get_wav_path(self, track):
        """
        Get the path to the audio of a track.

        Parameters
        ----------
        track : string
          MAESTRO track name

        Returns
        ----------
        wav_path : string
          Path to the audio of the specified track
        """

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, f'{track}.{tools.WAV_EXT}')

        return wav_path

    def get_midi_path(self, track):
        """
        Get the path to the annotations of a track.

        Parameters
        ----------
        track : string
          MAESTRO track name

        Returns
        ----------
        midi_path : string
          Path to the MIDI file of the specified track
        """

        # Get the path to the annotations
        midi_path = os.path.join(self.base_dir, f'{track}.{tools.MIDI_EXT}')

        return midi_path

    @staticmethod
    def available_splits():
        """
        Obtain a list of possible splits. Currently, the splits are train/validation/test.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset for different stages of training
        """

        splits = ['train', 'validation', 'test']

        return splits

    @staticmethod
    def download(save_dir):
        """
        This is to be extended by a child class.

        Parameters
        ----------
        save_dir : string
          Directory in which to save the contents of MAESTRO
        """

        TranscriptionDataset.download(save_dir)


# TODO - is it possible to just download the csv file and work with V2?
class MAESTRO_V1(_MAESTRO):
    """
    MAESTRO version 1.
    """

    def __init__(self, **kwargs):
        """
        Call the parent constructor.
        """

        super().__init__(**kwargs)

    @staticmethod
    def download(save_dir):
        """
        Download MAESTRO version 1 to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory in which to save the contents of MAESTRO
        """

        # Reset the directory if it already exists
        _MAESTRO.download(save_dir)

        print(f'Downloading {MAESTRO_V1.dataset_name()}')

        # URL pointing to the zip file
        url = f'https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0.zip'

        # Construct a path for saving the file
        save_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        tools.stream_url_resource(url, save_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        tools.unzip_and_remove(save_path)

        # Construct a path to the out-of-the-box top-level directory
        old_dir = os.path.join(save_dir, 'maestro-v1.0.0/')

        # Remove the out-of-the-box top-level directory from the path chain
        tools.change_base_dir(save_dir, old_dir)


class MAESTRO_V2(_MAESTRO):
    """
    MAESTRO version 2.
    """

    def __init__(self, **kwargs):
        """
        Call the parent constructor.
        """

        super().__init__(**kwargs)

    @staticmethod
    def download(save_dir):
        """
        Download MAESTRO version 2 to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory in which to save the contents of MAESTRO
        """

        # Reset the directory if it already exists
        _MAESTRO.download(save_dir)

        print(f'Downloading {MAESTRO_V2.dataset_name()}')

        # URL pointing to the zip file
        url = f'https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip'

        # Construct a path for saving the file
        save_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        tools.stream_url_resource(url, save_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        tools.unzip_and_remove(save_path)

        # Construct a path to the out-of-the-box top-level directory
        old_dir = os.path.join(save_dir, 'maestro-v2.0.0/')

        # Remove the out-of-the-box top-level directory from the path chain
        tools.change_base_dir(save_dir, old_dir)


class MAESTRO_V3(_MAESTRO):
    """
    MAESTRO version 3.
    """

    def __init__(self, **kwargs):
        """
        Call the parent constructor.
        """

        super().__init__(**kwargs)

    @staticmethod
    def download(save_dir):
        """
        Download MAESTRO version 3 to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory in which to save the contents of MAESTRO
        """

        # Reset the directory if it already exists
        _MAESTRO.download(save_dir)

        print(f'Downloading {MAESTRO_V3.dataset_name()}')

        # URL pointing to the zip file
        url = f'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip'

        # Construct a path for saving the file
        save_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        tools.stream_url_resource(url, save_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        tools.unzip_and_remove(save_path)

        # Construct a path to the out-of-the-box top-level directory
        old_dir = os.path.join(save_dir, 'maestro-v3.0.0/')

        # Remove the out-of-the-box top-level directory from the path chain
        tools.change_base_dir(save_dir, old_dir)

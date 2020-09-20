# My imports
from datasets.common import TranscriptionDataset

from tools.instrument import *
from tools.conversion import *
from tools.io import *

# Regular imports
from tqdm import tqdm

import pandas as pd
import numpy as np
import requests
import zipfile
import shutil
import os


class _MAESTRO(TranscriptionDataset):
    """
    Implements either version of the MAESTRO piano transcription dataset
    (https://magenta.tensorflow.org/datasets/maestro).
    """

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=16000, data_proc=None, profile=None,
                 num_frames=None, split_notes=False, reset_data=False, store_data=False, save_data=True, seed=0):
        """
        Initialize the dataset and establish parameter defaults in function signature.

        Parameters
        ----------
        See TranscriptionDataset class...
        """

        super().__init__(base_dir, splits, hop_length, sample_rate, data_proc, profile,
                         num_frames, split_notes, reset_data, store_data, save_data, seed)

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

    def load(self, track):
        """
        Load the ground-truth from memory or generate it from scratch.

        Parameters
        ----------
        track : string
          Name of the track to load, including the year directory

        Returns
        ----------
        data : dict
          Dictionary with ground-truth for the track
        """

        # Load the track data if it exists in memory, otherwise instantiate track data
        data = super().load(track)

        # If the track data is being instantiated, it will not have the 'audio' key
        if 'audio' not in data.keys():
            # Construct the path to the track's audio
            wav_path = os.path.join(self.base_dir, track + '.wav')
            # Load and normalize the audio
            audio, fs = load_audio(wav_path, self.sample_rate)
            # Add the audio and sampling rate to the track data
            data['audio'], data['fs'] = audio, fs

            # Construct the path to the track's MIDI data
            midi_path = os.path.join(self.base_dir, track + '.midi')
            # Load the notes from the MIDI data and remove the velocity
            notes = load_midi_notes(midi_path)[:, :-1]
            # Convert the note lists to a note array
            pitches, intervals = arr_to_note_groups(notes)

            # We need the frame times to convert from notes to frames
            times = self.data_proc.get_times(data['audio'])

            # Check which instrument profile is used
            if isinstance(self.profile, PianoProfile):
                # Decode the notes into pianoroll to obtain the frame-wise pitches
                pitch = midi_groups_to_pianoroll(pitches, intervals, times, self.profile.get_midi_range())
            else:
                raise AssertionError('Provided InstrumentProfile not supported...')

            # Add the frame-wise pitches to the track data
            data['pitch'] = pitch

            # Convert the note pitches to hertz
            notes[:, -1] = librosa.midi_to_hz(notes[:, -1])
            # Add the note array to the track data
            data['notes'] = notes

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_dir(track)
                # Create the year directory if it doesn't exist
                os.makedirs(os.path.dirname(gt_path), exist_ok=True)
                # Save the audio, sampling rate, frame-wise pitches, and notes
                np.savez(gt_path,
                         fs=fs,
                         audio=audio,
                         pitch=pitch,
                         notes=notes)

        return data

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
        pass

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
        This is to be overridden by a child class.
        Resets the target directory if it already exists.

        Parameters
        ----------
        save_dir : string
          Directory in which to save the contents of MAESTRO
        """

        # If the directory already exists, remove it
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)

        # Create the base directory
        os.mkdir(save_dir)

def stream_and_unzip(url, save_dir, chunk_size=1024):
    """
    Download a zip file at a URL by streaming it, unzip it, and remove the zip file.

    Parameters
    ----------
    url : string
      URL pointing to the zip file
    save_dir : string
      Directory to use when saving and extracting the zip file
    chunk_size : int
      Number of bytes to download at a time
    """

    # Construct a path for saving the zip file
    save_path = os.path.join(save_dir, os.path.basename(url))

    # Create an HTTP GET request
    r = requests.get(url, stream=True)

    # Determine the total number of bytes to be downloaded
    total_length = int(r.headers.get('content-length'))

    # Open the target zip file in write mode
    with open(save_path, 'wb') as zip_file:
        # Iteratively download chunks of the zip file,
        # displaying a progress bar in the console
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size),
                               total=int(total_length/chunk_size+1)):
            # If a chunk was successfully downloaded,
            if chunk:
                # Write the chunk to the zip file
                zip_file.write(chunk)

    # Open the zip file in read mode
    with zipfile.ZipFile(save_path, 'r') as zip_ref:
        # Extract the contents of the zip file in the same directory
        zip_ref.extractall(save_dir)

    # Delete the zip file
    shutil.rmtree(save_path)


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
        Download MAESTRO version 1.

        Parameters
        ----------
        save_dir : string
          Directory in which to save the contents of MAESTRO
        """

        # Reset the directory if it already exists
        _MAESTRO.download(save_dir)

        # URL pointing to the zip file
        url = f'https://storage.googleapis.com/magentadata/datasets/maestro/v1.0.0/maestro-v1.0.0.zip'

        # Download the zip file, unzip it, and remove it
        stream_and_unzip(url, save_dir, 1000 * 1024)


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
        Download MAESTRO version 1.

        Parameters
        ----------
        save_dir : string
          Directory in which to save the contents of MAESTRO
        """

        # Reset the directory if it already exists
        _MAESTRO.download(save_dir)

        # URL pointing to the zip file
        url = f'https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip'

        # Download the zip file, unzip it, and remove it
        stream_and_unzip(url, save_dir, 1000 * 1024)

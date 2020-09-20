# My imports
from datasets.common import TranscriptionDataset

from tools.instrument import *
from tools.conversion import *
from tools.io import *

# Regular imports
import numpy as np
import mirdata
import shutil
import os


class GuitarSet(TranscriptionDataset):
    """
    Implements the GuitarSet guitar transcription dataset (https://guitarset.weebly.com/).
    """

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=44100, data_proc=None, profile=None,
                 num_frames=None, split_notes=False, reset_data=False, store_data=True, save_data=True, seed=0):
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

        # Construct a path to GuitarSet's JAMS directory
        jams_dir = os.path.join(self.base_dir, 'annotation')
        # Extract the names of all the files in the directory
        jams_paths = os.listdir(jams_dir)
        # Sort all of the tracks alphabetically
        jams_paths.sort()

        # Remove the JAMS file extension from the file names
        tracks = [os.path.splitext(path)[0] for path in jams_paths]

        # Determine where the split starts within the sorted tracks
        split_start = int(split) * 60
        # Slice the appropriate tracks
        tracks = tracks[split_start : split_start + 60]

        return tracks

    def load(self, track):
        """
        Load the ground-truth from memory or generate it from scratch.

        Parameters
        ----------
        track : string
          Name of the track to load

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
            wav_path = os.path.join(self.base_dir, 'audio_mono-mic', track + '_mic.wav')
            # Load and normalize the audio
            audio, fs = load_audio(wav_path, self.sample_rate)
            # Add the audio and sampling rate to the track data
            data['audio'], data['fs'] = audio, fs

            # Construct the path to the track's JAMS data
            jams_path = os.path.join(self.base_dir, 'annotation', track + '.jams')
            # Extract the notes from the track's JAMS file
            pitches, intervals = load_jams_guitar_notes(jams_path)
            # Convert the note lists to a note array
            notes = note_groups_to_arr(pitches, intervals)
            # Add the note array to the track data
            data['notes'] = notes

            # We need the frame times to convert from notes to frames
            times = self.data_proc.get_times(data['audio'])

            # Check which instrument profile is used
            if isinstance(self.profile, GuitarProfile):
                # Load the frame-wise pitches as tablature from the track's JAMS file
                pitch = load_jams_guitar_tabs(jams_path, times, self.profile.tuning)
            elif isinstance(self.profile, PianoProfile):
                # Convert the note pitches to MIDI
                pitches = librosa.hz_to_midi(pitches)
                # Decode the notes into pianoroll to obtain the frame-wise pitches
                pitch = midi_groups_to_pianoroll(pitches, intervals, times, self.profile.get_midi_range())
            else:
                raise AssertionError('Provided InstrumentProfile not supported...')

            # Add the frame-wise pitches to the track data
            data['pitch'] = pitch

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_dir(track)
                # Save the audio, sampling rate, frame-wise pitches, and notes
                np.savez(gt_path,
                         fs=fs,
                         audio=audio,
                         pitch=pitch,
                         notes=notes)

        return data

    @staticmethod
    def available_splits():
        """
        Obtain a list of possible splits. Currently, the splits are by player,
        but we could equally do genre, key, etc., as long as get_tracks() is adapted.

        Returns
        ----------
        splits : list of strings
          Player codes listed at beginning of file names
        """

        splits = ['00', '01', '02', '03', '04', '05']

        return splits

    @staticmethod
    def download(save_dir):
        """
        Download GuitarSet to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory in which to save the contents of GuitarSet
        """

        TranscriptionDataset.download(save_dir)

        print(f'Downloading {GuitarSet.dataset_name()}')

        # TODO - can download directly from https://zenodo.org/record/3371780#.X2dWA3VKgk8
        # Download GuitarSet
        mirdata.guitarset.download(data_home=save_dir)

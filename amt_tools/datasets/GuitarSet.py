# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import TranscriptionDataset
from .. import tools

# Regular imports
from mirdata.datasets import guitarset

import numpy as np
import os


class GuitarSet(TranscriptionDataset):
    """
    Implements the GuitarSet guitar transcription dataset (https://guitarset.weebly.com/).
    """

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=44100, data_proc=None,
                 profile=None, num_frames=None, split_notes=False, reset_data=False, store_data=True,
                 save_data=True, save_loc=None, seed=0):
        """
        Initialize the dataset and establish parameter defaults in function signature.

        Parameters
        ----------
        See TranscriptionDataset class...
        """

        super().__init__(base_dir, splits, hop_length, sample_rate, data_proc, profile,
                         num_frames, split_notes, reset_data, store_data, save_data, save_loc, seed)

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
        if tools.KEY_AUDIO not in data.keys():
            # Construct the path to the track's audio
            wav_path = self.get_wav_path(track)
            # Load and normalize the audio along with the sampling rate
            data[tools.KEY_AUDIO], data[tools.KEY_FS] = tools.load_normalize_audio(wav_path, self.sample_rate)

            # Construct the path to the track's JAMS data
            jams_path = self.get_jams_path(track)

            # Load the notes by string from the JAMS file
            stacked_notes = tools.load_stacked_notes_jams(jams_path)

            # We need the frame times for the tablature
            times = self.data_proc.get_times(data[tools.KEY_AUDIO])

            # Convert the string-wise notes into a stacked multi pitch array
            stacked_multi_pitch = tools.stacked_notes_to_stacked_multi_pitch(stacked_notes, times, self.profile)

            # Convert the stacked multi pitch array into a single representation
            data[tools.KEY_MULTIPITCH] = tools.stacked_multi_pitch_to_multi_pitch(stacked_multi_pitch)

            # Convert the stacked multi pitch array into tablature
            data[tools.KEY_TABLATURE] = tools.stacked_multi_pitch_to_tablature(stacked_multi_pitch, self.profile)

            # Consider the length of a hop as the ambiguity for onsets/offsets
            ambiguity = self.hop_length / self.sample_rate

            # Obtain note onsets from the notes in tablature format
            stacked_onsets = tools.stacked_notes_to_stacked_onsets(stacked_notes, times, self.profile, ambiguity)
            data[tools.KEY_ONSETS] = tools.stacked_multi_pitch_to_tablature(stacked_onsets, self.profile)

            # Obtain note offsets from the notes in tablature format
            stacked_offsets = tools.stacked_notes_to_stacked_offsets(stacked_notes, times, self.profile, ambiguity)
            data[tools.KEY_OFFSETS] = tools.stacked_multi_pitch_to_tablature(stacked_offsets, self.profile)

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_dir(track)

                # Save the data as a NumPy zip file
                keys = (tools.KEY_FS, tools.KEY_AUDIO,
                        tools.KEY_TABLATURE, tools.KEY_MULTIPITCH,
                        tools.KEY_ONSETS, tools.KEY_OFFSETS)
                tools.save_pack_npz(gt_path, keys, data[tools.KEY_FS], data[tools.KEY_AUDIO],
                                    data[tools.KEY_TABLATURE], data[tools.KEY_MULTIPITCH],
                                    data[tools.KEY_ONSETS], data[tools.KEY_OFFSETS])

        return data

    def get_wav_path(self, track):
        """
        Get the path to the audio of a track.

        Parameters
        ----------
        track : string
          GuitarSet track name

        Returns
        ----------
        wav_path : string
          Path to the audio of the specified track
        """

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, 'audio_mono-mic', track + '_mic.' + tools.WAV_EXT)

        return wav_path

    def get_jams_path(self, track):
        """
        Get the path to the annotations of a track.

        Parameters
        ----------
        track : string
          GuitarSet track name

        Returns
        ----------
        jams_path : string
          Path to the JAMS file of the specified track
        """

        # Get the path to the annotations
        jams_path = os.path.join(self.base_dir, 'annotation', f'{track}.{tools.JAMS_EXT}')

        return jams_path

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
        guitarset.Dataset(data_home=save_dir).download(force_overwrite=True, cleanup=True)

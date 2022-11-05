# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import TranscriptionDataset
from .. import tools

# Regular imports
import os


class MAPS(TranscriptionDataset):
    """
    Implements the MAPS piano transcription dataset
    (https://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/
    maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/).
    """

    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=16000, data_proc=None,
                 profile=None, num_frames=None, audio_norm=-1, split_notes=False, reset_data=False,
                 store_data=True, save_data=True, save_loc=None, seed=0):
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

        # Construct a path to the MAPS music piece directory
        split_dir = os.path.join(self.base_dir, split, 'MUS')
        # Extract the names of all the files in the directory
        split_paths = os.listdir(split_dir)

        # Remove the extensions (text, midi, audio), leaving three repeats per track
        tracks = [os.path.splitext(path)[0] for path in split_paths]
        # Collapse repeats by adding the extensionless file names to a set
        tracks = list(set(tracks))
        # Sort all of the tracks alphabetically
        tracks.sort()

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
        if not tools.query_dict(data, tools.KEY_AUDIO):
            # Construct the path to the track's audio
            wav_path = self.get_wav_path(track)
            # Load and normalize the audio along with the sampling rate
            audio, fs = tools.load_normalize_audio(wav_path,
                                                   fs=self.sample_rate,
                                                   norm=self.audio_norm)

            # We need the frame times for the multi pitch array
            times = self.data_proc.get_times(audio)

            # Construct the path to the track's MIDI data
            midi_path = self.get_midi_path(track)

            # Load the batch-friendly notes from the MIDI data and remove the velocity
            batched_notes = tools.load_notes_midi(midi_path)[..., :-1]

            # Convert the batch-friendly notes to notes
            pitches, intervals = tools.batched_notes_to_notes(batched_notes)

            # Represent the string-wise notes as a multi pitch array
            multi_pitch = tools.notes_to_multi_pitch(pitches, intervals, times, self.profile)

            # Consider the length of a hop as the ambiguity for onsets/offsets
            ambiguity = self.hop_length / self.sample_rate

            # Obtain onsets and offsets from the notes as stacked multi pitch arrays
            onsets = tools.notes_to_onsets(pitches, intervals, times, self.profile, ambiguity)
            offsets = tools.notes_to_offsets(pitches, intervals, times, self.profile, ambiguity)

            # Add all relevant ground-truth to the dictionary
            data.update({tools.KEY_FS : fs,
                         tools.KEY_AUDIO : audio,
                         tools.KEY_MULTIPITCH : multi_pitch,
                         tools.KEY_ONSETS : onsets,
                         tools.KEY_OFFSETS : offsets,
                         tools.KEY_NOTES : batched_notes})

            if self.save_data:
                # Get the appropriate path for saving the track data
                gt_path = self.get_gt_dir(track)

                # Create the (sub-directory) path if it doesn't exist
                os.makedirs(os.path.dirname(gt_path), exist_ok=True)

                # Save the data as a NumPy zip file
                tools.save_dict_npz(gt_path, data)

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

        tracks = []
        # Aggregate all the track names from the selected splits
        for split in splits:
            tracks += self.get_tracks(split)

        # Remove the piano from each track name
        tracks = ['_'.join(t.split('_')[:-1]) for t in tracks]
        # Rebuild the internal list of tracks with non-intersecting tracks
        self.tracks = [t for t in self.tracks if '_'.join(t.split('_')[:-1]) not in tracks]

        if self.store_data:
            # Loop through all internal ground-truth entries
            for key in list(self.data.keys()):
                # If the corresponding track entry no longer
                # exists, remove the ground-truth entry as well
                if key not in self.tracks:
                    self.data.pop(key)

    def get_track_dir(self, track):
        """
        Get the parent directory (piano) where the track is located.

        Parameters
        ----------
        track : string
          MAPS track name

        Returns
        ----------
        track_dir : string
          Path to the parent directory of the specified track
        """

        # Determine the piano used for the track (last part of track name)
        piano = track.split('_')[-1]
        # Construct a path to the directory containing pieces played on the piano
        track_dir = os.path.join(self.base_dir, piano, 'MUS')

        return track_dir

    def get_wav_path(self, track):
        """
        Get the path to the audio of a track.

        Parameters
        ----------
        track : string
          MAPS track name

        Returns
        ----------
        wav_path : string
          Path to the audio of the specified track
        """

        # Get the path to the audio
        wav_path = os.path.join(self.get_track_dir(track), f'{track}.{tools.WAV_EXT}')

        return wav_path

    def get_midi_path(self, track):
        """
        Get the path to the annotations of a track.

        Parameters
        ----------
        track : string
          MAPS track name

        Returns
        ----------
        midi_path : string
          Path to the MIDI file of the specified track
        """

        # Get the path to the annotations
        midi_path = os.path.join(self.get_track_dir(track), f'{track}.{tools.MID_EXT}')

        return midi_path

    @staticmethod
    def available_splits():
        """
        Obtain a list of possible splits. Currently, the splits are by
        piano, in accordance with the default organization of the dataset.

        Returns
        ----------
        splits : list of strings
          Names of pianos used in MAPS
        """

        splits = ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD',
                  'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl',
                  'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

        return splits

    @staticmethod
    def download(save_dir):
        """
        Currently, this function stops execution. I am not aware of a way to
        automatically download MAPS. I will consider this again some time in
        the future.

        Parameters
        ----------
        save_dir : string
          Directory in which to save the contents of MAPS
        """

        # TODO - add link

        assert False, 'MAPS must be requested and downloaded manually'

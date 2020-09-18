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
    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=44100, data_proc=None, profile=None,
                 num_frames=None, split_notes=False, reset_data=False, store_data=True, save_data=True, seed=0):
        super().__init__(base_dir, splits, hop_length, sample_rate, data_proc, profile,
                         num_frames, split_notes, reset_data, store_data, save_data, seed)

    def get_tracks(self, split):
        jams_dir = os.path.join(self.base_dir, 'annotation')
        jams_paths = os.listdir(jams_dir)
        jams_paths.sort()

        tracks = [os.path.splitext(path)[0] for path in jams_paths]

        split_start = int(split) * 60

        tracks = tracks[split_start : split_start + 60]

        return tracks

    def load(self, track):
        data = super().load(track)

        if 'audio' not in data.keys():
            wav_path = os.path.join(self.base_dir, 'audio_mono-mic', track + '_mic.wav')
            audio, fs = load_audio(wav_path, self.sample_rate)
            data['audio'], data['fs'] = audio, fs

            times = self.data_proc.get_times(data['audio'])

            jams_path = os.path.join(self.base_dir, 'annotation', track + '.jams')
            if isinstance(self.profile, GuitarProfile):
                # TODO - add format function to profile to call to_x?
                pitch = load_jams_guitar_tabs(jams_path, times, self.profile.tuning)
            elif isinstance(self.profile, PianoProfile):
                pitches, intervals = load_jams_guitar_notes(jams_path, hz=False)
                pitch = midi_groups_to_pianoroll(pitches, intervals, times, self.profile.get_midi_range())
            else:
                raise AssertionError('Provided InstrumentProfile not supported...')
            data['pitch'] = pitch

            pitches, intervals = load_jams_guitar_notes(jams_path)
            notes = note_groups_to_arr(pitches, intervals)
            data['notes'] = notes

            if self.save_data:
                gt_path = self.get_gt_dir(track)
                np.savez(gt_path,
                         fs=fs,
                         audio=audio,
                         pitch=pitch,
                         notes=notes)

        return data

    @staticmethod
    def available_splits():
        return ['00', '01', '02', '03', '04', '05']

    @staticmethod
    def download(save_dir):
        # If the directory already exists, remove it
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)

        # Create the base directory
        os.mkdir(save_dir)

        print(f'Downloading {GuitarSet.dataset_name()}')

        # Download GuitarSet
        # TODO - mirdata might be overkill if I don't use its load function
        # TODO - "flac" option which download flac instead
        mirdata.guitarset.download(data_home=save_dir)

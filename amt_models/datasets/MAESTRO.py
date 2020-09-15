# My imports
from datasets.common import TranscriptionDataset

from tools.instrument import *
from tools.conversion import *
from tools.io import *

# Regular imports
from pretty_midi import *

import pandas as pd
import numpy as np
import os


class MAESTRO_V1(TranscriptionDataset):
    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=16000, data_proc=None, profile=None,
                 num_frames=None, split_notes=False, reset_data=False, store_data=False, save_data=True, seed=0):
        super().__init__(base_dir, splits, hop_length, sample_rate, data_proc, profile,
                         num_frames, split_notes, reset_data, store_data, save_data, seed)

    def get_tracks(self, split):
        csv_data = pd.read_csv(os.path.join(self.base_dir, 'maestro-v1.0.0.csv'))

        associations = list(csv_data['split'])

        tracks = list(csv_data['audio_filename'])
        tracks = [tracks[i] for i in range(len(tracks)) if associations[i] == split]
        tracks = [os.path.splitext(track)[0] for track in tracks]
        tracks.sort()

        return tracks

    def load(self, track):
        data = super().load(track)

        if 'audio' not in data.keys():
            # TODO - clean this up significantly and implement sustain
            wav_path = os.path.join(self.base_dir, track + '.wav')
            audio, fs = load_audio(wav_path, self.sample_rate)
            data['audio'], data['fs'] = audio, fs

            mid_path = os.path.join(self.base_dir, track + '.midi')
            mid_data = PrettyMIDI(mid_path)

            notes = np.array([[], [], []]).T
            for note in mid_data.instruments[0].notes:
                notes = np.append(notes, np.array([[note.start], [note.end], [note.pitch]]).T, axis=0)

            pitches, intervals = arr_to_note_groups(notes)

            times = self.data_proc.get_times(data['audio'])
            if isinstance(self.profile, PianoProfile):
                pitch = midi_groups_to_pianoroll(pitches, intervals, times, self.profile.get_midi_range())
            else:
                raise AssertionError('Provided InstrumentProfile not supported...')
            data['pitch'] = pitch

            notes[:, -1] = librosa.midi_to_hz(notes[:, -1])
            data['notes'] = notes

            # TODO - bring this out to common?
            if self.save_data:
                gt_path = self.get_gt_dir(track)
                # TODO - get rid of the base dir or do for all to stay consistent
                os.makedirs(os.path.dirname(gt_path), exist_ok=True)
                np.savez(gt_path,
                         fs=fs,
                         audio=audio,
                         pitch=pitch,
                         notes=notes)

        return data

    @staticmethod
    def available_splits():
        return ['train', 'validation', 'test']

    @staticmethod
    def download(save_dir):
        # TODO
        return NotImplementedError

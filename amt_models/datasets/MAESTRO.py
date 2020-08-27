# My imports
from datasets.common import TranscriptionDataset

from tools.conversion import *
from tools.io import *

# Regular imports
from mir_eval.io import load_valued_intervals
from pretty_midi import *

import pandas as pd
import numpy as np
import os


class MAESTRO_V1(TranscriptionDataset):
    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=16000, data_proc=None,
                 num_frames=None, split_notes=False, reset_data=False, store_data=False, save_data=True, seed=0):
        super().__init__(base_dir, splits, hop_length, sample_rate, data_proc,
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
            audio, _ = load_audio(wav_path, self.sample_rate)
            data['audio'] = audio

            mid_path = os.path.join(self.base_dir, track + '.midi')
            mid_data = PrettyMIDI(mid_path)

            notes = np.array([[], [], []]).T
            for note in mid_data.instruments[0].notes:
                notes = np.append(notes, np.array([[note.start], [note.end], [note.pitch]]).T, axis=0)

            pitches, intervals = arr_to_note_groups(notes)

            times = self.data_proc.get_times(data['audio'])
            pianoroll = midi_groups_to_pianoroll(pitches, intervals, times, PIANO_RANGE)
            data['pianoroll'] = pianoroll

            notes[:, -1] = librosa.midi_to_hz(notes[:, -1])
            data['notes'] = notes

            onsets = get_pianoroll_onsets(pianoroll)
            data['onsets'] = onsets

            # TODO - bring this out to common?
            if self.save_data:
                gt_path = self.get_gt_dir(track)
                # TODO - get rid of the base dir or do for all to stay consistent
                os.makedirs(os.path.dirname(gt_path), exist_ok=True)
                np.savez(gt_path,
                         audio=audio,
                         pianoroll=pianoroll,
                         onsets=onsets,
                         notes=notes)

        return data

    @staticmethod
    def available_splits():
        return ['train', 'validation', 'test']

    @staticmethod
    def dataset_name():
        return 'MAESTRO_V1'

    @staticmethod
    def download(save_dir):
        return NotImplementedError

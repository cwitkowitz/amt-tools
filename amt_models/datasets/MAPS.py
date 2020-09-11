# My imports
from datasets.common import TranscriptionDataset

from tools.instrument import *
from tools.conversion import *
from tools.io import *

# Regular imports
from mir_eval.io import load_valued_intervals

import numpy as np
import os


class MAPS(TranscriptionDataset):
    def __init__(self, base_dir=None, splits=None, hop_length=512, sample_rate=16000, data_proc=None, profile=None,
                 num_frames=None, split_notes=False, reset_data=False, store_data=True, save_data=True, seed=0):
        super().__init__(base_dir, splits, hop_length, sample_rate, data_proc, profile,
                         num_frames, split_notes, reset_data, store_data, save_data, seed)

    def remove_overlapping(self, splits):
        # Initialize list of tracks to remove
        tracks = []

        for split in splits:
            tracks += self.get_tracks(split)

        tracks = ['_'.join(t.split('_')[:-1]) for t in tracks]

        self.tracks = [t for t in self.tracks if '_'.join(t.split('_')[:-1]) not in tracks]

        for key in list(self.data.keys()):
            if key not in self.tracks:
                self.data.pop(key)

    def get_tracks(self, split):
        split_dir = os.path.join(self.base_dir, split, 'MUS')
        split_paths = os.listdir(split_dir)

        # Remove extensions
        tracks = [os.path.splitext(path)[0] for path in split_paths]
        # Collapse repeat names
        tracks = list(set(tracks))
        tracks.sort()

        return tracks

    def load(self, track):
        data = super().load(track)

        if 'audio' not in data.keys():
            # TODO - implement sustain
            piano = track.split('_')[-1]
            track_dir = os.path.join(self.base_dir, piano, 'MUS')

            wav_path = os.path.join(track_dir, track + '.wav')
            audio, _ = load_audio(wav_path, self.sample_rate)
            data['audio'] = audio

            txt_path = os.path.join(track_dir, track + '.txt')
            notes = load_valued_intervals(txt_path, comment='O|\n')
            notes = np.append(notes[0], np.expand_dims(notes[1], axis=-1), axis=-1)

            pitches, intervals = arr_to_note_groups(notes)
            times = self.data_proc.get_times(data['audio'])

            if isinstance(self.profile, PianoProfile):
                pitch = midi_groups_to_pianoroll(pitches, intervals, times, self.profile.get_midi_range())
            else:
                raise AssertionError('Provided InstrumentProfile not supported...')
            data['pitch'] = pitch

            notes[:, -1] = librosa.midi_to_hz(notes[:, -1])
            data['notes'] = notes

            if self.save_data:
                gt_path = self.get_gt_dir(track)
                np.savez(gt_path,
                         audio=audio,
                         pitch=pitch,
                         notes=notes)

        return data

    @staticmethod
    def available_splits():
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb',
                'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    @staticmethod
    def download(save_dir):
        # TODO - see if there is a way around this
        assert False, 'MAPS must be requested and downloaded manually'

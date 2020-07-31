# My imports
from datasets.common import TranscriptionDataset
from tools.io import *

# Regular imports
from mir_eval.io import load_valued_intervals

import numpy as np
import os


class MAPS(TranscriptionDataset):
    def __init__(self, base_dir=None, splits=None, hop_length=512,
                 data_proc=None, frame_length=None, split_notes=False, reset_data=False, seed=0):
        super().__init__(base_dir, splits, hop_length, data_proc, frame_length, split_notes, reset_data, seed)

    def remove_overlapping(self, splits):
        # Initialize list of tracks to remove
        tracks = []

        for split in splits:
            tracks += self.get_tracks(split)

        tracks = ['_'.join(t.split('_')[:-1]) for t in tracks]

        self.tracks = [t for t in self.tracks if '_'.join(t.split('_')[:-1]) not in tracks]

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

        if data is None:
            data = {}

            piano = track.split('_')[-1]
            track_dir = os.path.join(self.base_dir, piano, 'MUS')

            wav_path = os.path.join(track_dir, track + '.wav')
            audio, _ = load_audio(wav_path)
            data['audio'] = audio

            num_frames = self.data_proc.get_expected_frames(audio)

            txt_path = os.path.join(track_dir, track + '.txt')
            notes = load_valued_intervals(txt_path, comment='O')
            notes = np.append(np.expand_dims(notes[1], axis=-1), notes[0], axis=-1)
            data['notes'] = notes

            gt_path = self.get_gt_dir(track)
            np.savez(gt_path, audio=audio, frames=frames, notes=notes)

        data['track'] = track

        return data

    @staticmethod
    def available_splits():
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb',
                'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

    @staticmethod
    def dataset_name():
        return 'MAPS'

    @staticmethod
    def download(save_dir):
        # TODO - see if there is a way around this
        assert False, 'MAPS must be requested and downloaded manually'

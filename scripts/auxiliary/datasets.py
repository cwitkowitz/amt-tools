# My imports
from constants import *

# Regular imports
from torch.utils.data import Dataset
from abc import abstractmethod

import torch.nn.functional as F
import numpy as np
import mirdata
import shutil
import os

# TODO - ComboDataset

# TODO - build data proc module on top which is general - i.e. whole/local cqt or filterbank learning module

class TranscriptionDataset(Dataset):
    def __init__(self, splits, win_len, hop_len, seed):
        self.track_ids = track_ids

    def __len__(self):
        return len(self.track_ids)

    def __getitem__(self, index):
        track_name = self.track_ids[index]
        track = GuitarSetHandle[track_name]

        save_path = os.path.join(GEN_GT_DIR, f'{track_name}.npz')

        data = {}

        cnt_wn = 9

        if os.path.exists(save_path):
            clip = np.load(save_path)

            cqt = clip['cqt']
            tabs = clip['tabs']
        else:
            audio, fs = track.audio_mic

            # TODO - fix this
            audio = librosa.util.normalize(audio)

            audio = librosa.resample(audio, fs, SAMPLE_RATE)

            notes = track.notes  # Dictionary of notes for this track

            cqt = abs(librosa.cqt(audio, fs, self.hop_len, n_bins=192, bins_per_octave=24))

            num_bins = cqt.shape[0]
            num_frames = cqt.shape[-1]

            pad = cnt_wn // 2

            cqt = np.concatenate((np.zeros((num_bins, pad)),
                                  cqt, np.zeros((num_bins, pad))), axis=-1)

            cqt = np.expand_dims(cqt, axis=0)

            tabs = np.zeros((NUM_STRINGS, NUM_FRETS + 2, num_frames))

            jam = jams.load(track.jams_path)
            frame_indices = range(num_frames)
            t_ref = librosa.frames_to_time(frame_indices, SAMPLE_RATE, self.hop_len)

            for s in range(NUM_STRINGS):
                anno = jam.annotations['note_midi'][s]
                pitch = anno.to_samples(t_ref)
                silent = [pitch[i] == [] for i in range(len(pitch))]
                tabs[s, -1, silent] = 1
                for i in range(len(pitch)):
                    if silent[i]:
                        pitch[i] = [0]
                pitch = np.array(pitch).squeeze()
                midi_pitches = (np.round(pitch[pitch != 0] - librosa.note_to_midi(TUNING[s]))).astype('uint32')
                tabs[s, midi_pitches, pitch != 0] = 1

            np.savez(save_path, cqt=cqt, tabs=tabs)

        num_frames = cqt.shape[-1] - cnt_wn + 1

        if self.mode == 'train':
            step_begin = self.random.randint(num_frames)
            step_end = step_begin + cnt_wn

            data['cqt'] = cqt[:, :, step_begin:step_end]
            data['tabs'] = tabs[:, :, step_begin]
        else:
            # Batch size must be 1
            cqt = np.concatenate([cqt[:, :, i : i + cnt_wn] for i in range(num_frames)], axis=0)
            data['cqt'] = np.expand_dims(cqt, axis=1)
            data['tabs'] = np.transpose(tabs, (2, 0, 1))

        data['id'] = track_name
        data['cqt'] = torch.from_numpy(data['cqt']).float()
        data['tabs'] = torch.from_numpy(data['tabs']).long()

        return data

    @staticmethod
    @abstractmethod
    def available_splits():
        return NotImplementedError

    @staticmethod
    @abstractmethod
    def download(save_dir):
        return NotImplementedError

class MAPS(TranscriptionDataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    def available_splits():
        return ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl', 'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

class MAESTRO(TranscriptionDataset):
    def __init__(self):
        super().__init__()

    @staticmethod
    def available_splits():
        return ['train', 'validation', 'test']

class GuitarSet(TranscriptionDataset):
    def __init__(self, base_dir, track_ids, win_len, hop_len, mode, seed):
        super().__init__()

        # Get a reference to the dataset
        handle = mirdata.guitarset.load(data_home=base_dir)

        self.track_ids = track_ids

        self.win_len = win_len
        self.hop_len = hop_len
        self.mode = mode
        self.random = np.random.RandomState(seed)

    @staticmethod
    def available_splits():
        return ['train', 'validation', 'test']

    @staticmethod
    def download(save_dir):
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        os.mkdir(save_dir)

        mirdata.guitarset.download(data_home=save_dir)

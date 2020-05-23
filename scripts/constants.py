import numpy as np
import librosa
import os

SCPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCPT_DIR, '..'))
DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'data'))
GENR_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'generated'))

GEN_AUDIO_DIR   = os.path.join(GENR_DIR, 'audio')
GEN_ESTIM_DIR   = os.path.join(GENR_DIR, 'estimated')
GEN_RESULTS_DIR = os.path.join(GENR_DIR, 'results')
GEN_DICT_DIR    = os.path.join(GENR_DIR, 'dictionaries')
GEN_ACTS_DIR    = os.path.join(GENR_DIR, 'activations')
GEN_FIGS_DIR    = os.path.join(GENR_DIR, 'figures')
GEN_CLASS_DIR    = os.path.join(GENR_DIR, 'classifiers')
GEN_GT_DIR    = os.path.join(GENR_DIR, 'ground_truth')

GSET_DIR = os.path.join(DATA_DIR, 'GuitarSet')

TUNING = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']

NUM_STRINGS = 6
NUM_FRETS = 19

# Create an array with the midi numbers of each open string
TUNING_MIDI = np.array([librosa.note_to_midi(TUNING)]).T

# The lowest possible note - i.e. the open note of the lowest string
LOWEST_NOTE = librosa.note_to_midi(TUNING[0])

# The highest possible note - i.e. the maximum fret on the highest string
HIGHEST_NOTE = librosa.note_to_midi(TUNING[NUM_STRINGS - 1]) + NUM_FRETS

NUM_NOTES = HIGHEST_NOTE - LOWEST_NOTE + 1
NUM_GROUPS = NUM_STRINGS * (NUM_FRETS + 1)

SAMPLE_RATE = 44100

SEED = 0

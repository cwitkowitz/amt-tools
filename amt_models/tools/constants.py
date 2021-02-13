import os

## Directory Structure
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
SCPT_DIR = os.path.dirname(os.path.join(TOOL_DIR))
ROOT_DIR = os.path.dirname(os.path.join(SCPT_DIR))

## Default Paths
# TODO - make these all overridable (I think they are)
HOME = os.path.expanduser('~')
DEFAULT_DATASETS_DIR = os.path.join(HOME, 'Desktop', 'Datasets')
DEFAULT_GENERATED_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'generated'))
DEFAULT_FEATURES_GT_DIR = os.path.join(DEFAULT_GENERATED_DIR, 'data')
DEFAULT_EXPERIMENTS_DIR = os.path.join(DEFAULT_GENERATED_DIR, 'experiments')
DEFAULT_VISUALIZATION_DIR = os.path.join(DEFAULT_GENERATED_DIR, 'visualization')

## File Extensions
WAV_EXT = '.wav'
JAMS_EXT = '.jams'

## Ground-Truth & Prediction Keys
KEY_TRACK = 'track'
KEY_AUDIO = 'audio'
KEY_FS = 'fs'
KEY_MULTIPITCH = 'multipitch'
KEY_TABLATURE = 'tabs'
KEY_ONSET = 'onsets'
KEY_TIMES = 'times'
KEY_NOTES = 'notes'
# TODO - stacked multipitch, stacked notes, and continuous multipitch

## JAMS Attributes
JAMS_NOTE_MIDI = 'note_midi'
JAMS_PITCH_HZ = 'pitch_contour'

# Evaluation Keys
PR_KEY = 'precision'
RC_KEY = 'recall'
F1_KEY = 'f1-score'

NOTE_ON = 'note-on'
NOTE_OFF = 'note-off'

TAB_PITCH = 'pitch-tab'
TAB_NOTES = 'note-tab'

TDR = 'tdr'

## Default Instrument Parameters
# Guitar
DEFAULT_GUITAR_TUNING = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
DEFAULT_GUITAR_NUM_FRETS = 19

# Piano
DEFAULT_PIANO_LOWEST_PITCH = 21
DEFAULT_PIANO_HIGHEST_PITCH = 108

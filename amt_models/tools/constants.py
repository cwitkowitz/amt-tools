import os

##################################################
# PROJECT STRUCTURE                              #
##################################################

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
SCPT_DIR = os.path.dirname(os.path.join(TOOL_DIR))
ROOT_DIR = os.path.dirname(os.path.join(SCPT_DIR))

##################################################
# DEFAULT PATHS                                  #
##################################################

# TODO - make these all overridable (I think they are)
# TODO - should I highlight DIR vs. PATH?
HOME = os.path.expanduser('~')
DEFAULT_DATASETS_DIR = os.path.join(HOME, 'Desktop', 'Datasets')
DEFAULT_GENERATED_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'generated'))
GROUND_TRUTH_DIR = 'ground_truth'

DEFAULT_FEATURES_GT_DIR = os.path.join(DEFAULT_GENERATED_DIR, 'data')
DEFAULT_EXPERIMENTS_DIR = os.path.join(DEFAULT_GENERATED_DIR, 'experiments')
DEFAULT_VISUALIZATION_DIR = os.path.join(DEFAULT_GENERATED_DIR, 'visualization')

##################################################
# FILE EXTENSIONS                                #
##################################################

WAV_EXT = '.wav'
MID_EXT = '.mid'
MIDI_EXT = '.midi'
JAMS_EXT = '.jams'
NPZ_EXT = '.npz'
TXT_EXT = '.txt'

##################################################
# GROUND TRUTH / PREDICTION KEYS                 #
##################################################

KEY_TRACK = 'track'
KEY_AUDIO = 'audio'
KEY_FS = 'fs'
KEY_HOP = 'hop_length'
KEY_FEATS = 'features'
KEY_MULTIPITCH = 'multi_pitch'
KEY_TABLATURE = 'tablature'
KEY_ONSET = 'onsets'
KEY_OFFSET = 'offsets'
KEY_TIMES = 'times'
KEY_NOTES = 'notes'

##################################################
# JAMS ATTRIBUTES                                #
##################################################

JAMS_NOTE_MIDI = 'note_midi'
JAMS_PITCH_HZ = 'pitch_contour'

##################################################
# MIDI ATTRIBUTES                                #
##################################################

MIDI_NOTE_ON = 'note_on'
MIDI_NOTE_OFF = 'note_off'
MIDI_SUSTAIN_ON = 'sustain_on'
MIDI_SUSTAIN_OFF = 'sustain_off'
MIDI_SUSTAIN_CONTROL_NUM = 64
MIDI_CONTROL_CHANGE = 'control_change'

##################################################
# EVALUATION KEYS                                #
##################################################

PR_KEY = 'precision'
RC_KEY = 'recall'
F1_KEY = 'f1-score'

NOTE_ON = 'note-on'
NOTE_OFF = 'note-off'

TAB_PITCH = 'pitch-tab'
TAB_NOTES = 'note-tab'

TDR = 'tdr'

##################################################
# DEFAULT INSTRUMENT PARAMETERS                  #
##################################################

# Guitar
DEFAULT_GUITAR_TUNING = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
DEFAULT_GUITAR_NUM_FRETS = 19

# Piano
DEFAULT_PIANO_LOWEST_PITCH = 21
DEFAULT_PIANO_HIGHEST_PITCH = 108

##################################################
# DATA TYPES                                     #
##################################################

UINT = 'uint'
FLOAT32 = 'float32'

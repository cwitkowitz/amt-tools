# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

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

# TODO - should I explicitly state DIR vs. PATH?
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

WAV_EXT = 'wav'
MID_EXT = 'mid'
MIDI_EXT = 'midi'
JAMS_EXT = 'jams'
NPZ_EXT = 'npz'
TXT_EXT = 'txt'
PYT_EXT = 'pt'
CSV_EXT = 'csv'

##################################################
# GROUND TRUTH / PREDICTION KEYS                 #
##################################################

KEY_TRACK = 'track'
KEY_AUDIO = 'audio'
KEY_FS = 'fs'
KEY_HOP = 'hop_length'
KEY_FEATS = 'features'
KEY_MULTIPITCH = 'multi_pitch'
KEY_PITCHLIST = 'pitch_list'
KEY_TABLATURE = 'tablature'
KEY_ONSETS = 'onsets'
KEY_OFFSETS = 'offsets'
KEY_TIMES = 'times'
KEY_NOTES = 'notes'
KEY_OUTPUT = 'model_output'
KEY_ACCURACY = 'accuracy'

KEY_LOSS = 'loss'
KEY_LOSS_TOTAL = 'loss_total'
KEY_LOSS_ONSETS = 'loss_onsets'
KEY_LOSS_OFFSETS = 'loss_offsets'
KEY_LOSS_PITCH = 'loss_pitch'
KEY_LOSS_TABS = 'loss_tabs'
KEY_LOSS_KLD = 'loss_kld'
KEY_LOSS_INH = 'loss_inhib'
KEY_LOSS_REC = 'loss_recon'

##################################################
# JAMS ATTRIBUTES                                #
##################################################

JAMS_NOTE_MIDI = 'note_midi'
JAMS_PITCH_HZ = 'pitch_contour'
JAMS_STRING_IDX = 'data_source'
JAMS_METADATA = 'file_metadata'

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
# LOGGING/EVALUATION KEYS                        #
##################################################

TRAIN = 'train'
VAL = 'validation'
TEST = 'test'

KEY_PRECISION = 'precision'
KEY_RECALL = 'recall'
KEY_F1 = 'f1-score'

KEY_NOTE_ON = 'note-on'
KEY_NOTE_OFF = 'note-off'

KEY_TDR = 'tdr'

##################################################
# DEFAULT INSTRUMENT PARAMETERS                  #
##################################################

# Guitar
DEFAULT_GUITAR_LABELS = ['E', 'A', 'D', 'G', 'B', 'e']
DEFAULT_GUITAR_TUNING = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
DEFAULT_GUITAR_NUM_FRETS = 19

# Piano
DEFAULT_PIANO_LOWEST_PITCH = 21
DEFAULT_PIANO_HIGHEST_PITCH = 108

##################################################
# DATA TYPES                                     #
##################################################

UINT = 'uint'
INT = 'int'
INT64 = 'int64'
FLOAT = 'float'
FLOAT32 = 'float32'
FLOAT64 = 'float64'

##################################################
# MISCELLANEOUS                                  #
##################################################

PYT_MODEL = 'model'
PYT_STATE = 'opt-state'

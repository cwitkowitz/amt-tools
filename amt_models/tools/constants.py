import os

# TODO - add extensions, folder names, dictionary keys, etc.

HOME = os.path.expanduser('~')

# Directory Structure
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
SCPT_DIR = os.path.dirname(os.path.join(TOOL_DIR))
ROOT_DIR = os.path.dirname(os.path.join(SCPT_DIR))

# Default Paths
# TODO - make these parameterizable
GENR_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'generated'))
GEN_DATA_DIR = os.path.join(GENR_DIR, 'data')
GEN_EXPR_DIR = os.path.join(GENR_DIR, 'experiments')
GEN_VISL_DIR = os.path.join(GENR_DIR, 'visualization')

# Ground-Truth & Prediction Keys
KEY_AUDIO = 'audio'
KEY_FS = 'fs'
KEY_TRACK = 'track'
KEY_MULTIPITCH = 'pitch'
KEY_TABLATURE = 'tabs'
KEY_ONSET = 'onsets'
KEY_TIMES = 'times'
KEY_NOTES = 'notes'
# TODO - stacked multipitch, stacked notes, and continuous multipitch

# JAMS Attributes
JAMS_NOTE_MIDI = 'note_midi'

# Evaluation Keys
PR_KEY = 'precision'
RC_KEY = 'recall'
F1_KEY = 'f1-score'

NOTE_ON = 'note-on'
NOTE_OFF = 'note-off'

TAB_PITCH = 'pitch-tab'
TAB_NOTES = 'note-tab'

TDR = 'tdr'

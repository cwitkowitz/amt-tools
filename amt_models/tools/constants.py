import os

# TODO - add extensions, folder names, dictionary keys, etc.

# Directory Structure
HOME = os.path.expanduser('~')
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
SCPT_DIR = os.path.dirname(os.path.join(TOOL_DIR))
ROOT_DIR = os.path.dirname(os.path.join(SCPT_DIR))
GENR_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'generated'))
GEN_DATA_DIR = os.path.join(GENR_DIR, 'data')
GEN_EXPR_DIR = os.path.join(GENR_DIR, 'experiments')

# Dataset Keys
TR_ID = 'track'
PITCH = 'pitch'
ONSET = 'onsets'
TIMES = 'times'
NOTES = 'notes'

# Transcription Keys
SOLO_PITCH = 'pitch_single'
SOLO_NOTES = 'notes_single'
MULT_PITCH = 'pitch_multi'
MULT_NOTES = 'notes_multi'

# Evaluation Keys
PR_KEY = 'precision'
RC_KEY = 'recall'
F1_KEY = 'f1-score'

NOTE_ON = 'note-on'
NOTE_OFF = 'note-off'

TAB_PITCH = 'pitch-tab'
TAB_NOTES = 'note-tab'

TDR = 'tdr'

import os

# TODO - add extensions and folder names

HOME = os.path.expanduser('~')
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
SCPT_DIR = os.path.dirname(os.path.join(TOOL_DIR))
ROOT_DIR = os.path.dirname(os.path.join(SCPT_DIR))
GENR_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'generated'))
GEN_DATA_DIR = os.path.join(GENR_DIR, 'data')
GEN_EXPR_DIR = os.path.join(GENR_DIR, 'experiments')

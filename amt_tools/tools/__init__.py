"""
Should be able to use the following import patterns (e.g.):
------------------------------------------------------------
import amt_tools
amt_tools.tools.load_normalize_audio()
------------------------------------------------------------
import amt_tools.tools as tools
tools.load_notes_midi()
------------------------------------------------------------
from amt_tools import tools
tools.array_to_tensor()
------------------------------------------------------------
from amt_tools.tools import *
notes_to_multi_pitch()
------------------------------------------------------------
from amt_tools.tools import GuitarProfile
GuitarProfile()
------------------------------------------------------------
from amt_tools.tools.constants import KEY_AUDIO
KEY_AUDIO
"""

from .constants import *
from .instrument import InstrumentProfile, PianoProfile, TablatureProfile, GuitarProfile
from .io import *
from .utils import *
from .visualize import *

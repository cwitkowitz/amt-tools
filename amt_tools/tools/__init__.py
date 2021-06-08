"""
Should be able to use the following import structures (e.g.):
------------------------------------------------------------
import amt_tools
amt_tools.tools.rms_norm()
amt_tools.tools.load_normalize_audio()
------------------------------------------------------------
import amt_tools.tools as t
t.rms_norm()
t.load_normalize_audio()
t.KEY_AUDIO
------------------------------------------------------------
from amt_tools.tools import *
rms_norm()
load_normalize_audio()
KEY_AUDIO
------------------------------------------------------------
from amt_tools.tools import rms_norm
rms_norm()
------------------------------------------------------------
from amt_tools.tools.utils import rms_norm
rms_norm()
"""

from .constants import *
from .instrument import *
from .io import *
from .utils import *
#from .visualize import *

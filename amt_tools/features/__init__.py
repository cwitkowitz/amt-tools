"""
Should be able to use the following import patterns (e.g.):
------------------------------------------------------------
import amt_tools
amt_tools.features.STFT()
------------------------------------------------------------
import amt_tools.features as ft
ft.CQT()
------------------------------------------------------------
from amt_tools import features
features.VQT()
------------------------------------------------------------
from amt_tools.features import *
MelSpec()
------------------------------------------------------------
from amt_tools.features import HCQT
HCQT()
------------------------------------------------------------
from amt_tools.features.hvqt import HVQT
HVQT()
"""

from .combo import *
from .common import *
from .cqt import *
from .hcqt import *
from .hvqt import *
from .mel import *
from .stft import *
from .vqt import *

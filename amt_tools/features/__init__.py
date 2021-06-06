"""
Should be able to use the following import structures (e.g.):
------------------------------------------------------------
import amt_tools
amt_tools.features.VQT()
amt_tools.features.MelSpec()
------------------------------------------------------------
import amt_tools.features as f
f.VQT()
f.MelSpec()
------------------------------------------------------------
from amt_tools.features import *
VQT()
MelSpec()
------------------------------------------------------------
from amt_tools.features import VQT
VQT()
------------------------------------------------------------
from amt_tools.features.vqt import VQT
VQT()
"""

from .combo import *
from .common import *
from .cqt import *
from .hcqt import *
from .hvqt import *
from .mel import *
from .stft import *
from .vqt import *

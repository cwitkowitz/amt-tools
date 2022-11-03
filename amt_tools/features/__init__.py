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

from .combo import FeatureCombo
from .common import FeatureModule
from .cqt import CQT
from .hcqt import HCQT
from .hvqt import HVQT
from .mel import MelSpec
from .power import SignalPower
from .stft import STFT
from .stream import FeatureStream, MicrophoneStream, AudioStream, AudioFileStream
from .vqt import VQT
from .waveform import WaveformWrapper

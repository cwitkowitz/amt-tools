"""
Should be able to use the following import patterns (e.g.):
------------------------------------------------------------
import amt_tools
amt_tools.datasets.GuitarSet()
------------------------------------------------------------
import amt_tools.datasets as dt
dt.GuitarSet()
------------------------------------------------------------
from amt_tools import datasets
datasets.MAESTRO_V3()
------------------------------------------------------------
from amt_tools.datasets import *
MAESTRO_V3()
------------------------------------------------------------
from amt_tools.datasets import MAPS
MAPS()
------------------------------------------------------------
from amt_tools.datasets.MAPS import MAPS
MAPS()
"""

from .combo import DatasetCombo
from .common import TranscriptionDataset
from .GuitarSet import GuitarSet
from .MAESTRO import _MAESTRO, MAESTRO_V1, MAESTRO_V2, MAESTRO_V3
from .MAPS import MAPS

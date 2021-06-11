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

from .combo import *
from .common import *
from .GuitarSet import *
from .MAESTRO import *
from .MAPS import *

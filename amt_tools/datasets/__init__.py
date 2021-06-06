"""
Should be able to use the following import structures (e.g.):
------------------------------------------------------------
import amt_tools
amt_tools.datasets.GuitarSet()
amt_tools.datasets.MAESTRO_V2()
------------------------------------------------------------
import amt_tools.datasets as d
d.GuitarSet()
d.MAESTRO_V2()
------------------------------------------------------------
from amt_tools.datasets import *
GuitarSet()
MAESTRO_V2()
------------------------------------------------------------
from amt_tools.datasets import GuitarSet
GuitarSet()
------------------------------------------------------------
from amt_tools.datasets.GuitarSet import GuitarSet
GuitarSet()
"""

from .combo import *
from .common import *
from .GuitarSet import *
from .MAESTRO import *
from .MAPS import *

"""
Should be able to use the following import structures (e.g.):
------------------------------------------------------------
import amt_models
amt_models.datasets.GuitarSet()
amt_models.datasets.MAESTRO_V2()
------------------------------------------------------------
import amt_models.datasets as d
d.GuitarSet()
d.MAESTRO_V2()
------------------------------------------------------------
from amt_models.datasets import *
GuitarSet()
MAESTRO_V2()
------------------------------------------------------------
from amt_models.datasets import GuitarSet
GuitarSet()
------------------------------------------------------------
from amt_models.datasets.GuitarSet import GuitarSet
GuitarSet()
"""

from .combo import *
from .common import *
from .GuitarSet import *
from .MAESTRO import *
from .MAPS import *

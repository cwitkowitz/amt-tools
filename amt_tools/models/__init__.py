"""
Should be able to use the following import structures (e.g.):
------------------------------------------------------------
import amt_tools
amt_tools.models.TranscriptionModel()
amt_tools.models.LogisticBank()
amt_tools.models.OnsetsFrames()
------------------------------------------------------------
import amt_tools.models as m
m.TranscriptionModel()
m.LogisticBank()
m.OnsetsFrames()
------------------------------------------------------------
from amt_tools.models import *
TranscriptionModel()
LogisticBank()
OnsetsFrames()
------------------------------------------------------------
from amt_tools.models import OnsetsFrames
OnsetsFrames()
------------------------------------------------------------
from amt_tools.models.onsetsframes import OnsetsFrames
OnsetsFrames()
"""

from .common import *
from .onsetsframes import *
from .tabcnn import *

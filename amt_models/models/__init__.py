"""
Should be able to use the following import structures (e.g.):
------------------------------------------------------------
import amt_models
amt_models.models.TranscriptionModel()
amt_models.models.LogisticBank()
amt_models.models.OnsetsFrames()
------------------------------------------------------------
import amt_models.models as m
m.TranscriptionModel()
m.LogisticBank()
m.OnsetsFrames()
------------------------------------------------------------
from amt_models.models import *
TranscriptionModel()
LogisticBank()
OnsetsFrames()
------------------------------------------------------------
from amt_models.models import OnsetsFrames
OnsetsFrames()
------------------------------------------------------------
from amt_models.models.onsetsframes import OnsetsFrames
OnsetsFrames()
"""

from .common import *
from .onsetsframes import *
from .tabcnn import *

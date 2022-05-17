"""
Should be able to use the following import patterns (e.g.):
------------------------------------------------------------
import amt_tools
amt_tools.models.TabCNN()
------------------------------------------------------------
import amt_tools.models as md
md.OnsetsFrames()
------------------------------------------------------------
from amt_tools import models
models.OnsetsFrames2()
------------------------------------------------------------
from amt_tools.models import *
AcousticModel()
------------------------------------------------------------
from amt_tools.models import LanguageModel
LanguageModel()
------------------------------------------------------------
from amt_tools.models.common import LogisticBank, SoftmaxGroups
LogisticBank()
SoftmaxGroups()
"""

from .common import TranscriptionModel, OutputLayer, SoftmaxGroups, LogisticBank
from .onsetsframes import OnsetsFrames, OnsetsFrames2, AcousticModel, LanguageModel, OnlineLanguageModel
from .tabcnn import TabCNN

"""
Should be able to use the following import structures (e.g.):
------------------------------------------------------------
import amt_models
amt_models.train()
amt_models.transcribe()
amt_models.evaluate()
------------------------------------------------------------
from amt_models import train
train()
------------------------------------------------------------
from amt_models.train import train
train()
"""

from .evaluate import *
from .train import *
from .transcribe import *

"""
These are necessary in order to be able to access classes
and functions in submodules using only 'import amt_models'
"""

from . import datasets
from . import features
from . import models
from . import tools

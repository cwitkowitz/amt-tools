"""
Should be able to use the following import structures (e.g.):
------------------------------------------------------------
import amt_tools
amt_tools.train()
amt_tools.transcribe()
amt_tools.evaluate()
------------------------------------------------------------
from amt_tools import train
train()
------------------------------------------------------------
from amt_tools.train import train
train()
"""

from .evaluate import *
from .train import *
from .transcribe import *

"""
These are necessary in order to be able to access classes
and functions in submodules using only 'import amt_tools'
"""

from . import datasets
from . import features
from . import models
from . import tools

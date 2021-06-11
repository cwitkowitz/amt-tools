"""
Should be able to use the following import patterns (e.g.):
------------------------------------------------------------
import amt_tools
amt_tools.train.train()
------------------------------------------------------------
import amt_tools.train as tr
tr.validate()
------------------------------------------------------------
from amt_tools import transcribe
transcribe.Estimator()
------------------------------------------------------------
from amt_tools.evaluate import Evaluator
Evaluator()
------------------------------------------------------------
from amt_tools.evaluate import *
Evaluator()
"""

# Subpackages
from . import datasets
from . import features
from . import models
from . import tools

# Scripts
from . import evaluate
from . import train
from . import transcribe

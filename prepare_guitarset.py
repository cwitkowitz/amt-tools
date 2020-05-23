"""
Download GuitarSet Dataset
"""

# My imports
from constants import *

# Regular imports
import mirdata
import shutil
import os

if os.path.exists(GSET_DIR):
    shutil.rmtree(GSET_DIR)

os.mkdir(GSET_DIR)

mirdata.guitarset.download(data_home=GSET_DIR)

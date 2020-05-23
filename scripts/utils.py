# My imports
from constants import *

# Regular imports
import mirdata
import shutil
import os

# Get a reference to the dataset
GuitarSetHandle = mirdata.guitarset.load(data_home=GSET_DIR)

def reset_generated_dir(dir_path, subdirs, rmv = True):
    """
    Description

    Parameters
    ----------
    dir_path : str
      Base-directory path
    subdirs : list of str
      List of sub-directories underneath the base directory
    rmv : bool
      Switch for recursive removal of base directory
    """
    # TODO - os.makedirs(exist_ok = true)
    # Remove the directory and everything underneath it if it exists
    if os.path.exists(dir_path) and rmv:
        shutil.rmtree(dir_path)

    # If the directory was removed, create it again
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    for dir in subdirs:
        # Subdirectory path
        path = os.path.join(dir_path, dir)

        # If the sub-directory was removed, create it again
        if not os.path.exists(path):
            reset_generated_dir(path, [], False)

def clean_track_list(dset, single, player, rmv):
    if single != '':
        track_keys = [single]
    else:
        # Create a copy of the track list to iterate through
        track_keys = list(dset.keys())
        track_keys_copy = track_keys.copy()
        # Remove any tracks that will not be used for testing data
        for id in track_keys_copy:
            track = dset[id]

            # Remove any tracks with non-matching selected attributes
            if (track.player_id != player and not rmv) or (track.player_id == player and rmv):
                track_keys.remove(id)

    return track_keys
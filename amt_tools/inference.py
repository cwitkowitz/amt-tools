# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .transcribe import *
from . import tools

# Regular imports
import numpy as np
import argparse
import torch


def run_offline(track_data, model, estimator=None):
    """
    Perform inference in an offline fashion.

    Parameters
    ----------
    track_data : dict
      Dictionary containing relevant features for a track
    model : TranscriptionModel
      Model to use for inference
    estimator : Estimator
      Estimation protocol to use

    Returns
    ----------
    predictions : dict
      Dictionary containing predictions for a track
    """

    # Obtain the name of the track if it exists
    track_id = tools.unpack_dict(track_data, tools.KEY_TRACK)

    # Treat the track data as a batch
    track_data = tools.dict_unsqueeze(tools.dict_to_tensor(track_data))

    # Get the model predictions and convert them to NumPy arrays
    predictions = tools.dict_squeeze(tools.dict_to_array(model.run_on_batch(track_data)), dim=0)

    if estimator is not None:
        # Perform any estimation steps (e.g. note transcription)
        predictions.update(estimator.process_track(predictions, track_id))

    return predictions


def run_single_frame(track_data, model, estimator=None):
    """
    Perform inference on a single frame.

    Parameters
    ----------
    track_data : dict
      Dictionary containing relevant features for a track
    model : TranscriptionModel
      Model to use for inference
    estimator : Estimator
      Estimation protocol to use

    Returns
    ----------
    predictions : dict
      Dictionary containing predictions for a track
    """

    # Obtain the name of the track if it exists
    track_id = tools.unpack_dict(track_data, tools.KEY_TRACK)

    # Make sure the track data consists of tensors
    track_data = tools.dict_to_tensor(track_data)

    # Run the frame group through the model
    new_predictions = tools.dict_squeeze(tools.dict_to_array(model.run_on_batch(track_data)), dim=0)

    if estimator is not None:
        # Perform any estimation steps (e.g. note transcription)
        new_predictions.update(estimator.process_track(new_predictions, track_id))

    return new_predictions


def run_online(track_data, model, estimator=None):
    """
    Perform inference in an mock-online fashion.

    Parameters
    ----------
    track_data : dict
      Dictionary containing relevant features for a track
    model : TranscriptionModel
      Model to use for inference
    estimator : Estimator
      Estimation protocol to use

    Returns
    ----------
    predictions : dict
      Dictionary containing predictions for a track
    """

    # Obtain the features and times from the track data
    features = tools.unpack_dict(track_data, tools.KEY_FEATS)
    times = tools.unpack_dict(track_data, tools.KEY_TIMES)

    # Determine the number of frame groups to feed through the model
    num_frame_groups = features.shape[-1]

    # Window the features to mimic real-time operation
    features = tools.framify_activations(features, model.frame_width)
    # Convert the features to PyTorch tensor and add to device
    features = tools.array_to_tensor(features, model.device)

    # Initialize a dictionary to hold predictions
    predictions = {}

    # Feed the frame groups to the model one-at-a-time
    for i in range(num_frame_groups):
        # Treat the next frame groups as a batch of features
        batch = tools.dict_unsqueeze({tools.KEY_FEATS : features[..., i, :],
                                      tools.KEY_TIMES : times[..., i : i+1]})
        # Perform inference on a single frame
        new_predictions = run_single_frame(batch, model, estimator)
        # Append the new predictions
        predictions = tools.dict_append(predictions, new_predictions)

    return predictions

def run_inference(model, dataset, estimator=None, online=False):
    """
    Implements the inference loop for a model and dataset partition. Optionally save predictions.

    Parameters
    ----------
    model : TranscriptionModel
      Model to validate or evaluate
    dataset : TranscriptionDataset
      Dataset (partition) to use for validation or evaluation
    estimator : Estimator
      Estimation protocol to use
    online : bool
      Whether to evaluate the model in a mock-real-time fashion

    Returns
    ----------
    all_predictions : dict
      Inference results for all tracks in the dataset
    """

    # Make sure the model is in evaluation mode
    model.eval()

    # Initialize an empty dictionary to hold predictions
    all_predictions = {}

    # Turn off gradient computation
    with torch.no_grad():
        # Loop through the validation track ids
        for track_id in dataset.tracks:
            # Obtain the track data
            track_data = dataset.get_track_data(track_id)

            if online:
                # Perform the inference step in mock-real-time fashion
                predictions = run_online(track_data, model, estimator)
            else:
                # Perform the inference step offline
                predictions = run_offline(track_data, model, estimator)

            # Add the predictions to the dictionary
            all_predictions[track_id] = predictions

    return all_predictions

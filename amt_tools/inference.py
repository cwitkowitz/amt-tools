# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from . import tools

__all__ = [
    'run_offline',
    'run_online'
]


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

    # Convert all numpy arrays in the data dictionary to float32
    track_data = tools.dict_to_dtype(track_data, dtype=tools.FLOAT32)

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
    TODO - this is all repeat code of run_offline - only difference is lack of dict_unsqueeze

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

    # Convert all numpy arrays in the data dictionary to float32
    track_data = tools.dict_to_dtype(track_data, dtype=tools.FLOAT32)

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
    TODO - no loss entry in predictions causes whole dict to be interpreted as loss
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

    # TODO - if available, should ground-truth be fed in one frame at a time?

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

    # Check if there are notes in the predictions
    if tools.query_dict(predictions, tools.KEY_NOTES):
        # TODO - this won't work for stacked notes - is there a need to support those here?
        # If so, they will need to be transposed, since they were processed in an online fashion
        predictions[tools.KEY_NOTES] = tools.transpose_batched_notes(predictions[tools.KEY_NOTES])

    if estimator is not None:
        # Reset the state for the next track
        estimator.reset_state()

    return predictions

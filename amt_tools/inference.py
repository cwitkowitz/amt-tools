# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .transcribe import *
from . import tools

# Regular imports
import numpy as np
import argparse
import torch

# TODO - turn this into an executable with parameterizable model and stream (or audio)
# TODO - add capability for realtime plotting of results
# TODO - optional evaluation for mock
# TODO - better function names
# TODO - link feature extraction somehow


def run_offline(track_data, model, estimator=None):
    """
    TODO
    """

    # Obtain the name of the track if it exists
    track_id = tools.try_unpack_dict(track_data, tools.KEY_TRACK)

    # Treat the track data as a batch
    track_data = tools.dict_unsqueeze(tools.dict_to_tensor(track_data))

    # Get the model predictions and convert them to NumPy arrays
    predictions = tools.dict_squeeze(tools.dict_to_array(model.run_on_batch(track_data)))

    if estimator is not None:
        # Perform any estimation steps (e.g. note transcription)
        predictions.update(estimator.process_track(predictions, track_id))

    return predictions


def run_online(track_data, model, estimator=None):
    """
    TODO
    """

    # Obtain the name of the track if it exists
    track_id = tools.try_unpack_dict(track_data, tools.KEY_TRACK)

    # Treat the track data as a batch
    track_data = tools.track_to_batch(track_data)

    # Obtain the features from the track data as a NumPy array
    features = tools.tensor_to_array(tools.try_unpack_dict(track_data, tools.KEY_FEATS))
    # Window the features to mimic real-time operation
    features = tools.framify_activations(features, model.frame_width)
    # Convert the features back to PyTorch tensor and add to original device
    features = tools.array_to_tensor(features, model.device)

    predictions = {}

    # TODO - define outside
    #estimator = ComboEstimator[TablatureWrapper(stacked=True),
    #                           IterativeStackedNoteTranscriber()]

    # Feed the frame groups to the model one-at-a-time
    for i in range(features.size(-2)):
        batch = {tools.KEY_FEATS: features[..., i, :]}
        predictions = tools.append_track(predictions, model.run_on_batch(batch))
        #predictions = estimator.step(predictions)

    if estimator is not None:
        # Perform any estimation steps (e.g. note transcription)
        predictions = tools.track_to_cpu(predictions.update(estimator.process_track(predictions, track_id)))

    # TODO - iterative estimator

    return predictions


def stream(stream, model):
    """
    TODO - keep track of a buffer of size frame_width and feed into model
    """

    pass


"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process audio using a transcription model.')
    parser.add_argument('model_path',
                        metavar='path/to/model',
                        help='path to transcription model')
    parser.add_argument('--audio_path',
                        metavar='path/to/audio',
                        help='path to audio file to transcribe instead of streaming')
    parser.add_argument('-o', '--online',
                        action='store_true',
                        help='whether to process specified audio in online fashion')
    parser.add_argument('-g', '--gpu_id',
                        type=int,
                        help='index of GPU to use for inference')
    args = parser.parse_args()

    #print(args)

    device = torch.device(f'cuda:{args.gpu_id}'
                          if torch.cuda.is_available() else 'cpu')

    model = torch.load(args.model_path, map_location=device)
    model.change_device(args.gpu_id)

    if args.audio_path:
        audio = tools.load_normalize_audio(args.audio_path)
        if args.online:
            mock_realtime(audio, model)
        else:
            offline(audio, model)
    else:
        print('realtime')
        realtime()
"""

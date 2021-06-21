# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
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


def offline(audio, model):
    pass


def mock_realtime(audio, model):
    """
    TODO - takes an entire audio files and runs it through a model as if in real-time
    """

    pass


def realtime(stream, model):
    """
    TODO - keep track of a buffer of size frame_width and feed into model
    """

    pass


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

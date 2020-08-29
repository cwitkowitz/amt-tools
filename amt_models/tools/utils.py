# My imports
from tools.constants import *

# Regular imports
import numpy as np
import mir_eval
import random
import torch


def seed_everything(seed):
    # WARNING: the number of workers in the training loader affects behavior:
    #          this is because each sample will inevitably end up being processed
    #          by a different worker if num_workers is changed, and each worker
    #          has its own random seed
    #          TODO - I will fix this in the future if possible

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


# TODO - should I be using a yield function instead of for batch in loader?


def rms_norm(audio):
    rms = np.sqrt(np.mean(audio ** 2))

    assert rms != 0

    audio = audio / rms

    return audio


def framify_tfr(tfr, win_length, hop_length, pad=None):
    # TODO - avoid conversion in collate_fn instead?
    to_torch = False
    if 'torch' in str(tfr.dtype):
        to_torch = True
        tfr = tfr.cpu().detach().numpy()

    # TODO - parameterize axis or just assume -1?
    if pad is not None:
        # TODO - librosa pad center?
        pad_amts = [(0,)] * (len(tfr.shape) - 1) + [(pad,)]
        tfr = np.pad(tfr, tuple(pad_amts))

    #tfr = np.asfortranarray(tfr)
    # TODO - this is a cleaner solution but seems to be very unstable
    #stack = librosa.util.frame(tfr, win_length, hop_length).copy()

    dims = tfr.shape
    num_hops = (dims[-1] - 2 * pad) // hop_length
    hops = np.arange(0, num_hops, hop_length)
    new_dims = dims[:-1] + (win_length, num_hops)

    tfr = tfr.reshape(np.prod(dims[:-1]), dims[-1])
    tfr = [np.expand_dims(tfr[:, i : i + win_length], axis=-1) for i in hops]

    stack = np.concatenate(tfr, axis=-1)
    stack = np.reshape(stack, new_dims)

    if to_torch:
        stack = torch.from_numpy(stack)

    return stack


def infer_lowest_note(pianoroll):
    note_range = pianoroll.shape[0]
    if note_range == PIANO_RANGE:
        return PIANO_LOWEST
    elif note_range == GUITAR_RANGE:
        return GUITAR_LOWEST
    else:
        # Something went awry
        return None


def threshold_arr(arr, thr):
    arr[arr < thr] = 0
    arr[arr != 0] = 1
    return arr


def valid_single(activations):
    single = True

    if len(activations.shape) != 2:
        single = False

    if isinstance(activations, np.ndarray):
        if np.max(activations) > 1:
            single = False

        if np.min(activations) < 0:
            single = False

    if isinstance(activations, torch.Tensor):
        if torch.max(activations) > 1:
            single = False

        if torch.min(activations) < 0:
            single = False

    return single


def valid_multi(activations):
    multi = True

    if len(activations.shape) != 3:
        multi = False

    return multi


def valid_tabs(activations):
    """
    Conceivably, there is a situation where
    activations could be valid pianoroll and tabs
    TODO - improve so this corner case doesn't exist
    """

    tabs = True

    if len(activations.shape) != 2:
        tabs = False

    # Must be exact integers
    if isinstance(activations, np.ndarray):
        if np.sum(activations - np.round(activations)) != 0:
            tabs = False

    if isinstance(activations, torch.Tensor):
        if torch.sum(activations - torch.round(activations)) != 0:
            tabs = False

    return tabs


def valid_notes(pitches, intervals):
    # Validate the intervals
    valid = librosa.util.valid_intervals(intervals)

    # Validate the pitches - should be in Hz
    try:
        mir_eval.util.validate_frequencies(pitches, 5000, 20)
    except ValueError:
        valid = False

    return valid


# TODO - use this standardized version everywhere
def get_batch_size(batch):
    if isinstance(batch, dict):
        bs = len(batch['track'])
    elif isinstance(batch, np.ndarray) or isinstance(batch, torch.Tensor):
        bs = batch.shape[0]
    else:
        bs = None

    return bs

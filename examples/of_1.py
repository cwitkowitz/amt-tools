# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.models import OnsetsFrames
from amt_tools.features import MelSpec
from amt_tools.datasets import MAPS

from amt_tools import train, validate
from amt_tools.transcribe import *
from amt_tools.evaluate import *

import amt_tools.tools as tools

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import torch
import os

EX_NAME = '_'.join([OnsetsFrames.model_name(),
                    MAPS.dataset_name(),
                    MelSpec.features_name()])

ex = Experiment('Onsets & Frames 1 w/ Mel Spectrogram on MAPS')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 16000

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 625

    # Number of training iterations to conduct
    iterations = 2000

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 40

    # Number of samples to gather for a batch
    batch_size = 8

    # The initial learning rate
    learning_rate = 6e-4

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different feature extraction parameters
    reset_data = True

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    root_dir = os.path.join(tools.DEFAULT_EXPERIMENTS_DIR, EX_NAME)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def onsets_frames_run(sample_rate, hop_length, num_frames, iterations, checkpoints,
                      batch_size, learning_rate, gpu_id, reset_data, seed, root_dir):
    # Seed everything with the same seed
    tools.seed_everything(seed)

    # Initialize the default piano profile
    profile = tools.PianoProfile()

    # Processing parameters
    dim_in = 229
    model_complexity = 2

    # Create the mel spectrogram data processing module
    data_proc = MelSpec(sample_rate=sample_rate,
                        n_mels=dim_in,
                        hop_length=hop_length)

    # Initialize the estimation pipeline
    validation_estimator = ComboEstimator([NoteTranscriber(profile=profile),
                                           PitchListWrapper(profile=profile)])

    # Initialize the evaluation pipeline
    evaluators = [LossWrapper(),
                  MultipitchEvaluator(),
                  NoteEvaluator(key=tools.KEY_NOTE_ON),
                  NoteEvaluator(offset_ratio=0.2, key=tools.KEY_NOTE_OFF)]
    validation_evaluator = ComboEvaluator(evaluators, patterns=['loss', 'f1'])

    # Get a list of the MAPS splits
    splits = MAPS.available_splits()

    # Initialize the testing splits as the real piano data
    test_splits = ['ENSTDkAm', 'ENSTDkCl']
    # Remove the real piano splits to get the training partition
    train_splits = splits.copy()
    for split in test_splits:
        train_splits.remove(split)

    print('Loading training partition...')

    # Create a dataset corresponding to the training partition
    maps_train = MAPS(splits=train_splits,
                      hop_length=hop_length,
                      sample_rate=sample_rate,
                      data_proc=data_proc,
                      profile=profile,
                      num_frames=num_frames,
                      reset_data=reset_data)

    # Remove tracks in both partitions from the training partitions
    print('Removing overlapping tracks from training partition')
    maps_train.remove_overlapping(test_splits)

    # Create a PyTorch data loader for the dataset
    train_loader = DataLoader(dataset=maps_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              drop_last=True)

    print('Loading testing partition...')

    # Create a dataset corresponding to the testing partition
    maps_test = MAPS(splits=test_splits,
                     hop_length=hop_length,
                     sample_rate=sample_rate,
                     data_proc=data_proc,
                     profile=profile,
                     store_data=True)

    print('Initializing model...')

    # Initialize a new instance of the model
    onsetsframes = OnsetsFrames(dim_in, profile, data_proc.get_num_channels(), model_complexity, False, gpu_id)
    onsetsframes.change_device()
    onsetsframes.train()

    # Initialize a new optimizer for the model parameters
    optimizer = torch.optim.Adam(onsetsframes.parameters(), learning_rate)

    print('Training classifier...')

    # Create a log directory for the training experiment
    model_dir = os.path.join(root_dir, 'models')

    # Train the model
    onsetsframes = train(model=onsetsframes,
                         train_loader=train_loader,
                         optimizer=optimizer,
                         iterations=iterations,
                         checkpoints=checkpoints,
                         log_dir=model_dir,
                         val_set=None)

    print('Transcribing and evaluating test partition...')

    # Add save directories to the estimators
    validation_estimator.set_save_dirs(os.path.join(root_dir, 'estimated'), ['notes', 'pitch'])

    # Add a save directory to the evaluators and reset the patterns
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results'))
    validation_evaluator.set_patterns(None)

    # Get the average results for the testing partition
    results = validate(onsetsframes, maps_test, evaluator=validation_evaluator, estimator=validation_estimator)

    # Log the average results in metrics.json
    ex.log_scalar('Final Results', results, 0)

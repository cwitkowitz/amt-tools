# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.models import OnsetsFrames2
from amt_tools.datasets import MAESTRO_V3, MAPS
from amt_tools.features import MelSpec

from amt_tools.train import train
from amt_tools.transcribe import *
from amt_tools.evaluate import *

import amt_tools.tools as tools

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import torch
import os

EX_NAME = '_'.join([OnsetsFrames2.model_name(),
                    MAESTRO_V3.dataset_name(),
                    MelSpec.features_name()])

ex = Experiment('Onsets & Frames 2 w/ Mel Spectrogram on MAESTRO')


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
    checkpoints = 100

    # Number of samples to gather for a batch
    batch_size = 8

    # The initial learning rate
    learning_rate = 6e-4

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different feature extraction parameters
    reset_data = False

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
    model_complexity = 3

    # Create the mel spectrogram data processing module
    data_proc = MelSpec(sample_rate=sample_rate,
                        hop_length=hop_length,
                        n_mels=dim_in,
                        htk=True)

    # Initialize the estimation pipeline
    validation_estimator = ComboEstimator([NoteTranscriber(profile=profile),
                                           PitchListWrapper(profile=profile)])

    # Initialize the evaluation pipeline
    evaluators = [LossWrapper(),
                  MultipitchEvaluator(),
                  NoteEvaluator(results_key=tools.KEY_NOTE_ON),
                  NoteEvaluator(offset_ratio=0.2, results_key=tools.KEY_NOTE_OFF)]
    validation_evaluator = ComboEvaluator(evaluators, patterns=['loss', 'f1'])

    # Construct the MAESTRO splits
    train_split = ['train']
    val_split = ['validation']
    test_split = ['test']

    print('Loading training partition...')

    # Create a dataset corresponding to the training partition
    mstro_train = MAESTRO_V3(splits=train_split,
                             hop_length=hop_length,
                             sample_rate=sample_rate,
                             data_proc=data_proc,
                             profile=profile,
                             num_frames=num_frames,
                             reset_data=reset_data,
                             store_data=False)

    # Create a PyTorch data loader for the dataset
    train_loader = DataLoader(dataset=mstro_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True)

    print('Loading validation partition...')

    # Create a dataset corresponding to the validation partition
    mstro_val = MAESTRO_V3(splits=val_split,
                           hop_length=hop_length,
                           sample_rate=sample_rate,
                           data_proc=data_proc,
                           profile=profile,
                           num_frames=num_frames,
                           reset_data=reset_data,
                           store_data=False)

    print('Loading testing partitions...')

    # Create a dataset corresponding to the MAESTRO testing partition
    mstro_test = MAESTRO_V3(splits=test_split,
                            hop_length=hop_length,
                            sample_rate=sample_rate,
                            data_proc=data_proc,
                            profile=profile,
                            reset_data=reset_data,
                            store_data=False)

    # Initialize the MAPS testing splits as the real piano data
    test_splits = ['ENSTDkAm', 'ENSTDkCl']

    # Create a dataset corresponding to the MAPS testing partition
    # Need to reset due to HTK Mel-Spectrogram spacing
    maps_test = MAPS(splits=test_splits,
                     hop_length=hop_length,
                     sample_rate=sample_rate,
                     data_proc=data_proc,
                     profile=profile,
                     reset_data=True,
                     store_data=False)

    print('Initializing model...')

    # Initialize a new instance of the model
    onsetsframes = OnsetsFrames2(dim_in, profile, data_proc.get_num_channels(), model_complexity, True, gpu_id)
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
                         val_set=mstro_val,
                         estimator=validation_estimator,
                         evaluator=validation_evaluator)

    print('Transcribing and evaluating test partition...')

    # Add save directories to the estimators
    validation_estimator.set_save_dirs(os.path.join(root_dir, 'estimated', 'MAESTRO'), ['notes', 'pitch'])

    # Add a save directory to the evaluators and reset the patterns
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results', 'MAESTRO'))
    validation_evaluator.set_patterns(None)

    # Get the average results for the MAESTRO testing partition
    results = validate(onsetsframes, mstro_test, evaluator=validation_evaluator, estimator=validation_estimator)

    # Log the average results in metrics.json
    ex.log_scalar('MAESTRO Results', results, 0)

    # Reset the evaluator
    validation_evaluator.reset_results()

    # Update save directories for the estimators
    validation_estimator.set_save_dirs(os.path.join(root_dir, 'estimated', 'MAPS'), ['notes', 'pitch'])

    # Update save directory for the evaluators
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results', 'MAPS'))

    # Get the average results for the MAPS testing partition
    results = validate(onsetsframes, maps_test, evaluator=validation_evaluator, estimator=validation_estimator)

    # Log the average results in metrics.json
    ex.log_scalar('MAPS Results', results, 0)

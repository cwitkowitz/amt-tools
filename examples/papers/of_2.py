# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from amt_tools.models import OnsetsFrames2
from amt_tools.datasets import MAESTRO_V3, MAPS
from amt_tools.features import MelSpec

from amt_tools.train import train
from amt_tools.transcribe import ComboEstimator, \
                                 NoteTranscriber, \
                                 PitchListWrapper
from amt_tools.evaluate import ComboEvaluator, \
                               LossWrapper, \
                               MultipitchEvaluator, \
                               NoteEvaluator, \
                               validate

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

    # The fixed learning rate
    learning_rate = 6e-4

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to re-acquire ground-truth data and re-calculate
    # features (useful if testing out different parameters)
    reset_data = False

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment files
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

    # Create a Mel-scaled spectrogram feature extraction module
    # spanning all frequencies w/ length-2048 FFT and 229 bands
    data_proc = MelSpec(sample_rate=sample_rate,
                        hop_length=hop_length,
                        n_mels=229,
                        htk=True)

    # Initialize the estimation pipeline (Multi Pitch / Onsets -> Notes & Pitch List)
    validation_estimator = ComboEstimator([NoteTranscriber(profile=profile),
                                           PitchListWrapper(profile=profile)])

    # Initialize the evaluation pipeline - (Loss | Multi Pitch | Notes)
    validation_evaluator = ComboEvaluator([LossWrapper(),
                                           MultipitchEvaluator(),
                                           NoteEvaluator(results_key=tools.KEY_NOTE_ON),
                                           NoteEvaluator(offset_ratio=0.2,
                                                         results_key=tools.KEY_NOTE_OFF)])

    # Set validation patterns for logging during training
    validation_evaluator.set_patterns(['loss', 'pr', 're', 'f1'])

    # Construct the MAESTRO splits
    train_split = ['train']
    val_split = ['validation']
    test_split = ['test']

    # Keep all cached data/features here
    data_cache = os.path.join('..', '..', 'generated', 'data')

    # Create a dataset corresponding to the training partition
    mstro_train = MAESTRO_V3(base_dir=None,
                             splits=train_split,
                             hop_length=hop_length,
                             sample_rate=sample_rate,
                             num_frames=num_frames,
                             data_proc=data_proc,
                             profile=profile,
                             reset_data=reset_data,
                             store_data=False,
                             save_loc=data_cache)

    # Create a PyTorch data loader for the dataset
    train_loader = DataLoader(dataset=mstro_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True)

    # Create a dataset corresponding to the validation partition
    mstro_val = MAESTRO_V3(base_dir=None,
                           splits=val_split,
                           hop_length=hop_length,
                           sample_rate=sample_rate,
                           num_frames=num_frames,
                           data_proc=data_proc,
                           profile=profile,
                           store_data=False,
                           save_loc=data_cache)

    # Create a dataset corresponding to the MAESTRO testing partition
    mstro_test = MAESTRO_V3(base_dir=None,
                            splits=test_split,
                            hop_length=hop_length,
                            sample_rate=sample_rate,
                            num_frames=None,
                            data_proc=data_proc,
                            profile=profile,
                            store_data=False,
                            save_loc=data_cache)

    # Initialize the MAPS testing splits as the real piano data
    test_splits = ['ENSTDkAm', 'ENSTDkCl']

    # Create a dataset corresponding to the MAPS testing partition
    maps_test = MAPS(base_dir=None,
                     splits=test_splits,
                     hop_length=hop_length,
                     sample_rate=sample_rate,
                     num_frames=None,
                     data_proc=data_proc,
                     profile=profile,
                     reset_data=reset_data,
                     store_data=False,
                     save_loc=data_cache)

    print('Initializing model...')

    # Initialize a new instance of the model
    onsetsframes = OnsetsFrames2(dim_in=data_proc.get_feature_size(),
                                 profile=profile,
                                 in_channels=data_proc.get_num_channels(),
                                 model_complexity=3,
                                 detach_heads=True,
                                 device=gpu_id)
    onsetsframes.change_device()
    onsetsframes.train()

    # Initialize a new optimizer for the model parameters
    optimizer = torch.optim.Adam(onsetsframes.parameters(), learning_rate)

    print('Training model...')

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

    # Reset the evaluation patterns to log everything
    validation_evaluator.set_patterns(None)

    print('Transcribing and evaluating MAESTRO test partition...')

    # Add save directories to the estimators
    validation_estimator.set_save_dirs(os.path.join(root_dir, 'estimated', 'MAESTRO'), ['notes', 'pitch'])

    # Add a save directory to the evaluators
    validation_evaluator.set_save_dir(os.path.join(root_dir, 'results', 'MAESTRO'))

    # Get the average results for the MAESTRO testing partition
    results = validate(onsetsframes, mstro_test, evaluator=validation_evaluator, estimator=validation_estimator)

    # Log the average results in metrics.json
    ex.log_scalar('MAESTRO Results', results, 0)

    print('Transcribing and evaluating MAPS test partition...')

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

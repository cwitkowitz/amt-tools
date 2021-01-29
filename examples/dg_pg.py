# My imports
from amt_models.pipeline import train, validate
from amt_models.models import OnsetsFrames
from amt_models.features import MelSpec
from amt_models.tools import seed_everything, PianoProfile
from amt_models.datasets import MAESTRO_V2, GuitarSet
from amt_models.tools.constants import *

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import torch

ex = Experiment('Baseline Domain Generalization Piano -> Guitar')

@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 500

    # Number of training iterations to conduct
    iterations = 2000

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 40

    # Number of samples to gather for a batch
    batch_size = 8

    # The initial learning rate
    learning_rate = 5e-4

    # The id of the gpu to use, if available
    gpu_id = 1

    # Flag to control whether sampled blocks of frames should avoid splitting notes
    split_notes = False

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different feature extraction parameters
    reset_data = False

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    root_dir = 'Baseline_DG'
    root_dir = os.path.join(GEN_EXPR_DIR, root_dir)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))

@ex.automain
def onsets_frames_run(sample_rate, hop_length, num_frames, iterations, checkpoints,
                      batch_size, learning_rate, gpu_id, split_notes, reset_data, seed, root_dir):
    # Seed everything with the same seed
    seed_everything(seed)

    # Construct the MAESTRO splits
    train_split = ['train']

    # Validate and evaluate on the full GuitarSet data TODO - for now (not good practice to validate on test)
    val_split = GuitarSet.available_splits()

    # Initialize the default piano profile
    profile = PianoProfile()

    # Processing parameters
    dim_in = 229
    model_complexity = 3

    # Create the mel spectrogram data processing module
    data_proc = MelSpec(sample_rate=sample_rate,
                        n_mels=dim_in,
                        hop_length=hop_length,
                        htk=True)

    print('Loading training partition...')

    # Create a dataset corresponding to the training partition
    mstro_train = MAESTRO_V2(splits=train_split,
                             hop_length=hop_length,
                             sample_rate=sample_rate,
                             data_proc=data_proc,
                             profile=profile,
                             num_frames=num_frames,
                             split_notes=split_notes,
                             reset_data=reset_data,
                             store_data=False)

    # Create a PyTorch data loader for the dataset
    train_loader = DataLoader(dataset=mstro_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=8,
                              drop_last=True)

    print('Loading validation partition...')

    # Create a dataset corresponding to the training partition
    gset_val = GuitarSet(splits=val_split,
                         hop_length=hop_length,
                         sample_rate=sample_rate,
                         data_proc=data_proc,
                         profile=profile,
                         reset_data=reset_data)

    print('Initializing model...')

    # Initialize a new instance of the model
    of2 = OnsetsFrames(dim_in, profile, data_proc.get_num_channels(), model_complexity, gpu_id)
    of2.change_device()
    of2.train()

    # Initialize a new optimizer for the model parameters
    optimizer = torch.optim.Adam(of2.parameters(), learning_rate)

    print('Training classifier...')

    # Create a log directory for the training experiment
    model_dir = os.path.join(root_dir, 'models')

    # Train the model
    of2 = train(model=of2,
                train_loader=train_loader,
                optimizer=optimizer,
                iterations=iterations,
                checkpoints=checkpoints,
                log_dir=model_dir,
                val_set=gset_val)

    print('Transcribing and evaluating test partition...')

    estim_dir = os.path.join(root_dir, 'estimated')
    results_dir = os.path.join(root_dir, 'results')

    # Get the average results for the testing partition
    results = validate(of2, gset_val, estim_dir, results_dir)

    # Log the average results in metrics.json
    ex.log_scalar('results', results, 0)

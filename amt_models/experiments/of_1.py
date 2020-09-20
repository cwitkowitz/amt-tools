# My imports
from pipeline.transcribe import *
from pipeline.evaluate import *
from pipeline.train import *

from models.onsetsframes import *

from features.melspec import *

from tools.instrument import *

from datasets.MAPS import *

# Regular imports
from sacred.observers import FileStorageObserver
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sacred import Experiment

ex = Experiment('Onsets & Frames 1 w/ Mel Spectrogram on MAPS')

@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 16000

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
    root_dir = '_'.join([OnsetsFrames.model_name(), MAPS.dataset_name(), MelSpec.features_name()])
    root_dir = os.path.join(GEN_EXPR_DIR, root_dir)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))

@ex.automain
def onsets_frames_run(sample_rate, hop_length, num_frames, iterations, checkpoints,
                      batch_size, learning_rate, gpu_id, split_notes, reset_data, seed, root_dir):
    # Seed everything with the same seed
    seed_everything(seed)

    # Get a list of the MAPS splits
    splits = MAPS.available_splits()

    # Initialize the default piano profile
    profile = PianoProfile()

    # Initialize the testing splits as the real piano data
    test_splits = ['ENSTDkAm', 'ENSTDkCl']
    # Remove the real piano splits to get the training partition
    train_splits = splits.copy()
    for split in test_splits:
        train_splits.remove(split)

    # Processing parameters
    dim_in = 229
    model_complexity = 2

    # Create the mel spectrogram data processing module
    data_proc = MelSpec(sample_rate=sample_rate,
                        n_mels=dim_in,
                        hop_length=hop_length)

    print('Loading training partition...')

    # Create a dataset corresponding to the training partition
    maps_train = MAPS(splits=train_splits,
                      hop_length=hop_length,
                      sample_rate=sample_rate,
                      data_proc=data_proc,
                      profile=profile,
                      num_frames=num_frames,
                      split_notes=split_notes,
                      reset_data=reset_data)

    # Remove tracks in both partitions from the training partitions
    #print('Removing overlapping tracks from training partition')
    #maps_train.remove_overlapping(test_splits)

    # Create a PyTorch data loader for the dataset
    train_loader = DataLoader(dataset=maps_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=16,
                              drop_last=True)

    # TODO - validation using overlapping tracks which were removed

    print('Loading testing partition...')

    # Create a dataset corresponding to the testing partition
    maps_test = MAPS(splits=test_splits,
                     hop_length=hop_length,
                     sample_rate=sample_rate,
                     data_proc=data_proc,
                     profile=profile,
                     store_data=False)

    print('Initializing model...')

    # Initialize a new instance of the model
    onsetsframes = OnsetsFrames(dim_in, profile, model_complexity, gpu_id)
    onsetsframes.change_device()
    onsetsframes.train()

    # Initialize a new optimizer for the model parameters
    optimizer = torch.optim.Adam(onsetsframes.parameters(), learning_rate)
    # Decay the learning rate over the course of training
    scheduler = StepLR(optimizer, iterations, 0.95)

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
                         scheduler=scheduler,
                         val_set=maps_test,
                         resume=False)

    print('Transcribing and evaluating test partition...')

    estim_dir = os.path.join(root_dir, 'estimated')
    results_dir = os.path.join(root_dir, 'results')

    # Get the average results for the testing partition
    results = validate(onsetsframes, maps_test, estim_dir, results_dir)

    # Log the average results in metrics.json
    ex.log_scalar('results', results, 0)

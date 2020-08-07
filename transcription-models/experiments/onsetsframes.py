# My imports
from pipeline.transcribe import transcribe
from pipeline.evaluate import *
from pipeline.train import train

from models.onsetsframes import *

from features.melspec import MelSpec

from datasets.MAPS import *

# Regular imports
from torch.utils.data import DataLoader
from sacred.observers import FileStorageObserver
from sacred import Experiment

# TODO - clean up text output
ex = Experiment('Onsets & Frames MAPS Experiment')

@ex.config
def config():
    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    seq_length = 500

    # Number of training iterations to conduct
    iterations = 2000

    # How many training iterations in between each save/validation point - 0 to disable
    checkpoints = iterations // 20

    # Number of samples to gather for a batch
    batch_size = 8

    # The initial learning rate
    learning_rate = 5e-4

    # Minimum number of active frames required for a note
    min_note_span = 5

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to control whether sampled blocks of frames can split notes
    split_notes = False

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different parameters
    reset_data = False

    # Flag for printing extraneous information
    # TODO - add in verbose text
    verbose = False

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    root_dir = '_'.join([OnsetsFrames.model_name(), MAPS.dataset_name(), MelSpec.features_name()])
    root_dir = os.path.join(GEN_EXPR_DIR, root_dir)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))

@ex.automain
def onsets_frames_run(hop_length, seq_length, iterations, checkpoints, batch_size, learning_rate,
                      min_note_span, gpu_id, split_notes, reset_data, verbose, seed, root_dir):
    # Seed everything with the same seed
    seed_everything(seed)

    # Get a list of the GuitarSet splits
    splits = MAPS.available_splits()

    # Processing parameters
    dim_in = 229
    dim_out = PIANO_RANGE
    model_complexity = 2

    # Create the cqt data processing module
    data_proc = MelSpec(sample_rate=16000, n_mels=dim_in, n_fft=2048, hop_length=512, htk=False, norm=None)

    # Remove the hold out split to get the training partition
    train_splits = splits.copy()

    test_splits = ['ENSTDkAm', 'ENSTDkCl']
    for split in test_splits:
        train_splits.remove(split)

    print('Loading training partition...')

    # Create a data loader for this training partition of MAPS
    maps_train = MAPS(base_dir=None, splits=train_splits, hop_length=hop_length, sample_rate=16000,
                      data_proc=data_proc, frame_length=seq_length, split_notes=split_notes, reset_data=reset_data, seed=seed)
    print('Removing overlapping tracks')
    maps_train.remove_overlapping(test_splits)
    train_loader = DataLoader(maps_train, batch_size, shuffle=True, num_workers=16, drop_last=True)

    # Initialize a new instance of the model
    onsetsframes = OnsetsFrames(dim_in, dim_out, model_complexity, gpu_id)
    onsetsframes.change_device()
    onsetsframes.train()

    # Initialize a new optimizer for the model parameters
    optimizer = torch.optim.Adam(onsetsframes.parameters(), learning_rate)

    # Create a log directory for the training experiment
    model_dir = os.path.join(root_dir, 'models')

    print('Loading testing partition...')

    # Create a data loader for the validation step
    # TODO - can't aggregate slices because of notes array mismatch
    maps_val = MAPS(base_dir=None, splits=test_splits, hop_length=hop_length, sample_rate=16000,
                    data_proc=data_proc, frame_length=seq_length, reset_data=reset_data)

    print('Training classifier...')

    # Train the model
    onsetsframes = train(onsetsframes, train_loader, optimizer, iterations, checkpoints, model_dir, maps_val)
    estim_dir = os.path.join(root_dir, 'estimated')

    print('Transcribing and evaluating test partition...')

    # Create a data loader for the testing partition of MAPS
    maps_test = MAPS(base_dir=None, splits=test_splits, hop_length=hop_length, sample_rate=16000,
                     data_proc=data_proc, frame_length=None, reset_data=reset_data)

    results_dir = os.path.join(root_dir, 'results')

    # Generate predictions for the test set
    onsetsframes.eval()
    results = get_results_format()
    for track in maps_test:
        predictions = transcribe(onsetsframes, track, hop_length, 16000, min_note_span, estim_dir)
        track_results = evaluate(predictions, track, hop_length, 16000, results_dir, verbose)
        results = add_result_dicts(results, track_results)
    results = average_results(results)

    ex.log_scalar('results', results, 0)

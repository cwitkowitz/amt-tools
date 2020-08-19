# My imports
from pipeline.transcribe import transcribe
from pipeline.evaluate import *
from pipeline.train import train

from models.onsetsframes import *

from features.melspec import MelSpec

from datasets.MAESTRO import *

# Regular imports
from torch.utils.data import DataLoader
from sacred.observers import FileStorageObserver
from sacred import Experiment

# TODO - clean up text output
ex = Experiment('Onsets & Frames MAESTRO Experiment')

@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 16000

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    seq_length = 500

    # Number of training iterations to conduct
    iterations = 5000

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 20

    # Number of samples to gather for a batch
    batch_size = 8

    # The initial learning rate
    learning_rate = 5e-4

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to control whether sampled blocks of frames can split notes
    split_notes = False

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different feature extraction parameters
    reset_data = False

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    root_dir = '_'.join([OnsetsFrames.model_name(), MAESTRO_V1.dataset_name(), MelSpec.features_name()])
    root_dir = os.path.join(GEN_EXPR_DIR, root_dir)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))

@ex.automain
def onsets_frames_run(sample_rate, hop_length, seq_length, iterations, checkpoints, batch_size,
                      learning_rate, gpu_id, split_notes, reset_data, seed, root_dir):
    # Seed everything with the same seed
    seed_everything(seed)

    # Processing parameters
    dim_in = 229
    dim_out = PIANO_RANGE
    model_complexity = 3

    # Create the cqt data processing module
    data_proc = MelSpec(sample_rate=sample_rate, n_mels=dim_in, n_fft=2048, hop_length=hop_length, htk=False, norm=None)

    train_split = ['train']
    val_split = ['validation']
    test_split = ['test']

    print('Loading training partition...')

    # Create a data loader for this training partition of MAPS
    mstro_train = MAESTRO_V1(base_dir=None, splits=train_split, hop_length=hop_length, sample_rate=sample_rate,
                             data_proc=data_proc, frame_length=seq_length, split_notes=split_notes,
                             reset_data=reset_data, store_data=False)

    train_loader = DataLoader(mstro_train, batch_size, shuffle=True, num_workers=8, drop_last=True)

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
    mstro_val = MAESTRO_V1(base_dir=None, splits=val_split, hop_length=hop_length, sample_rate=sample_rate,
                           data_proc=data_proc, frame_length=seq_length, split_notes=split_notes,
                           reset_data=reset_data, store_data=False)

    print('Training classifier...')

    # Train the model
    onsetsframes = train(onsetsframes, train_loader, optimizer, iterations,
                         checkpoints, model_dir, mstro_val, resume=True)
    estim_dir = os.path.join(root_dir, 'estimated')

    print('Transcribing and evaluating test partition...')

    # Create a data loader for the testing partition of MAPS
    mstro_test = MAESTRO_V1(base_dir=None, splits=test_split, hop_length=hop_length, sample_rate=sample_rate,
                            data_proc=data_proc, frame_length=None, reset_data=reset_data, store_data=False)

    results_dir = os.path.join(root_dir, 'results')

    # Generate predictions for the test set
    onsetsframes.eval()
    results = get_results_format()
    for track_id in mstro_test.tracks:
        track = mstro_test.get_track_data(track_id)
        predictions = transcribe(onsetsframes, track, estim_dir)
        track_results = evaluate(predictions, track, results_dir, True)
        results = add_result_dicts(results, track_results)
    results = average_results(results)

    ex.log_scalar('results', results, 0)

# My imports
from pipeline import *
from models import *
from features import *
from tools import *
from datasets import *

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

ex = Experiment('Onsets & Frames 2 w/ Mel Spectrogram on MAPS')


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 500

    # Number of training iterations to conduct
    iterations = 10000

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 200

    # Number of samples to gather for a batch
    batch_size = 15

    # The initial learning rate
    learning_rate = 5e-4

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to control whether sampled blocks of frames should avoid splitting notes
    split_notes = False

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different feature extraction parameters
    reset_data = False

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    root_dir = '_'.join([OnsetsFrames.model_name(), MAESTRO_V2.dataset_name(), MelSpec.features_name()])
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
    val_split = ['validation']
    test_split = ['test']

    # Initialize the default piano profile
    profile = PianoProfile()

    # Processing parameters
    dim_in = 229
    dim_out = profile.get_range_len()
    model_complexity = 3

    # Create the mel spectrogram data processing module
    data_proc = MelSpec(sample_rate=sample_rate,
                        n_mels=dim_in,
                        hop_length=hop_length,
                        htk=True)

    print('Loading training partition...')

    # TODO - 119 MelSpec 2004 vs 120 gt 2004?
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
                              num_workers=0,
                              drop_last=True)

    print('Loading validation partition...')

    # Create a dataset corresponding to the validation partition
    mstro_val = MAESTRO_V2(splits=val_split,
                           hop_length=hop_length,
                           sample_rate=sample_rate,
                           data_proc=data_proc,
                           profile=profile,
                           num_frames=num_frames,
                           split_notes=split_notes,
                           store_data=False)

    print('Loading testing partition...')

    # Create a dataset corresponding to the testing partition
    mstro_test = MAESTRO_V2(splits=test_split,
                            hop_length=hop_length,
                            sample_rate=sample_rate,
                            data_proc=data_proc,
                            profile=profile,
                            store_data=False)

    print('Initializing model...')

    # Initialize a new instance of the model
    onsetsframes = OnsetsFrames(dim_in, profile, data_proc.get_num_channels(), model_complexity, gpu_id)
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
                         val_set=mstro_val)

    print('Transcribing and evaluating test partition...')

    estim_dir = os.path.join(root_dir, 'estimated')
    results_dir = os.path.join(root_dir, 'results')

    # Get the average results for the testing partition
    results = validate(onsetsframes, mstro_test, estim_dir, results_dir)

    # Log the average results in metrics.json
    ex.log_scalar('results', results, 0)

# My imports
from amt_models.pipeline.transcribe import transcribe
from amt_models.pipeline.evaluate import *
from amt_models.pipeline.train import train

from amt_models.models.onsetsframes import *

from amt_models.features.melspec import MelSpec

from amt_models.datasets.GuitarSet import *

# Regular imports
from torch.utils.data import DataLoader
from sacred.observers import FileStorageObserver
from sacred import Experiment

# TODO - multi-threading
# TODO - clean up text output - actually remove verbose except for evaluate
ex = Experiment('6-Fold Guitarset Using OF Model')

@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 16000

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    seq_length = 500

    # Number of training iterations to conduct
    iterations = 2000

    # How many training iterations in between each save/validation point - 0 to disable
    checkpoints = 20

    # Number of samples to gather for a batch
    batch_size = 30

    # The initial learning rate
    learning_rate = 5e-4

    # The id of the gpu to use, if available
    gpu_id = 1

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different parameters
    reset_data = False

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    root_dir = '_'.join([OnsetsFrames.model_name(), GuitarSet.dataset_name(), MelSpec.features_name()])
    root_dir = os.path.join(GEN_EXPR_DIR, root_dir)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))

@ex.automain
def tabcnn_cross_val(sample_rate, hop_length, seq_length, iterations, checkpoints,
                     batch_size, learning_rate, gpu_id, reset_data, seed, root_dir):
    # Seed everything with the same seed
    seed_everything(seed)

    # Get a list of the GuitarSet splits
    splits = GuitarSet.available_splits()

    # Processing parameters
    dim_in = 229
    dim_out = NUM_STRINGS * (NUM_FRETS + 2)
    model_complexity = 2

    # Create the cqt data processing module
    data_proc = MelSpec(sample_rate=sample_rate, n_mels=dim_in, n_fft=2048, hop_length=hop_length, htk=False, norm=None)

    # Perform each fold of cross-validation
    for k in range(6):
        # Determine the name of the split being removed
        hold_out = '0' + str(k)

        print(f'Fold {hold_out}:')

        # Remove the hold out split to get the training partition
        train_splits = splits.copy()
        train_splits.remove(hold_out)

        print('Loading training partition...')

        # Create a data loader for this training partition of GuitarSet
        gset_train = GuitarSet(base_dir=None, splits=train_splits, hop_length=hop_length, sample_rate=sample_rate,
                               data_proc=data_proc, num_frames=seq_length, split_notes=False, reset_data=reset_data, seed=seed)
        train_loader = DataLoader(gset_train, batch_size, shuffle=True, num_workers=16, drop_last=True)

        # Initialize a new instance of the model
        of1 = OnsetsFrames(dim_in, dim_out, model_complexity, gpu_id)
        of1.onsets[-1] = MLSoftmax(of1.dim_lm1, NUM_STRINGS, NUM_FRETS + 2, 'onsets')
        of1.pianoroll[-1] = MLSoftmax(of1.dim_am, NUM_STRINGS, NUM_FRETS + 2)
        of1.adjoin[-1] = MLSoftmax(of1.dim_lm2, NUM_STRINGS, NUM_FRETS + 2, 'tabs')

        of1.change_device()
        of1.train()

        # Initialize a new optimizer for the model parameters
        optimizer = torch.optim.Adam(of1.parameters(), learning_rate)

        # Create a log directory for the training experiment
        model_dir = os.path.join(root_dir, 'models', 'fold-' + str(k))

        print('Loading validation partition...')

        # Create a data loader for this testing partition of GuitarSet
        test_splits = [hold_out]
        gset_val = GuitarSet(base_dir=None, splits=test_splits, hop_length=hop_length, sample_rate=sample_rate,
                             data_proc=data_proc, num_frames=seq_length, split_notes=False, reset_data=reset_data)

        print('Training classifier...')

        # Train the model
        of1 = train(of1, train_loader, optimizer, iterations, checkpoints, model_dir, resume=True, val_set=gset_val)

        estim_dir = os.path.join(root_dir, 'estimated')

        print('Loading testing partition...')

        test_splits = [hold_out]
        gset_test = GuitarSet(base_dir=None, splits=test_splits, hop_length=hop_length, sample_rate=sample_rate,
                              data_proc=data_proc, num_frames=None, split_notes=False, reset_data=reset_data)

        print('Transcribing and evaluating test partition...')

        results_dir = os.path.join(root_dir, 'results')

        # Generate predictions for the test set
        of1.eval()
        fold_results = get_results_format()
        for track_id in gset_test.tracks:
            track = gset_test.get_track_data(track_id)
            predictions = transcribe(of1, track, estim_dir)
            track_results = evaluate(predictions, track, results_dir, False)
            fold_results = add_result_dicts(fold_results, track_results)
        fold_results = average_results(fold_results)

        ex.log_scalar('fold_results', fold_results, k)

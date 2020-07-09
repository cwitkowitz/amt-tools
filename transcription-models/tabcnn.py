# My imports
from pipeline.transcribe import transcribe
from pipeline.evaluate import evaluate
from pipeline.train import train

from tools.datasets import *
from tools.dataproc import *
from tools.models import *
from tools.utils import *

# Regular imports
from torch.utils.data import DataLoader
from sacred.observers import FileStorageObserver
from sacred import Experiment

# TODO - clean up text output
ex = Experiment('6-Fold TabCNN')

@ex.config
def config():
    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    seq_length = 50

    # Number of training iterations to conduct
    iterations = 2000

    # How many training iterations in between each save point
    checkpoints = iterations // 10

    # Number of samples to gather for a batch
    batch_size = 128

    # The initial learning rate
    learning_rate = 1e0

    # Minimum number of active frames required for a note
    min_note_span = 5

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different parameters
    reset_data = False

    # Flag for printing extraneous information
    # TODO - add in verbose text
    verbose = False

    # The random seed for this experiment
    seed = SEED

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    root_dir = '_'.join([TabCNN.model_name(), GuitarSet.dataset_name(), CQT.features_name()])
    root_dir = os.path.join(GENR_DIR, root_dir)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))

@ex.automain
def tabcnn_cross_val(hop_length, seq_length, iterations, checkpoints, batch_size, learning_rate,
                     min_note_span, gpu_id, reset_data, verbose, seed, root_dir):
    # Seed everything with the same seed
    seed_everything(seed)

    # Get a list of the GuitarSet splits
    splits = GuitarSet.available_splits()

    # Processing parameters
    dim_in = 192
    dim_out = NUM_STRINGS * (NUM_FRETS + 2)
    model_complexity = 1

    # Create the cqt data processing module
    data_proc = CQT(hop_length, None, dim_in, 24)

    # Initialize a list to hold the results from each fold
    fold_results = []

    # Perform each fold of cross-validation
    for k in range(6):
        # Determine the name of the split being removed
        hold_out = '0' + str(k)

        # Remove the hold out split to get the training partition
        train_splits = splits.copy()
        train_splits.remove(hold_out)

        # Create a data loader for this training partition of GuitarSet
        gset_train = GuitarSet(None, train_splits, hop_length, data_proc, seq_length, reset_data)
        train_loader = DataLoader(gset_train, batch_size, shuffle=True, num_workers=0, drop_last=True)

        # Initialize a new instance of the model
        tabcnn = TabCNN(dim_in, dim_out, model_complexity, gpu_id)
        tabcnn.change_device()
        tabcnn.train()

        # Initialize a new optimizer for the model parameters
        optimizer = torch.optim.Adadelta(tabcnn.parameters(), learning_rate)

        # Create a log directory for the training experiment
        model_dir = os.path.join(root_dir, 'models', 'fold-' + str(k))

        # Train the model
        tabcnn = train(tabcnn, train_loader, optimizer, iterations, checkpoints, model_dir)

        # Create a data loader for this testing partition of GuitarSet
        test_splits = [hold_out]
        gset_test = GuitarSet(None, test_splits, hop_length, data_proc, None, reset_data)
        test_loader = DataLoader(gset_test, 1, shuffle=False, num_workers=0, drop_last=False)

        estim_dir = os.path.join(root_dir, 'estimated')

        # Generate predictions for the test set
        tabcnn.eval()
        transcribe(tabcnn, test_loader, hop_length, min_note_span, estim_dir)

        results_dir = os.path.join(root_dir, 'results')

        # Evaluate the test set predictions
        fold_results += [evaluate(test_loader, hop_length, estim_dir, results_dir, verbose)]

        ex.log_scalar('fold_results', fold_results[k], k)

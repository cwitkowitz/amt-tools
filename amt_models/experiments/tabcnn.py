# My imports
from pipeline.transcribe import *
from pipeline.evaluate import *
from pipeline.train import *

from datasets.GuitarSet import *

from tools.instrument import *

from features.cqt import *

from models.tabcnn import *

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

ex = Experiment('TabCNN w/ CQT on GuitarSet 6-fold Cross Validation')

@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 200

    # Number of training iterations to conduct
    iterations = 1000

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 20

    # Number of samples to gather for a batch
    batch_size = 30

    # The initial learning rate
    learning_rate = 1.0

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different parameters
    reset_data = False

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    root_dir = '_'.join([TabCNN.model_name(), GuitarSet.dataset_name(), CQT.features_name()])
    root_dir = os.path.join(GEN_EXPR_DIR, root_dir)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))

@ex.automain
def tabcnn_cross_val(sample_rate, hop_length, num_frames, iterations, checkpoints,
                     batch_size, learning_rate, gpu_id, reset_data, seed, root_dir):
    # Seed everything with the same seed
    seed_everything(seed)

    # Get a list of the GuitarSet splits
    splits = GuitarSet.available_splits()

    # Initialize the default guitar profile
    profile = GuitarProfile()

    # Processing parameters
    dim_in = 192
    model_complexity = 1

    # Create the cqt data processing module
    data_proc = CQT(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_bins=dim_in,
                    bins_per_octave=24)

    # Perform each fold of cross-validation
    for k in range(6):
        # Determine the name of the split being removed
        hold_out = '0' + str(k)

        print('--------------------')
        print(f'Fold {hold_out}:')
        print('Loading training partition...')

        # Remove the hold out split to get the training partition
        train_splits = splits.copy()
        train_splits.remove(hold_out)

        # Create a dataset corresponding to the training partition
        gset_train = GuitarSet(splits=train_splits,
                               hop_length=hop_length,
                               sample_rate=sample_rate,
                               data_proc=data_proc,
                               profile=profile,
                               num_frames=num_frames,
                               reset_data=reset_data)

        # Create a PyTorch data loader for the dataset
        train_loader = DataLoader(dataset=gset_train,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=16,
                                  drop_last=True)

        print('Loading testing partition...')

        # Create a dataset corresponding to the training partition
        test_splits = [hold_out]
        gset_test = GuitarSet(splits=test_splits,
                              hop_length=hop_length,
                              sample_rate=sample_rate,
                              data_proc=data_proc,
                              profile=profile,
                              store_data=False)

        print('Initializing model...')

        # Initialize a new instance of the model
        tabcnn = TabCNN(dim_in, profile, model_complexity, gpu_id)
        tabcnn.change_device()
        tabcnn.train()

        # Initialize a new optimizer for the model parameters
        optimizer = torch.optim.Adadelta(tabcnn.parameters(), learning_rate)

        print('Training model...')

        # Create a log directory for the training experiment
        model_dir = os.path.join(root_dir, 'models', 'fold-' + str(k))

        # Train the model
        tabcnn = train(model=tabcnn,
                       train_loader=train_loader,
                       optimizer=optimizer,
                       iterations=iterations,
                       checkpoints=checkpoints,
                       log_dir=model_dir,
                       val_set=gset_test)

        print('Transcribing and evaluating test partition...')

        estim_dir = os.path.join(root_dir, 'estimated')
        results_dir = os.path.join(root_dir, 'results')

        # Get the average results for the fold
        fold_results = validate(tabcnn, gset_test, estim_dir, results_dir)

        # Log the average results for the fold in metrics.json
        ex.log_scalar('fold_results', fold_results, k)

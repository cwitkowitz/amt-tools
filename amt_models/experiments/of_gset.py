# My imports
from pipeline.transcribe import *
from pipeline.evaluate import *
from pipeline.train import *

from models.onsetsframes import *

from datasets.GuitarSet import *

from features.melspec import *

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

# TODO - multi-threading
ex = Experiment('Onsets & Frames w/ Mel Spectrogram on GuitarSet')

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
def tabcnn_cross_val(sample_rate, hop_length, num_frames, iterations, checkpoints,
                     batch_size, learning_rate, gpu_id, reset_data, seed, root_dir):
    # Seed everything with the same seed
    seed_everything(seed)

    # Get a list of the GuitarSet splits
    splits = GuitarSet.available_splits()

    # Processing parameters
    dim_in = 229
    dim_out = NUM_STRINGS * (NUM_FRETS + 2)
    model_complexity = 2

    # Create the mel spectrogram data processing module
    data_proc = MelSpec(sample_rate=sample_rate,
                        n_mels=dim_in,
                        hop_length=hop_length)

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
                              split_notes=False)

        print('Initializing model...')

        # Initialize a new instance of the model and exchange the
        # logistic banks for group softmax layers
        of1 = OnsetsFrames(dim_in, dim_out, model_complexity, gpu_id)
        of1.onsets[-1] = SoftmaxGroups(of1.dim_lm1, NUM_STRINGS, NUM_FRETS + 2, 'onsets')
        of1.pianoroll[-1] = SoftmaxGroups(of1.dim_am, NUM_STRINGS, NUM_FRETS + 2)
        of1.adjoin[-1] = SoftmaxGroups(of1.dim_lm2, NUM_STRINGS, NUM_FRETS + 2, 'tabs')
        of1.change_device()
        of1.train()

        # Initialize a new optimizer for the model parameters
        optimizer = torch.optim.Adam(of1.parameters(), learning_rate)

        print('Training classifier...')

        # Create a log directory for the training experiment
        model_dir = os.path.join(root_dir, 'models', 'fold-' + str(k))

        # Train the model
        of1 = train(model=of1,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    iterations=iterations,
                    checkpoints=checkpoints,
                    log_dir=model_dir)

        print('Transcribing and evaluating test partition...')

        estim_dir = os.path.join(root_dir, 'estimated')
        results_dir = os.path.join(root_dir, 'results')

        # Put the model in evaluation mode
        of1.eval()

        # Create a dictionary to hold the evaluation results
        fold_results = get_results_format()

        # Loop through the testing track ids
        for track_id in gset_test.tracks:
            # Obtain the track data
            track = gset_test.get_track_data(track_id)
            # Transcribe the track
            predictions = transcribe(of1, track, estim_dir)
            # Evaluate the predictions
            track_results = evaluate(predictions, track, results_dir)
            # Add the results to the dictionary
            fold_results = add_result_dicts(fold_results, track_results)

        # Average the results from all tracks
        fold_results = average_results(fold_results)

        # Log the average results for the fold in metrics.json
        ex.log_scalar('fold_results', fold_results, k)

# My imports
from amt_models.models import OnsetsFrames, LanguageModel, SoftmaxGroups
from amt_models.datasets import GuitarSet
from amt_models.features import LHVQT
from amt_models import train, validate

import amt_models.tools as tools

# Regular imports
from sacred.observers import FileStorageObserver
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sacred import Experiment

import torch.nn as nn
import torch
import os

EX_NAME = '_'.join([OnsetsFrames.model_name(),
                    GuitarSet.dataset_name(),
                    LHVQT.features_name(),
                    'vd_test'])

ex = Experiment('Onsets & Frames w/ Learnable Filterbank on GuitarSet')

desc = 'Same parameters as baseline experiment with variational dropout conv1d in hilbert variant'


class OnsetsFramesLHVQTVariationalDropout(OnsetsFrames):
    # TODO - split into Softmax, LHVQT, and Var Drop variant?
    def __init__(self, dim_in, profile, in_channels, lhvqt, model_complexity=2, device='cpu'):
        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Exchange the logistic banks for group softmax layers
        self.onsets[-1] = SoftmaxGroups(self.dim_lm, profile, 'onsets')
        self.pianoroll[-1] = SoftmaxGroups(self.dim_am, profile)
        self.dim_aj = self.onsets[-1].dim_out + self.pianoroll[-1].dim_out
        self.adjoin = nn.Sequential(
            LanguageModel(self.dim_aj, self.dim_lm),
            SoftmaxGroups(self.dim_lm, profile, 'pitch')
        )

        # Append the filterbank learning module to the front of the model
        self.feat_ext.add_module('fb', lhvqt.lhvqt)
        self.feat_ext.add_module('rl', nn.ReLU())

    def post_proc(self, batch):
        preds = super().post_proc(batch)

        if 'loss' in preds.keys():
            fb_module = torch.nn.Sequential(*list(self.feat_ext.fb.tfs.children()))[0].time_conv
            preds['loss'] = preds['loss'] + fb_module.kld(fb_module.log_alpha)

        return preds


def visualize(model, i=None):
    vis_dir = os.path.join(tools.DEFAULT_VISUALIZATION_DIR, EX_NAME)

    # TODO - overwrites fold-wise

    if i is not None:
        vis_dir = os.path.join(vis_dir, f'checkpoint-{i}')

    model.feat_ext.fb.plot_time_weights(vis_dir)
    model.feat_ext.fb.plot_freq_weights(vis_dir)


@ex.config
def config():
    # Number of samples per second of audio
    sample_rate = 22050

    # Number of samples between frames
    hop_length = 512

    # Number of consecutive frames within each example fed to the model
    num_frames = 200

    # Number of training iterations to conduct
    iterations = 10000

    # How many equally spaced save/validation checkpoints - 0 to disable
    checkpoints = 200

    # Number of samples to gather for a batch
    batch_size = 20

    # The initial learning rate
    learning_rate = 5e-4

    # The id of the gpu to use, if available
    gpu_id = 0

    # Flag to re-acquire ground-truth data and re-calculate-features
    # This is useful if testing out different parameters
    reset_data = False

    # The random seed for this experiment
    seed = 0

    # Create the root directory for the experiment to hold train/transcribe/evaluate materials
    root_dir = os.path.join(tools.DEFAULT_EXPERIMENTS_DIR, EX_NAME)
    os.makedirs(root_dir, exist_ok=True)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def tabcnn_cross_val(sample_rate, hop_length, num_frames, iterations, checkpoints,
                     batch_size, learning_rate, gpu_id, reset_data, seed, root_dir):
    # Seed everything with the same seed
    tools.seed_everything(seed)

    # Get a list of the GuitarSet splits
    splits = GuitarSet.available_splits()

    # Initialize the default guitar profile
    profile = tools.GuitarProfile()

    # Processing parameters
    dim_in = 8 * 24
    model_complexity = 3

    # Perform each fold of cross-validation
    for k in range(6):
        # Determine the name of the split being removed
        test_hold_out = '0' + str(k)
        val_hold_out = '0' + str(5 - k)

        print('--------------------')
        print(f'Fold {test_hold_out}:')

        # Remove the hold out split to get the training partition
        train_splits = splits.copy()
        train_splits.remove(test_hold_out)
        train_splits.remove(val_hold_out)

        val_splits = [val_hold_out]
        test_splits = [test_hold_out]

        from lhvqt.lvqt_hilb import LVQT as lower
        from lhvqt.lhvqt import LHVQT as upper
        # Initialize learnable filterbank data processing module
        lhvqt = LHVQT(sample_rate=sample_rate,
                      hop_length=hop_length,
                      lhvqt=upper,
                      lvqt=lower,
                      n_bins=dim_in,
                      bins_per_octave=24,
                      harmonics=[1],
                      random=True)
        data_proc = lhvqt

        print('Loading training partition...')

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
                                  num_workers=0,
                                  drop_last=True)

        print('Loading validation partition...')

        # Create a dataset corresponding to the validation partition
        gset_val = GuitarSet(splits=val_splits,
                             hop_length=hop_length,
                             sample_rate=sample_rate,
                             data_proc=data_proc,
                             profile=profile)

        print('Loading testing partition...')

        # Create a dataset corresponding to the training partition
        gset_test = GuitarSet(splits=test_splits,
                              hop_length=hop_length,
                              sample_rate=sample_rate,
                              data_proc=data_proc,
                              profile=profile,
                              store_data=False)

        print('Initializing model...')

        # Initialize a new instance of the model
        of1 = OnsetsFramesLHVQTVariationalDropout(dim_in, profile, data_proc.get_num_channels(), lhvqt, model_complexity, gpu_id)
        of1.change_device()
        of1.train()

        # Initialize a new optimizer for the model parameters
        optimizer = torch.optim.Adam(of1.parameters(), learning_rate)

        # Decay the learning rate over the course of training
        scheduler = StepLR(optimizer, iterations, 0.99)

        print('Training classifier...')

        # Create a log directory for the training experiment
        model_dir = os.path.join(root_dir, 'models', 'fold-' + str(k))

        visualize(of1, 0)

        # Train the model
        of1 = train(model=of1,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    iterations=iterations,
                    checkpoints=checkpoints,
                    log_dir=model_dir,
                    val_set=gset_val,
                    vis_fnc=visualize)
                    #scheduler=scheduler)

        print('Transcribing and evaluating test partition...')

        estim_dir = os.path.join(root_dir, 'estimated')
        results_dir = os.path.join(root_dir, 'results')

        # Get the average results for the fold
        fold_results = validate(of1, gset_test, estim_dir, results_dir)

        # Log the average results for the fold in metrics.json
        ex.log_scalar('fold_results', fold_results, k)

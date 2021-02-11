# My imports
from amt_models.pipeline.train import train, validate
from amt_models.pipeline.transcribe import Estimator
from amt_models.pipeline.evaluate import MultipitchEvaluator
from amt_models.models.onsetsframes import OnsetsFrames, LanguageModel
from amt_models.models.common import SoftmaxGroups
from amt_models.features.vqt import VQT
from amt_models.tools.utils import seed_everything
from amt_models.tools.instrument import GuitarProfile
from amt_models.datasets.GuitarSet import GuitarSet
from amt_models.tools.constants import *

# Regular imports
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment

import numpy as np
import torch.nn as nn
import torch

ex = Experiment('Onsets & Frames w/ VQT on GuitarSet')


class OnsetsFramesExperimental(OnsetsFrames):
    def __init__(self, dim_in, profile, in_channels, model_complexity=2, device='cpu'):
        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        """
        self.onsets[-1] = SoftmaxGroups(self.dim_lm, profile, 'onsets')
        self.pianoroll[-1] = SoftmaxGroups(self.dim_am, profile)
        self.dim_aj = self.onsets[-1].dim_out + self.pianoroll[-1].dim_out
        """

        # Exchange the final logistic bank (multipitch) for group softmax layer
        self.adjoin = nn.Sequential(
            LanguageModel(self.dim_aj, self.dim_lm),
            SoftmaxGroups(self.dim_lm, profile, 'pitch')
        )

    def forward(self, feats):
        multipitch_label = self.pianoroll[-1].tag
        preds = self.pianoroll(feats)
        multipitch = preds[multipitch_label]

        onsets_label = self.onsets[-1].tag
        preds.update(self.onsets(feats))
        onsets = preds[onsets_label]

        joint = torch.cat((onsets, multipitch), -1)
        preds.update(self.adjoin(joint))

        return preds

    def post_proc(self, batch):
        preds = batch['preds']

        onsets_layer = self.onsets[-1]
        # TODO - can reuse everything and just add tablature stuff if I take multipitch from end of adjoin
        multipitch_layer = self.pianoroll[-1]
        tablature_layer = self.adjoin[-1]

        onsets_label = onsets_layer.tag
        multipitch_label = multipitch_layer.tag
        tablature_label = tablature_layer.tag

        onsets = preds[onsets_label]
        multipitch = preds[multipitch_label]
        tablature = preds[tablature_label]

        loss = None

        reference_tablature = batch[tablature_label]
        reference_multipitch = np.max(tabs_to_multi_pianoroll(reference_tablature.cpu().detach().numpy(), self.profile), axis=-3)
        reference_multipitch = torch.Tensor(reference_multipitch).to(self.device)
        reference_onsets = get_onsets(reference_multipitch, self.profile)

        onsets_loss = onsets_layer.get_loss(onsets, reference_onsets)
        multipitch_loss = multipitch_layer.get_loss(multipitch, reference_multipitch)
        tablature_loss = tablature_layer.get_loss(tablature, reference_tablature)

        loss = onsets_loss + multipitch_loss + tablature_loss

        preds.update({
            'loss': loss
        })

        preds[onsets_label] = onsets_layer.finalize_output(onsets)
        preds[multipitch_label] = multipitch_layer.finalize_output(multipitch)
        preds[tablature_label] = tablature_layer.finalize_output(tablature)

        return preds

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
    root_dir = '_'.join([OnsetsFrames.model_name(), GuitarSet.dataset_name(), VQT.features_name(), 'multipitch'])
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
    dim_in = 8 * 24
    model_complexity = 3

    # Create the mel spectrogram data processing module
    data_proc = VQT(sample_rate=sample_rate,
                    hop_length=hop_length,
                    n_bins=dim_in,
                    bins_per_octave=24)

    # Patterns for score logging
    patterns = ['loss', 'f1-score']
    val_eval = MultipitchEvaluator(profile=profile)
    test_eval = MultipitchEvaluator(profile=profile)

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
                             profile=profile,
                             store_data=True)

        print('Loading testing partition...')

        # Create a dataset corresponding to the training partition
        gset_test = GuitarSet(splits=test_splits,
                              hop_length=hop_length,
                              sample_rate=sample_rate,
                              data_proc=data_proc,
                              profile=profile,
                              split_notes=False)

        print('Initializing model...')

        # Initialize a new instance of the model
        model = OnsetsFramesExperimental(dim_in, profile, data_proc.get_num_channels(), model_complexity, gpu_id)
        model.change_device()
        model.train()

        # Initialize a new optimizer for the model parameters
        optimizer = torch.optim.Adam(model.parameters(), learning_rate)

        print('Training classifier...')

        # Create a log directory for the training experiment
        model_dir = os.path.join(root_dir, 'models', 'fold-' + str(k))

        # Train the model
        model = train(model=model,
                      train_loader=train_loader,
                      optimizer=optimizer,
                      iterations=iterations,
                      checkpoints=checkpoints,
                      log_dir=model_dir,
                      val_set=gset_val,
                      evaluator=val_eval)

        print('Transcribing and evaluating test partition...')

        estim_dir = os.path.join(root_dir, 'estimated')
        results_dir = os.path.join(root_dir, 'results')

        # Get the average results for the fold
        fold_results = validate(model, gset_test,
                                evaluator=test_eval)

        # Log the average results for the fold in metrics.json
        ex.log_scalar('fold_results', fold_results, k)

    # TODO - average metrics in metrics.json

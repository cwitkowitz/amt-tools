# My imports
# None of my imports used

# Regular imports
from abc import abstractmethod
from torch import nn

import torch.nn.functional as F
import torch


class TranscriptionModel(nn.Module):
    """
    Implements a generic music transcription model.
    """

    def __init__(self, dim_in, dim_out, model_complexity=1, device='cpu'):
        """
        Initialize parameters common to all models as model fields and instantiate
        model as a PyTorch processing Module.

        Parameters
        ----------
        dim_in : int
          Dimensionality of framewise input vectors along the frequency axis
        dim_out : int
          Dimensionality of framewise output vectors along the frequency axis
        model_complexity : int, optional (default 1)
          Scaling parameter for sizes of model's components
        device : string, optional (default /'cpu/')
          Device with which to perform processing steps
        """

        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.model_complexity = model_complexity
        self.device = device

    def change_device(self, device=None):
        """
        Change the device and load the model onto the new device.

        Parameters
        ----------
        device : string or None, optional (default None)
          Device to load model onto
        """

        if device is None:
            # If the function is called without a device, use the current device
            device = self.device

        # Create the appropriate device object
        device = torch.device(f'cuda:{device}'
                              if torch.cuda.is_available() else 'cpu')

        # Change device field
        self.device = device
        # Load the transcription model onto the device
        self.to(self.device)

    @abstractmethod
    def pre_proc(self, batch):
        """
        Perform necessary pre-processing steps for the transcription model.

        Parameters
        ----------
        batch : dict
          Dictionary containing all relevant fields for a group of tracks - if a single
          track is to be transcribed, it must be organized as a batch of size 1
        """

        return NotImplementedError

    @abstractmethod
    def forward(self, feats):
        """
        Perform the main processing steps for the transcription model.

        Parameters
        ----------
        feats : Tensor (B x C x H x W)
          input features for a batch of tracks,
          TODO - check that this is accurate for what I have so far
          B - batch size,
          C - number of channels,
          H - dimensionality across vertical axis,
          W - dimensionality across horizontal axis
        """

        return NotImplementedError

    @abstractmethod
    def post_proc(self, batch):
        """
        Perform necessary post-processing steps for the transcription model.

        Parameters
        ----------
        batch : dict
          Dictionary containing all relevant fields, including model output, for a group
          of tracks - if a single track is to be transcribed, it must be organized as a
          batch of size 1
        """

        return NotImplementedError

    def run_on_batch(self, batch):
        """
        Perform all processing steps of the transcription model on a batch.

        Parameters
        ----------
        batch : dict
          Dictionary containing all relevant fields for a group of tracks - if a single
          track is to be transcribed, it must be organized as a batch of size 1

        Returns
        ----------
        preds : dict
          Dictionary containing loss and relevant predictions for a group of tracks
        """

        # Pre-process batch
        batch = self.pre_proc(batch)

        # Obtain the model output for the batch of features
        batch['out'] = self(batch['feats'])

        # Post-process batch,
        preds = self.post_proc(batch)

        return preds

    # TODO - special_steps() function
    # clip_grad_norm_(model.parameters(), 3)

    @classmethod
    def model_name(cls):
        """
        Retrieve an appropriate tag, the class name, for the model.

        Returns
        ----------
        tag : str
          TODO - make sure this is a string
          Name of the child class calling the function
        """

        tag = cls.__name__

        return tag

# TODO - convert reference and prediction to expected data types within get_loss function
class OutputLayer(nn.Module):
    """
    Implements a generic output layer for transcription models.
    """

    def __init__(self, dim_in, dim_out):
        """
        Initialize parameters common to all output layers as model fields
        and instantiate layers as a PyTorch processing Module.

        Parameters
        ----------
        dim_in : int
          Dimensionality of input features
        dim_out : int
          Dimensionality of output vectors
        """

        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

    @abstractmethod
    def forward(self, feats):
        """
        Perform the main processing steps for the output layer.

        Parameters
        ----------
        feats : Tensor (B x F x T)
          input features for a batch of tracks,
          TODO - check that this is accurate for what I have so far
          B - batch size,
          F - dimensionality of input features,
          T - number of time steps
        """

        return NotImplementedError

    @abstractmethod
    def get_loss(self, output, reference):
        """
        Perform the loss calculation at the output layer.

        Parameters
        ----------
        output : Tensor (B x F x T)
          output vectors for a batch of tracks,
          TODO - check that this is accurate for what I have so far
          B - batch size,
          F - dimensionality of output features,
          T - number of time steps
        reference : Tensor (B x F x T)
          ground-truth for a batch of tracks,
          TODO - check that this is accurate for what I have so far
          B - batch size,
          F - dimensionality of output features,
          T - number of time steps
        """

        return NotImplementedError


class MLSoftmax(OutputLayer):
    """
    Implements a multi-label softmax output layer designed to produce tablature,
    by treating each degree of freedom of a polyphonic instrument as a separate
    softmax problem.

    A straightforward example could correspond to a guitar with 6 degrees of
    freedom (the strings), where the output of each string is the softmax operation
    across the possibilities (the string's fretting).
    """

    def __init__(self, dim_in, num_dofs=6, num_poss=22):
        """
        Initialize fields of the multi-label softmax layer.

        Parameters
        ----------
        dim_in : int
          Dimensionality of input features
        num_dofs : int
          Number of degrees of freedom on instrument, e.g. number of strings
        num_poss : int
          Number of possibilities for each degree of freedom, e.g. number of
          frets + 2 (for open string and no activity possibilities)
        """

        self.num_dofs = num_dofs
        self.num_poss = num_poss

        dim_out = self.num_dofs * self.num_poss

        super().__init__(dim_in, dim_out)

        self.output_layer = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, feats):
        """
        Perform the main processing steps for the output layer.

        Parameters
        ----------
        feats : Tensor (B x F x T)
          input features for a batch of tracks,
          TODO - check that this is accurate for what I have so far
          B - batch size,
          F - dimensionality of input features,
          T - number of time steps
        """

        tabs = self.output_layer(feats)
        return tabs

    def get_loss(self, output, reference):
        """
        Compute the cross entropy softmax loss for each string independently.

        Parameters
        ----------
        output : Tensor (B x F x T)
          output vectors for a batch of tracks,
          TODO - check that this is accurate for what I have so far
          B - batch size,
          F - dimensionality of output features,
          T - number of time steps
        reference : Tensor (B x F x T)
          ground-truth for a batch of tracks,
          TODO - check that this is accurate for what I have so far
          B - batch size,
          F - dimensionality of output features,
          T - number of time steps
        """

        reference = reference.flatten().long()
        loss = F.cross_entropy(output.float(), reference, reduction='none')
        return loss


class MLLogistic(OutputLayer):
    """
    Implements a multi-label logistic output layer designed to produce key activity,
    or more generally, quantized pitch activity.

    A straightforward example could correspond to a keyboard with 88 keys,
    where the output of each key is the sigmoid operation indicating whether
    or not the key is active.
    """

    def __init__(self, dim_in, num_keys):
        """
        Initialize fields of the multi-label logistic layer.

        Parameters
        ----------
        dim_in : int
          Dimensionality of input features
        num_keys : int
          Total number of independent keys or quantized pitches
        """

        self.num_keys = num_keys

        dim_out = self.num_keys

        super().__init__(dim_in, dim_out)

        self.output_layer = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_out),
            # TODO - make sure stability is not an issue - see ""_with_logits()
            nn.Sigmoid()
        )

    def forward(self, feats):
        """
        Perform the main processing steps for the output layer.

        Parameters
        ----------
        feats : Tensor (B x F x T)
          input features for a batch of tracks,
          TODO - check that this is accurate for what I have so far
          B - batch size,
          F - dimensionality of input features,
          T - number of time steps
        """

        keys = self.output_layer(feats)
        return keys

    def get_loss(self, output, reference):
        """
        Compute the binary cross entropy loss for each key independently.

        Parameters
        ----------
        output : Tensor (B x F x T)
          output vectors for a batch of tracks,
          TODO - check that this is accurate for what I have so far
          B - batch size,
          F - dimensionality of output features,
          T - number of time steps
        reference : Tensor (B x F x T)
          ground-truth for a batch of tracks,
          TODO - check that this is accurate for what I have so far
          B - batch size,
          F - dimensionality of output features,
          T - number of time steps
        """

        loss = F.binary_cross_entropy(output.float(), reference.float())
        return loss

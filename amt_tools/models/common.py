# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .. import tools

# Regular imports
from abc import abstractmethod
from copy import deepcopy
from torch import nn

import torch.nn.functional as F
import torch

# TODO - Generalize Softmax Groups to remove same number of classes constraint across groups
# TODO - Logistic Groups?


class TranscriptionModel(nn.Module):
    """
    Implements a generic music transcription model.
    """

    def __init__(self, dim_in, profile, in_channels=1, model_complexity=1, frame_width=1, device='cpu'):
        """
        Initialize parameters common to all models and instantiate
        model as a PyTorch Module.

        Parameters
        ----------
        dim_in : int
          Dimensionality of framewise input vectors along the feature axis
        profile : InstrumentProfile (tools/instrument.py)
          Instructions for organizing output layers
        in_channels : int
          Number of channels in input features
        model_complexity : int, optional (default 1)
          Scaling parameter for size of model's components
        frame_width : int
          Number of frames required for a single prediction
        device : string, optional (default /'cpu/')
          Device with which to perform processing steps
        """

        nn.Module.__init__(self)

        self.dim_in = dim_in
        self.profile = profile
        self.in_channels = in_channels
        self.model_complexity = model_complexity
        self.frame_width = frame_width
        self.device = device

        # Initialize a counter to keep track of how many iterations have been run
        self.iter = 0

        # Placeholder for appending additional modules, such as learnable filterbanks
        self.frontend = nn.Sequential()

    def change_device(self, device=None):
        """
        Change the device and load the model onto the new device.

        Parameters
        ----------
        device : string, int or None, optional (default None)
          Device to load model onto
        """

        if device is None:
            # If the function is called without a device, use the current device
            device = self.device

        if isinstance(device, int):
            # If device is an integer, assume device represents GPU number
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
          Dictionary containing all relevant fields for a group of tracks

        Returns
        ----------
        batch : dict
          Dictionary with all PyTorch Tensors added to the appropriate device
          and all pre-processing steps complete (local copy)
        """

        # Create a local copy of the batch so it is only modified within scope
        # TODO
        # batch = deepcopy(batch)

        # Make sure all data is on correct device
        batch = tools.dict_to_device(batch, self.device)

        # Try to extract audio and features from the input data
        audio = tools.unpack_dict(batch, tools.KEY_AUDIO)
        feats = tools.unpack_dict(batch, tools.KEY_FEATS)

        if audio is not None and len(self.frontend):
            # Add a channel dimension to the audio
            audio = audio.unsqueeze(-2)
            # Run audio through the frontend module, which does nothing by default
            frontend_feats = self.frontend(audio)
            # Append precomputed features to the frontend output if they exist
            feats = frontend_feats if feats is None else torch.cat((feats, frontend_feats), dim=1)

        # Add the features back to the input data
        batch[tools.KEY_FEATS] = feats

        return batch

    @abstractmethod
    def forward(self, feats):
        """
        Perform the main processing steps for the transcription model.

        Parameters
        ----------
        feats : Tensor (B x ...)
          Input features for a batch of tracks,
          B - batch size
        """

        return NotImplementedError

    @abstractmethod
    def post_proc(self, batch):
        """
        Perform necessary post-processing steps for the transcription model.

        Parameters
        ----------
        batch : dict
          Dictionary including model output for a group of tracks
        """

        return NotImplementedError

    def run_on_batch(self, batch):
        """
        Perform all processing steps of the transcription model on a batch.

        Parameters
        ----------
        batch : dict
          Dictionary containing all relevant fields for a group of tracks - if a single
          track is to be transcribed this way, it must be organized as a batch of size 1

        Returns
        ----------
        output : dict
          Dictionary containing loss and relevant predictions for a group of tracks
        """

        # Create a local copy of the batch so it is only modified within scope
        # TODO
        # batch = deepcopy(batch)

        # Pre-process batch
        batch = self.pre_proc(batch)

        # Obtain the model output for the batch of features
        batch[tools.KEY_OUTPUT] = self(batch[tools.KEY_FEATS])

        # Post-process batch
        output = self.post_proc(batch)

        # Add the frame times to the output if they exist
        if tools.query_dict(batch, tools.KEY_TIMES):
            output[tools.KEY_TIMES] = batch[tools.KEY_TIMES]

        return output

    @classmethod
    def model_name(cls):
        """
        Retrieve an appropriate tag, the class name, for the model.

        Returns
        ----------
        tag : str
          Name of the child class calling the function
        """

        tag = cls.__name__

        return tag


class OutputLayer(nn.Module):
    """
    Implements a generic output layer for transcription models.
    """

    def __init__(self, dim_in, dim_out, weights=None):
        """
        Initialize parameters common to all output layers as model fields
        and instantiate layers as a PyTorch processing Module.

        Parameters
        ----------
        dim_in : int
          Dimensionality of input features
        dim_out : int
          Dimensionality of output activations
        weights : ndarray (G x C) or None (optional)
          Class weights for calculating loss
          G - number of independent softmax groups
          C - number of classes per softmax group
        """

        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        if weights is None:
            self.weights = weights
        else:
            self.set_weights(self.weights.flatten())

    def set_weights(self, weights, device='cpu'):
        """
        Update the class weighting.

        Parameters
        ----------
        weights : ndarray (N) or None (optional)
          Class weights for calculating loss
          N - number of activations
        device : string, optional (default /'cpu/')
          Device to hold the weights
        """

        if isinstance(device, int):
            # If device is an integer, assume device represents GPU number
            device = torch.device(f'cuda:{device}'
                                  if torch.cuda.is_available() else 'cpu')

        self.weights = torch.Tensor(weights).to(device)

    @abstractmethod
    def forward(self, feats):
        """
        Perform the main processing steps for the output layer.

        Parameters
        ----------
        feats : Tensor (see child class for expected dimensions)
          Input features for a batch of tracks
        """

        return NotImplementedError

    @abstractmethod
    def get_loss(self, estimated, reference):
        """
        Perform the loss calculation at the output layer.

        Parameters
        ----------
        estimated : Tensor (see child class for expected dimensions)
          Estimated activations for a batch of tracks
        reference : Tensor (see child class for expected dimensions)
          Ground-truth activations for a batch of tracks
        """

        return NotImplementedError

    @abstractmethod
    def finalize_output(self, raw_output):
        """
        Convert loss-friendly output into actual symbolic transcription.

        Parameters
        ----------
        raw_output : Tensor (child class for expected dimensions)
          Raw model output used for calculating loss

        Returns
        ----------
        final_output : Tensor (child class for expected dimensions)
          Symbolic transcription serving as final predictions
        """

        # Create a new copy (to be double-safe not to affect gradient),
        # and remove the Tensor from the computational graph
        final_output = raw_output.clone().detach()

        return final_output


class SoftmaxGroups(OutputLayer):
    """
    Implements a multi-label softmax output layer designed to produce tablature,
    by treating each degree of freedom of a polyphonic instrument as a separate
    softmax problem.

    A straightforward example could correspond to a guitar with 6 degrees of
    freedom (the strings), where the output of each string is the softmax operation
    across the possibilities (the string's fretting).
    """

    def __init__(self, dim_in, num_groups, num_classes, weights=None):
        """
        Initialize fields of the multi-label softmax layer.

        Parameters
        ----------
        dim_in : int
          Dimensionality of input features
        num_groups : int
          Number of independent softmax groups
        num_classes : int
          Number of classes per softmax group

        See OutputLayer class for others...
        """

        self.num_groups = num_groups
        self.num_classes = num_classes

        # Total number of output neurons
        dim_out = self.num_groups * self.num_classes

        super().__init__(dim_in, dim_out, weights)

        # Intitialize the output layer
        self.output_layer = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, feats):
        """
        Perform the main processing steps for the softmax groups.

        Parameters
        ----------
        feats : Tensor (B x T x F)
          Input features for a batch of tracks
          B - batch size,
          T - number of time steps (frames),
          E - dimensionality of input features

        Returns
        ----------
        tablature : Tensor (B x T x O)
          Tablature activations
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Get the tablature
        tablature = self.output_layer(feats)

        return tablature

    def get_loss(self, estimated, reference):
        """
        Compute the cross entropy softmax loss for each group independently.

        Parameters
        ----------
        estimated : Tensor (B x T x O)
          Estimated tablature activations
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        reference : Tensor (B x DOFs x T)
          Ground-truth tablature activations
          B - batch size,
          DOFs - degrees of freedom,
          T - number of time steps (frames)

        Returns
        ----------
        loss : Tensor (1-D)
          Loss or error for entire batch
        """

        # Make clones so as not to modify originals out of function scope
        estimated = estimated.clone()
        reference = reference.clone()

        # Obtain the batch size before frame axis is collapsed
        batch_size = estimated.size(0)

        if self.weights is None:
            # Fold the degrees of freedom axis into the pseudo-batch axis
            estimated = estimated.view(-1, self.num_classes)

            # Transform ground-truth tabs into 1D softmax labels
            reference = reference.transpose(-2, -1)
            reference[reference == -1] = self.num_classes - 1
            reference = reference.flatten().long()

            # Calculate the loss for the entire pseudo-batch
            loss = F.cross_entropy(estimated.float(), reference, reduction='none')
            loss = loss.view(batch_size, -1, self.num_groups)
            # Sum loss across degrees of freedom
            loss = torch.sum(loss, dim=-1)
        else:
            # TODO - does there really need to be two branches here?
            # Initalize the loss
            loss = 0

            # Collapse the batch and frame dimension and break apart the activations by group
            estimated = estimated.view(-1, self.num_groups, self.num_classes).float()

            # Transform ground-truth tabs into 1D softmax labels
            reference[reference == -1] = self.num_classes - 1

            # Reshape activations to index by group
            weight = self.weights.view(self.num_groups, -1)

            # Loop through the Softmax groups
            for smax in range(self.num_groups):
                # Compute the weighted loss for each group
                loss += F.cross_entropy(estimated[:, smax], reference[:, smax].flatten().long(),
                                        weight=weight[smax], reduction='none')

            # Uncollapse the batch dimension
            loss = loss.view(batch_size, -1)

        # Average loss across frames
        loss = torch.mean(loss, dim=-1)
        # Average the loss across the batch
        loss = torch.mean(loss)

        return loss

    def finalize_output(self, raw_output, last_negative=True):
        """
        Convert loss-friendly output into actual symbolic transcription.

        Parameters
        ----------
        raw_output : Tensor (B x T x O)
          Raw model output used for calculating loss
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        last_negative : bool
          Whether to set the final class to -1

        Returns
        ----------
        final_output : Tensor (B x DOFs x T)
          Symbolic transcription serving as final predictions
          B - batch size,
          DOFs - degrees of freedom,
          T - number of time steps (frames)
        """

        final_output = super().finalize_output(raw_output)

        # Obtain the batch size
        batch_size = final_output.size(0)

        # Spread out the softmax groups across another dimension
        final_output = final_output.view(batch_size, -1, self.num_groups, self.num_classes)
        # Pick the choice with the most weight for each softmax group
        final_output = torch.argmax(torch.softmax(final_output, dim=-1), dim=-1)

        if last_negative:
            # Convert the last choice to the value representing silent tablature (-1)
            final_output[final_output == self.num_classes - 1] = -1

        # Switch the DOF and frame dimension to end up with tabs
        final_output = final_output.transpose(-2, -1)

        return final_output


class LogisticBank(OutputLayer):
    """
    Implements a multi-label logistic output layer designed to produce key activity,
    or more generally, quantized pitch activity.

    A straightforward example could correspond to a keyboard with 88 keys,
    where the output of each key is the sigmoid operation indicating whether
    or not the key is active.
    """

    def __init__(self, dim_in, dim_out, weights=None):
        """
        Initialize fields of the multi-label logistic layer.

        Parameters
        ----------
        dim_in : int
          Dimensionality of input features
        dim_out : int
          Dimensionality of output activations

        See OutputLayer class for others...
        """

        super().__init__(dim_in, dim_out, weights)

        # Initialize the output layer
        self.output_layer = nn.Linear(self.dim_in, self.dim_out)

    def forward(self, feats):
        """
        Perform the main processing steps for the output layer.

        Parameters
        ----------
        feats : Tensor (B x T x F)
          Input features for a batch of tracks
          B - batch size,
          T - number of time steps,
          E - dimensionality of input features

        Returns
        ----------
        multi_pitch : Tensor (B x T x O)
          Multi pitch activations
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Get the multi pitch activations
        multi_pitch = self.output_layer(feats)

        return multi_pitch

    def get_loss(self, estimated, reference):
        """
        Compute the binary cross entropy loss for each key independently.

        Parameters
        ----------
        estimated : Tensor (B x T x O)
          estimated activations for a batch of tracks
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        reference : Tensor (B x O x T)
          ground-truth activations for a batch of tracks
          B - batch size,
          O - number of output neurons (dim_out),
          T - number of time steps (frames)

        Returns
        ----------
        loss : Tensor (1-D)
          Loss or error for entire batch
        """

        # Make clones so as not to modify originals out of function scope
        estimated = estimated.clone()
        reference = reference.clone()

        # Switch the frame and key dimension
        estimated = estimated.transpose(-2, -1)

        # Add a frame dimension to the weights for broadcasting
        weight = self.weights.unsqueeze(-1) if self.weights is not None else None

        # Calculate the loss for the entire pseudo-batch
        loss = F.binary_cross_entropy_with_logits(estimated.float(), reference.float(),
                                                  weight=weight, reduction='none')
        # Average loss across frames
        loss = torch.mean(loss, dim=-1)
        # Sum loss across keys
        loss = torch.sum(loss, dim=-1)
        # Average loss across the batch
        loss = torch.mean(loss)

        return loss

    def finalize_output(self, raw_output, threshold=None):
        """
        Convert loss-friendly output into actual symbolic transcription.

        Parameters
        ----------
        raw_output : Tensor (B x T x O)
          Raw model output used for calculating loss
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        threshold : float (0, 1) or None (Optional)
          Threshold at which activations are considered positive

        Returns
        ----------
        final_output : Tensor (B x O x T)
          Symbolic transcription serving as final predictions
          B - batch size,
          O - number of output neurons (dim_out),
          T - number of time steps (frames)
        """

        final_output = super().finalize_output(raw_output)

        # Apply sigmoid activation
        final_output = torch.sigmoid(final_output)
        # Switch the frame and key dimension
        final_output = final_output.transpose(-2, -1)

        if threshold is not None:
            # Convert output to binary values based on the threshold
            final_output = tools.threshold_activations(final_output, threshold)

        return final_output

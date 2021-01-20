# My imports
from amt_models.tools import *

# Regular imports
from abc import abstractmethod
from torch import nn

import torch.nn.functional as F
import torch

# TODO - LogisticGroups - might be able to make child of LogisticBank instead


class TranscriptionModel(nn.Module):
    """
    Implements a generic music transcription model.
    """

    def __init__(self, dim_in, profile, in_channels=1, model_complexity=1, device='cpu'):
        """
        Initialize parameters common to all models and instantiate
        model as a PyTorch Module.

        Parameters
        ----------
        dim_in : int
          Dimensionality of framewise input vectors along the feature axis
        profile : InstrumentProfile (tools/instrument.py)
          Instructions for organizing output and ground-truth
        in_channels : int
          Number of channels in input features
        model_complexity : int, optional (default 1)
          Scaling parameter for sizes of model's components
        device : string, optional (default /'cpu/')
          Device with which to perform processing steps
        """

        super().__init__()

        self.dim_in = dim_in
        self.profile = profile
        self.in_channels = in_channels
        self.model_complexity = model_complexity
        self.device = device

        # Placeholder for appending additional modules, such as learnable filterbanks
        self.feat_ext = nn.Sequential(
            nn.Identity()
        )

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
        """

        batch = track_to_device(batch, self.device)

        # Get input audio
        input_audio = batch['audio'].unsqueeze(1)

        # Run audio through the feature extraction module, which does nothing by default
        auto_feats = self.feat_ext(input_audio)

        # If features exist, extract them
        if 'feats' in batch.keys():
            # Obtain any fixed features
            feats = batch['feats']
            # Check if any features were calculated automatically
            if not auto_feats.equal(input_audio):
                # If so, add to fixed features (number of frames must match)
                feats = torch.cat((feats, auto_feats), dim=1)
        else:
            # Otherwise we assume features were calculated automatically
            # - if feature extraction module was left as
            #   identity, our features are the audio itself
            feats = auto_feats

        batch['feats'] = feats

        return batch

    @abstractmethod
    def forward(self, feats):
        """
        Perform the main processing steps for the transcription model.

        Parameters
        ----------
        feats : Tensor (B x C x ...)
          Input features for a batch of tracks,
          B - batch size
          C - channels
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
          track is to be transcribed, it must be organized as a batch of size 1

        Returns
        ----------
        preds : dict
          Dictionary containing loss and relevant predictions for a group of tracks
        """

        # Pre-process batch
        batch = self.pre_proc(batch)

        # Obtain the model output for the batch of features
        batch['preds'] = self(batch['feats'])

        # Post-process batch
        preds = self.post_proc(batch)

        return preds

    @abstractmethod
    def special_steps(self):
        """
        Perform any final training steps specific to this model.
        """

        return NotImplementedError

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

    def __init__(self, dim_in, dim_out, profile, tag):
        """
        Initialize parameters common to all output layers as model fields
        and instantiate layers as a PyTorch processing Module.

        Parameters
        ----------
        dim_in : int
          Dimensionality of input features
        dim_out : int
          Dimensionality of output vectors
        profile : InstrumentProfile (tools/instrument.py)
          Instructions for organizing output and ground-truth
        tag : str
          Key to use for adding output to prediction dictionary
        """

        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.profile = profile
        self.tag = tag

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
    def get_loss(self, output, reference):
        """
        Perform the loss calculation at the output layer.

        Parameters
        ----------
        output : Tensor (child class for expected dimensions)
          Output vectors for a batch of tracks
        reference : Tensor (see child class for expected dimensions)
          Ground-truth for a batch of tracks
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

    def __init__(self, dim_in, profile=None, tag='tabs'):
        """
        Initialize fields of the multi-label softmax layer.

        Parameters
        ----------
        dim_in : int
          Dimensionality of input features
        profile : InstrumentProfile (tools/instrument.py)
          Instructions for organizing output and ground-truth
        tag : str
          Key to use for adding output to prediction dictionary
        """

        # Default the instrument profile
        if profile is None:
            profile = GuitarProfile()

        # Make sure the provided instrument profile is compatible
        # TODO - generalize to tabs in future
        assert isinstance(profile, GuitarProfile)

        # Degrees of freedom - number of softmax groups
        self.num_dofs = profile.num_strings
        # Possibilities - number of choices in each softmax
        self.num_poss = profile.num_frets + 2

        # Total number of output neurons
        dim_out = self.num_dofs * self.num_poss

        super().__init__(dim_in, dim_out, profile, tag)

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
          F - dimensionality of input features

        Returns
        ----------
        preds : dict w/ Tensor (B x T x O)
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Get the tabs
        tabs = self.output_layer(feats)

        # Create and return a new dictionary containing the tabs
        preds = {
            self.tag : tabs
        }

        return preds

    def get_loss(self, output, reference):
        """
        Compute the cross entropy softmax loss for each group independently.

        Parameters
        ----------
        output : Tensor (B x T x O)
          Tablature output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        reference : Tensor (B x DOFs x T)
          Ground-truth for a batch of tracks
          B - batch size,
          DOFs - degrees of freedom,
          T - number of time steps (frames)

        Returns
        ----------
        loss : Tensor (1-D)
          Loss or error for entire batch
        """

        # Obtain the true batch size
        bs = get_batch_size(output)
        # TODO - for debugging purposes - remove later
        #output_ = output.view(bs, -1, self.num_dofs, self.num_poss).transpose(1, 2)
        #reference_ = reference
        # Fold the degrees of freedom axis into the pseudo-batch axis
        output = output.view(-1, self.num_poss)

        # Ensure ground-truth is in tablature format
        # TODO - batch dimension is roughing me up here
        #reference = to_tabs(reference, self.profile)

        # Transform ground-truth tabs into 1D softmax labels
        reference = reference.transpose(1, 2)
        reference[reference == -1] = self.num_poss - 1
        reference = reference.flatten().long()

        # Calculate the loss for the entire pseudo-batch
        loss = F.cross_entropy(output.float(), reference, reduction='none')
        loss = loss.view(bs, -1, self.num_dofs)
        # TODO - for debugging purposes - remove later
        #print(output_[0, 0, 199])
        #print(f'GT: {reference_[0, 0, 199].item()}, loss: {loss[0, 199, 0].item()}')
        # Sum loss across degrees of freedom
        loss = torch.sum(loss, dim=-1)
        # Average loss across frames
        loss = torch.mean(loss, dim=-1)
        # Average the loss across the batch
        loss = torch.mean(loss)

        return loss

    def finalize_output(self, raw_output):
        """
        Convert loss-friendly output into actual symbolic transcription.

        Parameters
        ----------
        raw_output : Tensor (B x T x O)
          Raw model output used for calculating loss
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)

        Returns
        ----------
        final_output : Tensor (B x DOFs x T)
          Symbolic transcription serving as final predictions
          B - batch size,
          DOFs - degrees of freedom,
          T - number of time steps (frames)
        """

        final_output = super().finalize_output(raw_output)

        # Obtain the true batch size
        bs = get_batch_size(final_output)

        # Break apart the softmax groups across another dimension
        final_output = final_output.view(bs, -1, self.num_dofs, self.num_poss)
        # Pick the most choice with the most weight for each group
        final_output = torch.argmax(torch.softmax(final_output, dim=-1), dim=-1)
        # Convert the last choice to the value representing silent tablature (-1)
        final_output[final_output == self.num_poss - 1] = -1
        # Switch the DOF and frame dimension to end up with tabs
        final_output = final_output.transpose(1, 2)

        return final_output


class LogisticBank(OutputLayer):
    """
    Implements a multi-label logistic output layer designed to produce key activity,
    or more generally, quantized pitch activity.

    A straightforward example could correspond to a keyboard with 88 keys,
    where the output of each key is the sigmoid operation indicating whether
    or not the key is active.
    """

    def __init__(self, dim_in, profile=None, tag='keys'):
        """
        Initialize fields of the multi-label logistic layer.

        Parameters
        ----------
        dim_in : int
          Dimensionality of input features
        profile : InstrumentProfile (tools/instrument.py)
          Instructions for organizing output and ground-truth
        tag : str
          Key to use for adding output to prediction dictionary
        """

        # Default the instrument profile
        if profile is None:
            profile = PianoProfile()

        # Number of independent logistic units
        dim_out = profile.get_range_len()

        super().__init__(dim_in, dim_out, profile, tag)

        # Intitialize the output layer
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
        feats : Tensor (B x T x F)
          input features for a batch of tracks
          B - batch size,
          T - number of time steps,
          F - dimensionality of input features

        Returns
        ----------
        preds : dict w/ Tensor (B x T x O)
          Dictionary containing pianoroll output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Get the pianoroll
        keys = self.output_layer(feats)

        # Create and return a new dictionary containing the pianoroll
        preds = {
            self.tag : keys
        }

        return preds

    def get_loss(self, output, reference):
        """
        Compute the binary cross entropy loss for each key independently.

        Parameters
        ----------
        output : Tensor (B x T x O)
          output vectors for a batch of tracks
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        reference : Tensor (B x O x T)
          ground-truth for a batch of tracks
          B - batch size,
          O - number of output neurons (dim_out),
          T - number of time steps (frames)

        Returns
        ----------
        loss : Tensor (1-D)
          Loss or error for entire batch
        """

        # Switch the frame and key dimension
        output = output.transpose(1, 2)

        # Ensure ground-truth is in single pianoroll format
        # TODO - batch dimension is roughing me up here
        # reference = to_single(reference, self.profile)

        # Calculate the loss for the entire pseudo-batch
        loss = F.binary_cross_entropy(output.float(), reference.float(), reduction='none')
        # Average loss across frames
        loss = torch.mean(loss, dim=-1)
        # Average loss across keys
        loss = torch.mean(loss, dim=-1)
        # Average loss across the batch
        loss = torch.mean(loss)

        return loss

    def finalize_output(self, raw_output):
        """
        Convert loss-friendly output into actual symbolic transcription.

        Parameters
        ----------
        raw_output : Tensor (B x T x O)
          Raw model output used for calculating loss
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)

        Returns
        ----------
        final_output : Tensor (B x O x T)
          Symbolic transcription serving as final predictions
          B - batch size,
          O - number of output neurons (dim_out),
          T - number of time steps (frames)
        """

        final_output = super().finalize_output(raw_output)

        # Switch the frame and key dimension
        final_output = final_output.transpose(1, 2)
        # Convert output to binary values
        final_output = threshold_arr(final_output, 0.5)

        return final_output

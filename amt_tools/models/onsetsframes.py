# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import TranscriptionModel, LogisticBank
from .. import tools

# Regular imports
from torch import nn

import numpy as np
import torch

# TODO - velocity stuff
# TODO - optional weighted frame loss


class OnsetsFrames(TranscriptionModel):
    """
    Implements the Onsets & Frames model (V1) (https://arxiv.org/abs/1710.11153).
    """

    def __init__(self, dim_in, profile, in_channels=1, model_complexity=2, detach_heads=False, device='cpu'):
        """
        Initialize the model and establish parameter defaults in function signature.

        Parameters
        ----------
        See TranscriptionModel class for others...

        detach_heads : bool
          Whether to feed the gradient of the pitch head back into the onset head
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, 1, device)

        self.detach_heads = detach_heads

        # Number of output neurons for the acoustic models
        self.dim_am = 256 * self.model_complexity

        # Number of output neurons for the language models
        self.dim_lm = 256 * (self.model_complexity - 1)

        # Number of output neurons for each head's activations
        dim_out = self.profile.get_range_len()

        # Create the onset detector head
        self.onset_head = nn.Sequential(
            AcousticModel(self.dim_in, self.dim_am, self.in_channels, self.model_complexity),
            LanguageModel(self.dim_am, self.dim_lm),
            LogisticBank(self.dim_lm, dim_out)
        )

        # Create the multi pitch estimator head
        self.pitch_head = nn.Sequential(
            AcousticModel(self.dim_in, self.dim_am, self.in_channels, self.model_complexity),
            LogisticBank(self.dim_am, dim_out)
        )

        # Create the refined multi pitch estimator head
        self.dim_aj = 2 * dim_out
        self.adjoin = nn.Sequential(
            LanguageModel(self.dim_aj, self.dim_lm),
            LogisticBank(self.dim_lm, dim_out)
        )

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
          and all pre-processing steps complete
        """

        batch = super().pre_proc(batch)

        # Create a local copy of the batch so it is only modified within scope
        # TODO
        # batch = deepcopy(batch)

        # Switch the frequency and time axes
        batch[tools.KEY_FEATS] = batch[tools.KEY_FEATS].transpose(-1, -2)

        return batch

    def forward(self, feats):
        """
        Perform the main processing steps for Onsets & Frames (V1).

        Parameters
        ----------
        feats : Tensor (B x C x T x F)
          Input features for a batch of tracks,
          B - batch size
          C - channels
          T - number of frames
          F - number of features (frequency bins)

        Returns
        ----------
        output : dict w/ Tensors (B x T x O)
          Dictionary containing multi pitch and onsets output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Initialize an empty dictionary to hold output
        output = dict()

        # Obtain the initial multi pitch estimate
        multi_pitch = self.pitch_head(feats)

        # Obtain the onsets estimate and add it to the output dictionary
        onsets = self.onset_head(feats)
        output[tools.KEY_ONSETS] = onsets

        if self.detach_heads:
            # Disconnect the onset head from the pitch head's graph
            onsets = onsets.clone().detach()

        # Concatenate the above estimates
        joint = torch.cat((onsets, multi_pitch), -1)

        # Obtain a refined multi pitch estimate and add it to the output dictionary
        output[tools.KEY_MULTIPITCH] = self.adjoin(joint)

        return output

    def post_proc(self, batch):
        """
        Calculate loss and finalize model output.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing multi pitch and onsets output as well as loss
        """

        # Extract the raw output
        output = batch[tools.KEY_OUTPUT]

        # Obtain pointers to the output layers
        onset_output_layer = self.onset_head[-1]
        pitch_output_layer = self.adjoin[-1]

        # Obtain the onset and pitch estimations
        onsets_est = output[tools.KEY_ONSETS]
        multi_pitch_est = output[tools.KEY_MULTIPITCH]

        # Check to see if ground-truth multi pitch is available
        if tools.KEY_MULTIPITCH in batch.keys():
            # Keep track of all losses
            loss = dict()

            # Extract the ground-truth and calculate the multi pitch loss term
            multi_pitch_ref = batch[tools.KEY_MULTIPITCH]
            multi_pitch_loss = pitch_output_layer.get_loss(multi_pitch_est, multi_pitch_ref)
            loss[tools.KEY_LOSS_PITCH] = multi_pitch_loss

            # Check to see if ground-truth onsets are available
            if tools.KEY_ONSETS in batch.keys():
                # Extract the ground-truth
                onsets_ref = batch[tools.KEY_ONSETS]
            else:
                # Obtain the onset labels from the reference multi pitch
                onsets_ref = tools.multi_pitch_to_onsets(multi_pitch_ref)

            # Calculate the onsets loss term
            onsets_loss = onset_output_layer.get_loss(onsets_est, onsets_ref)
            loss[tools.KEY_LOSS_ONSETS] = onsets_loss

            # Compute the total loss and add it to the output dictionary
            total_loss = multi_pitch_loss + onsets_loss
            loss[tools.KEY_LOSS_TOTAL] = total_loss
            output[tools.KEY_LOSS] = loss

        # Finalize onset and pitch estimations
        output[tools.KEY_ONSETS] = onset_output_layer.finalize_output(onsets_est, 0.5)
        output[tools.KEY_MULTIPITCH] = pitch_output_layer.finalize_output(multi_pitch_est, 0.5)

        return output


class OnsetsFrames2(OnsetsFrames):
    """
    Implements the Onsets & Frames model (V2) (https://arxiv.org/abs/1810.12247).
    """

    def __init__(self, dim_in, profile, in_channels=1, model_complexity=3, detach_heads=True, device='cpu'):
        """
        Initialize the model and establish parameter defaults in function signature.

        Parameters
        ----------
        See OnsetsFrames class...
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, detach_heads, device)

        # Number of output neurons for each head's activations
        dim_out = self.profile.get_range_len()

        # Create the offset detector head
        self.offset_head = nn.Sequential(
            AcousticModel(self.dim_in, self.dim_am, self.in_channels, self.model_complexity),
            LanguageModel(self.dim_am, self.dim_lm),
            LogisticBank(self.dim_lm, dim_out)
        )

        # Increase the input size of the refinement stage
        self.dim_aj += dim_out
        self.adjoin[0] = LanguageModel(self.dim_aj, self.dim_lm)

    def forward(self, feats):
        """
        Perform the main processing steps for Onsets & Frames (V2).

        Parameters
        ----------
        feats : Tensor (B x C x T x F)
          Input features for a batch of tracks,
          B - batch size
          C - channels
          T - number of frames
          F - number of features (frequency bins)

        Returns
        ----------
        output : dict w/ Tensors (B x T x O)
          Dictionary containing multi pitch, onsets, and offsets output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Initialize an empty dictionary to hold output
        output = dict()

        # Obtain the initial multi pitch estimate
        multi_pitch = self.pitch_head(feats)

        # Obtain the onsets estimate and add it to the output dictionary
        onsets = self.onset_head(feats)
        output[tools.KEY_ONSETS] = onsets

        # Obtain the onsets estimate and add it to the output dictionary
        offsets = self.offset_head(feats)
        output[tools.KEY_OFFSETS] = offsets

        if self.detach_heads:
            # Disconnect the onset/offset heads from the pitch head's graph
            onsets = onsets.clone().detach()
            offsets = offsets.clone().detach()

        # Concatenate the above estimates
        joint = torch.cat((onsets, offsets, multi_pitch), -1)

        # Obtain a refined multi pitch estimate and add it to the output dictionary
        output[tools.KEY_MULTIPITCH] = self.adjoin(joint)

        return output

    def post_proc(self, batch):
        """
        Calculate loss and finalize model output.

        Parameters
        ----------
        batch : dict
          Dictionary including model output and potentially
          ground-truth for a group of tracks

        Returns
        ----------
        output : dict
          Dictionary containing multi pitch, onsets, and offsets output as well as loss
        """

        # Perform standard Onsets & Frames steps
        output = super().post_proc(batch)

        # Obtain pointers to the offset output layer
        offset_output_layer = self.offset_head[-1]

        # Obtain the offset estimations
        offsets_est = output[tools.KEY_OFFSETS]

        # Check to see if ground-truth offsets are available
        if tools.KEY_LOSS in output.keys():
            # Check to see if ground-truth offsets are available
            if tools.KEY_OFFSETS in batch.keys():
                # Extract the ground-truth
                offsets_ref = batch[tools.KEY_OFFSETS]
            else:
                # Obtain the offset labels from the reference multi pitch
                offsets_ref = tools.multi_pitch_to_offsets(batch[tools.KEY_MULTIPITCH])

            # Extract all of the losses
            loss = output[tools.KEY_LOSS]

            # Calculate the offsets loss term
            offsets_loss = offset_output_layer.get_loss(offsets_est, offsets_ref)
            loss[tools.KEY_LOSS_OFFSETS] = offsets_loss

            # Compute the total loss and add it back to the output dictionary
            loss[tools.KEY_LOSS_TOTAL] += offsets_loss
            output[tools.KEY_LOSS] = loss

        # Finalize offset estimations
        output[tools.KEY_OFFSETS] = offset_output_layer.finalize_output(offsets_est)

        return output


class AcousticModel(nn.Module):
    """
    Implements the acoustic model from Kelz 2016 (https://arxiv.org/abs/1710.11153),
    with the modifications made in Onsets & Frames. To the best of my knowledge, the
    architecture modifications consist only of an additional batch normalization
    after the first convolutional layer. See the following Magenta implementation:
    https://github.com/magenta/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
    """

    def __init__(self, dim_in, dim_out, in_channels=1, model_complexity=2):
        """
        Initialize the acoustic model and establish parameter defaults in function signature.

        Parameters
        ----------
        dim_in : int
          Dimensionality of framewise input vectors along the feature axis
        dim_out : int
          Dimensionality of framewise output vectors along the feature axis
        in_channels : int
          Number of channels in input features
        model_complexity : int, optional (default 1)
          Scaling parameter for size of model's components
        """

        super(AcousticModel, self).__init__()

        # Number of filters for each convolutional layer
        nf1 = 16 * model_complexity
        nf2 = nf1
        nf3 = 32 * model_complexity

        # Kernel size for each convolutional layer
        ks1 = (3, 3)
        ks2 = ks1
        ks3 = ks1

        # Padding amount for each convolutional layer
        pd1 = 1
        pd2 = pd1
        pd3 = pd1

        # Reduction size for each pooling operation
        rd1 = (1, 2)
        rd2 = rd1

        # Dropout percentages for each dropout operation
        dp1 = 0.25
        dp2 = dp1
        dp3 = 0.50

        self.layer1 = nn.Sequential(
            # 1st convolution
            nn.Conv2d(in_channels, nf1, ks1, padding=pd1),
            # 1st batch normalization
            nn.BatchNorm2d(nf1),
            # Activation function
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            # 2nd convolution
            nn.Conv2d(nf1, nf2, ks2, padding=pd2),
            # 2nd batch normalization
            nn.BatchNorm2d(nf2),
            # Activation function
            nn.ReLU(),
            # 1st reduction
            nn.MaxPool2d(rd1),
            # 1st dropout
            nn.Dropout(dp1)
        )

        self.layer3 = nn.Sequential(
            # 3rd convolution
            nn.Conv2d(nf2, nf3, ks3, padding=pd3),
            # 3rd batch normalization
            nn.BatchNorm2d(nf3),
            # Activation function
            nn.ReLU(),
            # 2nd reduction
            nn.MaxPool2d(rd2),
            # 2nd dropout
            nn.Dropout(dp2)
        )

        # Feature dimension was reduced by a factor of 4 as a result of pooling
        feat_map_height = dim_in // 4
        # Determine the number of features per sample (frame)
        feat_map_size = nf3 * feat_map_height

        self.fc1 = nn.Sequential(
            # 1st fully-connected
            nn.Linear(feat_map_size, dim_out),
            # 3rd dropout
            nn.Dropout(dp3)
        )

    def forward(self, in_feats):
        """
        Feed features through the acoustic model.

        Parameters
        ----------
        in_feats : Tensor (B x C x T x F)
          Input features for a batch of tracks,
          B - batch size
          C - channels
          T - number of frames
          F - number of features (frequency bins)

        Returns
        ----------
        out_feats : Tensor (B x T x E)
          Embeddings for a batch of tracks,
          B - batch size
          T - number of frames
          E - dimensionality of embeddings
        """

        # Feed features through convolutional layers
        x = self.layer1(in_feats)
        x = self.layer2(x)
        x = self.layer3(x)

        # Switch the channel and time axes
        x = x.transpose(-3, -2)
        # Combine the channel and feature axes
        x = x.flatten(-2)

        # Feed the convolutional features through the fully connected layer
        out_feats = self.fc1(x)

        return out_feats


class LanguageModel(nn.Module):
    """
    Implements a simple LSTM language model for refining features over time.
    """

    def __init__(self, dim_in, dim_out, chunk_len=512, bidirectional=True):
        """
        Initialize the language model and establish parameter defaults in function signature.

        Parameters
        ----------
        dim_in : int
          Dimensionality of framewise input vectors along the feature axis
        dim_out : int
          Dimensionality of framewise output vectors along the feature axis
        chunk_len : int
          Number of frames to process at a time during inference
        bidirectional : bool
          Whether LSTM is bidirectional
        """

        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.chunk_len = chunk_len

        # Keep track of the bidirectional argument as the number of directions
        self.num_directions = int(bidirectional) + 1

        # Determine the number of neurons
        self.hidden_size = self.dim_out // self.num_directions

        # Initialize the LSTM
        self.mlm = nn.LSTM(input_size=self.dim_in,
                           hidden_size=self.hidden_size,
                           batch_first=True,
                           bidirectional=bidirectional)

    def forward(self, in_feats):
        """
        Feed features through the music language model.

        Parameters
        ----------
        in_feats : Tensor (B x T x E)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          E - dimensionality of input embeddings (dim_in)

        Returns
        ----------
        out_feats : Tensor (B x T x E)
          Embeddings for a batch of tracks,
          B - batch size
          T - number of frames
          E - dimensionality of output embeddings (dim_out)
        """

        # Do not chunk the features during training
        if self.training:
            # Process the features, discarding the hidden state and cell state
            out_feats, _ = self.mlm(in_feats)
        else:
            # Determine the batch size and the number of frames given
            batch_size, seq_length, _ = in_feats.size()

            # Initialize the hidden state and cell state
            hidden = torch.zeros(self.num_directions, batch_size, self.hidden_size)
            cell = torch.zeros(self.num_directions, batch_size, self.hidden_size)

            # Create a placeholder for the entire output sequence
            out_feats = torch.zeros(batch_size, seq_length, self.dim_out)

            # Add the LSTM states and output placeholder to the appropriate device
            hidden = hidden.to(in_feats.device)
            cell = cell.to(in_feats.device)
            out_feats = out_feats.to(in_feats.device)

            # Determine the start and end of each chunk
            starts = np.arange(0, seq_length, self.chunk_len)
            ends = starts + self.chunk_len

            # Loop through each chunk in the forward direction
            for start, end in zip(starts, ends):
                # Chunk the input features
                chunk_feats = in_feats[..., start : end, :]
                # Process the chunk, using the previous hidden and cell state
                chunk_out, (hidden, cell) = self.mlm(chunk_feats, (hidden, cell))
                # Add the chunk's output to where it belongs in the placeholder
                out_feats[..., start : end, :] = chunk_out

            if self.mlm.bidirectional:
                # Reset the hidden and cell state
                hidden.zero_()
                cell.zero_()

                # Loop through each chunk in the backward direction. This needs
                # to be done since, above, the reverse direction part was given
                # the hidden and cell state with respect to the forward direction
                for start, end in zip(reversed(starts), reversed(ends)):
                    # Chunk the input features
                    chunk_feats = in_feats[..., start : end, :]
                    # Process the chunk, using the previous hidden and cell state
                    chunk_out, (hidden, cell) = self.mlm(chunk_feats, (hidden, cell))
                    # Overwrite the first half of the embeddings with the chunk's output
                    out_feats[:, start : end, self.hidden_size:] = chunk_out[:, :, self.hidden_size:]

        return out_feats


class OnlineLanguageModel(LanguageModel):
    """
    Implements a uni-directional and online-capable LSTM language model.
    """

    def __init__(self, dim_in, dim_out):
        """
        Initialize the language model and establish parameter defaults in function signature.

        Parameters
        ----------
        See LanguageModel class...
        """

        super().__init__(dim_in, dim_out, bidirectional=False)

        # Initialize the hidden and cell state
        self.hidden = None
        self.cell = None

        self.reset_state()

    def reset_state(self):
        """
        Reset the hidden and cell state to None.
        """

        self.hidden = None
        self.cell = None

    def train(self, mode=True):
        """
        Reset the hidden and cell state every time the model is put into evaluation mode.

        Parameters
        ----------
        mode : bool
          Whether to set to training mode [True] or evaluation mode [False]
        """

        if not mode:
            self.reset_state()

        super().train(mode)

    def forward(self, in_feats):
        """
        Feed features through the music language model.

        Parameters
        ----------
        in_feats : Tensor (B x 1 x E)
          Input features for a batch of tracks,
          B - batch size
          E - dimensionality of input embeddings (dim_in)

        Returns
        ----------
        out_feats : Tensor (B x 1 x E)
          Embeddings for a batch of tracks,
          B - batch size
          E - dimensionality of output embeddings (dim_out)
        """

        if self.training:
            # Call the regular forward function
            out_feats = super().forward(in_feats)
        else:
            # Determine the batch size of the features fed in
            batch_size = in_feats.size(0)

            if self.hidden is None:
                # Initialize the hidden state
                self.hidden = torch.zeros(self.num_directions, batch_size, self.hidden_size).to(in_feats.device)
            if self.cell is None:
                # Initialize the cell state
                self.cell = torch.zeros(self.num_directions, batch_size, self.hidden_size).to(in_feats.device)

            # Process the chunk, using the previous hidden and cell state
            out_feats, (self.hidden, self.cell) = self.mlm(in_feats, (self.hidden, self.cell))

        return out_feats

# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from .common import TranscriptionModel, SoftmaxGroups
from .. import tools

# Regular imports
from torch import nn


class TabCNN(TranscriptionModel):
    """
    Implements the TabCNN model (http://archives.ismir.net/ismir2019/paper/000033.pdf).
    """

    def __init__(self, dim_in, profile, in_channels=1, model_complexity=1, device='cpu'):
        """
        Initialize the model and establish parameter defaults in function signature.

        Parameters
        ----------
        See TranscriptionModel class for others...
        """

        super().__init__(dim_in, profile, in_channels, model_complexity, 9, device)

        # Initialize a flag to check whether to pad input features
        self.online = False

        # Number of filters for each stage
        nf1 = 32 * self.model_complexity
        nf2 = 64 * self.model_complexity
        nf3 = nf2

        # Kernel size for each stage
        ks1 = (3, 3)
        ks2 = ks1
        ks3 = ks1

        # Reduction size for each stage
        rd1 = (2, 2)

        # Dropout percentages for each stage
        dp1 = 0.25
        dp2 = 0.50

        self.conv = nn.Sequential(
            # 1st convolution
            nn.Conv2d(self.in_channels, nf1, ks1),
            # Activation function
            nn.ReLU(),
            # 2nd convolution
            nn.Conv2d(nf1, nf2, ks2),
            # Activation function
            nn.ReLU(),
            # 3rd convolution
            nn.Conv2d(nf2, nf3, ks3),
            # Activation function
            nn.ReLU(),
            # 1st reduction
            nn.MaxPool2d(rd1),
            # 1st dropout
            nn.Dropout(dp1)
        )

        # Determine the height, width, and total size of the feature map
        feat_map_height = (self.dim_in - 6) // 2
        feat_map_width = (self.frame_width - 6) // 2
        self.conv_embedding_size = nf3 * feat_map_height * feat_map_width

        # Number of neurons for each fully-connected stage
        self.fc_embedding_size = 128 * self.model_complexity

        # Extract tablature parameters
        num_groups = self.profile.get_num_dofs()
        num_classes = self.profile.num_pitches + 1

        self.dense = nn.Sequential(
            # 1st fully-connected
            nn.Linear(self.conv_embedding_size, self.fc_embedding_size),
            # Activation function
            nn.ReLU(),
            # 2nd dropout
            nn.Dropout(dp2),
            # 2nd fully-connected
            SoftmaxGroups(self.fc_embedding_size, num_groups, num_classes)
        )

    def toggle_online(self):
        """
        Toggle the flag for padding input features. This is necessary to differentiate
        between training/validation, where there is ground-truth for each input frame,
        and inference, where there are no labels to compare against and only one output
        is desired for a group of input features filling the expected frame width.
        """

        # Switch the flag
        self.online = not self.online

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

        # Extract the features from the batch as a NumPy array
        feats = tools.tensor_to_array(batch[tools.KEY_FEATS])
        # Window the features to mimic online/real-time operation
        feats = tools.framify_activations(feats, self.frame_width, pad=(not self.online))
        # Convert the features back to PyTorch tensor and add to device
        feats = tools.array_to_tensor(feats, self.device)
        # Switch the sequence-frame and feature axes
        feats = feats.transpose(-2, -3)
        # Switch the sequence-frame and channel axes
        feats = feats.transpose(-3, -4)

        batch[tools.KEY_FEATS] = feats

        return batch

    def forward(self, feats):
        """
        Perform the main processing steps for TabCNN.

        Parameters
        ----------
        feats : Tensor (B x T x C x F x W)
          Input features for a batch of tracks,
          B - batch size
          T - number of frames
          C - number of channels in features
          F - number of features (frequency bins)
          W - frame width of each sample

        Returns
        ----------
        output : dict w/ Tensor (B x T x O)
          Dictionary containing tablature output
          B - batch size,
          T - number of time steps (frames),
          O - number of output neurons (dim_out)
        """

        # Initialize an empty dictionary to hold output
        output = dict()

        # Obtain the batch size before sequence-frame axis is collapsed
        batch_size = feats.size(0)

        # Collapse the sequence-frame axis into the batch axis,
        # so that each windowed group of frames is treated as one
        # independent sample. This is not done during pre-processing
        # in order to maintain consistency with the notion of batch size
        feats = feats.reshape(-1, self.in_channels, self.dim_in, self.frame_width)

        # Obtain the feature embeddings
        embeddings = self.conv(feats)
        # Flatten spatial features into one embedding
        embeddings = embeddings.flatten(1)
        # Size of the embedding
        embedding_size = embeddings.size(-1)
        # Restore proper batch dimension, unsqueezing sequence-frame axis
        embeddings = embeddings.view(batch_size, -1, embedding_size)

        # Obtain the tablature estimate and add it to the output dictionary
        output[tools.KEY_TABLATURE] = self.dense(embeddings)

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
          Dictionary containing tablature as well as loss
        """

        # Extract the raw output
        output = batch[tools.KEY_OUTPUT]

        # Obtain pointers to the output layer
        tablature_output_layer = self.dense[-1]

        # Obtain the tablature estimation
        tablature_est = output[tools.KEY_TABLATURE]

        # Check to see if ground-truth tablature is available
        if tools.KEY_TABLATURE in batch.keys():
            # Extract the ground-truth, calculate the loss and add it to the dictionary
            tablature_ref = batch[tools.KEY_TABLATURE]
            tablature_loss = tablature_output_layer.get_loss(tablature_est, tablature_ref)
            output[tools.KEY_LOSS] = {tools.KEY_LOSS_TOTAL : tablature_loss}

        # Finalize tablature estimation
        output[tools.KEY_TABLATURE] = tablature_output_layer.finalize_output(tablature_est)

        return output

# My imports
from .common import *
from amt_models.tools import *

# Regular imports
from torch import nn


class OnsetsFrames(TranscriptionModel):
    # TODO - try to adhere to details of paper as much as possible
    def __init__(self, dim_in, profile, in_channels, model_complexity=2, device='cpu'):
        super().__init__(dim_in, profile, in_channels, model_complexity, device)

        # Number of output neurons for the acoustic models
        self.dim_am = 256 * self.model_complexity

        # Number of output neurons for the language models
        self.dim_lm1 = 128 * self.model_complexity

        self.onsets = nn.Sequential(
            AcousticModel(self.dim_in, self.dim_am, self.in_channels, self.model_complexity),
            LanguageModel(self.dim_am, self.dim_lm1),
            LogisticBank(self.dim_lm1, self.profile, 'onsets')
        )

        self.pianoroll = nn.Sequential(
            AcousticModel(self.dim_in, self.dim_am, self.in_channels, self.model_complexity),
            LogisticBank(self.dim_am, self.profile)
        )

        self.dim_lm2 = self.onsets[-1].dim_out + self.pianoroll[-1].dim_out

        self.adjoin = nn.Sequential(
            LanguageModel(self.dim_lm2, self.dim_lm2),
            LogisticBank(self.dim_lm2, self.profile, 'pitch')
        )

    def pre_proc(self, batch):
        # Add the batch to the model's device
        super().pre_proc(batch)

        # Switch the frequency and time axes
        batch['feats'] = batch['feats'].transpose(-1, -2)

        return batch

    def forward(self, feats):
        generic_label = self.pianoroll[-1].tag
        pianoroll = self.pianoroll(feats)[generic_label]

        onsets_label = self.onsets[-1].tag
        preds = self.onsets(feats)
        onsets = preds[onsets_label]

        joint = torch.cat((onsets, pianoroll), -1)
        preds.update(self.adjoin(joint))

        return preds

    def post_proc(self, batch):
        preds = batch['preds']

        onsets_layer = self.onsets[-1]
        pianoroll_layer = self.adjoin[-1]

        onsets_label = onsets_layer.tag
        pianoroll_label = pianoroll_layer.tag

        onsets = preds[onsets_label]
        pianoroll = preds[pianoroll_label]

        loss = None

        # Check to see if ground-truth is available
        if pianoroll_label in batch.keys():
            reference_pianoroll = batch[pianoroll_label]

            if onsets_label in batch.keys():
                reference_onsets = batch[onsets_label]
            else:
                reference_onsets = get_onsets(reference_pianoroll, self.profile)

            onsets_loss = onsets_layer.get_loss(onsets, reference_onsets)
            pianoroll_loss = pianoroll_layer.get_loss(pianoroll, reference_pianoroll)
            loss = onsets_loss + pianoroll_loss

        preds.update({
            'loss': loss
        })

        preds[onsets_label] = onsets_layer.finalize_output(onsets)
        preds[pianoroll_label] = pianoroll_layer.finalize_output(pianoroll)

        return preds

    def special_steps(self):
        # TODO - parameterize
        nn.utils.clip_grad_norm_(self.parameters(), 3)

class OnsetsFrames2(OnsetsFrames):
    def __init__(self, dim_in, dim_out, model_complexity=3, device='cpu'):
        super().__init__(dim_in, dim_out, model_complexity, device)

        # TODO - add offset head

    def pre_proc(self, batch):
        return batch

    def forward(self):
        pass

    def post_proc(self, batch):
        preds = None
        loss = None

        return preds, loss

class AcousticModel(nn.Module):
    def __init__(self, dim_in, dim_out, in_channels=1, model_complexity=2):
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

        feat_map_height = dim_in // 4
        self.feat_map_size = nf3 * feat_map_height

        self.fc1 = nn.Sequential(
            # 1st fully-connected
            nn.Linear(self.feat_map_size, dim_out),
            # 3rd dropout
            nn.Dropout(dp3)
        )

    def forward(self, feats):
        x = self.layer1(feats)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.transpose(1, 2).flatten(-2)

        out = self.fc1(x)

        return out


class LanguageModel(nn.Module):
    def __init__(self, dim_in, dim_out, inf_len=512):
        super().__init__()
        self.dim_in = dim_in
        self.dim_hd = dim_out // 2
        self.dim_out = dim_out
        self.inf_len = inf_len
        self.rnn = nn.LSTM(self.dim_in, self.dim_hd, batch_first=True, bidirectional=True)

    def forward(self, feats):
        if self.training:
            return self.rnn(feats)[0]
        else:
            # Process the features in chunks
            batch_size = feats.size(0)
            seq_length = feats.size(1)

            assert self.dim_in == feats.size(2)

            h = torch.zeros(2, batch_size, self.dim_hd).to(feats.device)
            c = torch.zeros(2, batch_size, self.dim_hd).to(feats.device)
            output = torch.zeros(batch_size, seq_length, 2 * self.dim_hd).to(feats.device)

            # Forward
            slices = range(0, seq_length, self.inf_len)
            for start in slices:
                end = start + self.inf_len
                output[:, start : end, :], (h, c) = self.rnn(feats[:, start : end, :], (h, c))

            # Backward
            h.zero_()
            c.zero_()

            for start in reversed(slices):
                end = start + self.inf_len
                result, (h, c) = self.rnn(feats[:, start : end, :], (h, c))
                output[:, start : end, self.dim_hd:] = result[:, :, self.dim_hd:]

            return output

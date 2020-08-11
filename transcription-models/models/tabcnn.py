# My imports
from models.common import *

from tools.utils import *

# Regular imports
import torch.nn.functional as F

# TODO - different file naming scheme?


class TabCNN(TranscriptionModel):
    def __init__(self, dim_in, dim_out, model_complexity=1, device='cpu'):
        super().__init__(dim_in, dim_out, model_complexity, device)

        # TODO - redo model complexity
        # Number of frames required for a prediction
        sample_width = 9

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

        # Number of neurons for each fully-connected stage
        nn1 = 128
        nn2 = dim_out

        # 1st convolution
        self.cn1 = nn.Conv2d(1, nf1, ks1)
        # 2nd convolution
        self.cn2 = nn.Conv2d(nf1, nf2, ks2)
        # 3rd convolution
        self.cn3 = nn.Conv2d(nf2, nf3, ks3)
        # 1st reduction
        self.mp1 = nn.MaxPool2d(rd1)
        # 1st dropout
        self.dp1 = nn.Dropout(dp1)

        feat_map_height = (dim_in - 6) // 2
        feat_map_width = (sample_width - 6) // 2
        self.feat_map_size = nf3 * feat_map_height * feat_map_width

        # 1st fully-connected
        self.fc1 = nn.Linear(self.feat_map_size, nn1)
        # 2nd dropout
        self.dp2 = nn.Dropout(dp2)
        # 2nd fully-connected
        self.out_layer = MLSoftmax(nn1, NUM_STRINGS, NUM_FRETS + 2)

    def pre_proc(self, batch):
        # TODO - clean this up
        feats = batch['feats']
        feats = framify_tfr(feats, 9, 1, 4)
        feats = feats.transpose(-1, -2)
        feats = feats.transpose(-2, -3)
        feats = feats.squeeze(1)

        batch_size, num_wins, num_bins, win_len = feats.size()

        feats = feats.to(self.device)
        feats = feats.view(batch_size * num_wins, 1, num_bins, win_len)
        batch['feats'] = feats

        # TODO - abstract this to a function
        keys = list(batch.keys())
        for key in keys:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)

        return batch

    def forward(self, feats):
        # 1st convolution
        x = F.relu(self.cn1(feats))
        # 2nd convolution
        x = F.relu(self.cn2(x))
        # 3rd convolution
        x = F.relu(self.cn3(x))
        # 1st reduction
        x = self.mp1(x).flatten()
        # 1st dropout
        x = self.dp1(x)
        # 1st fully-connected
        x = x.view(-1, self.feat_map_size)
        x = F.relu(self.fc1(x))
        # 2nd dropout
        x = self.dp2(x)
        # 2nd fully-connected
        out = self.out_layer(x)

        return out

    # TODO - if this will be the same for other transcription models, abstract it to a function and just call that
    def post_proc(self, batch):
        batch_size = len(batch['track'])

        out = batch['out']
        out = out.view(batch_size, -1, NUM_STRINGS, NUM_FRETS + 2)

        preds = torch.argmax(torch.softmax(out, dim=-1), dim=-1)
        preds[preds == NUM_FRETS + 1] = -1
        preds = preds.transpose(1, 2)

        loss = None

        # Check to see if ground-truth is available
        if 'tabs' in batch.keys():
            out = out.view(-1, NUM_FRETS + 2)

            # TODO - tabs to softmax function? - yes this was confusing
            tabs = batch['tabs'].transpose(1, 2)
            tabs[tabs == -1] = NUM_FRETS + 1
            """
            tabs = torch.zeros(tabs_temp.shape + tuple([NUM_FRETS + 2]))
            tabs = tabs.to(tabs_temp.device)

            b, f, s = tabs_temp.size()
            b, f, s = torch.meshgrid(torch.arange(b), torch.arange(f), torch.arange(s))
            tabs[b, f, s, tabs_temp] = 1
            tabs = tabs.view(-1, NUM_FRETS + 2).long()
            """
            loss = self.out_layer.get_loss(out, tabs.flatten())
            loss = loss.view(batch_size, -1, NUM_STRINGS)
            # Sum loss across strings
            loss = torch.sum(loss, dim=-1)
            # Average loss across frames
            loss = torch.mean(loss, dim=-1)

        return preds, loss

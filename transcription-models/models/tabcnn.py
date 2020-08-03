# My imports

from models.common import *

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
        self.fc2 = nn.Linear(nn1, nn2)

    def pre_proc(self, batch):
        feats = batch['feats']
        feats = framify_tfr(feats, 9, 1, 4)
        feats = feats.transpose(-1, -2)
        feats = feats.transpose(-2, -3)
        feats = feats.squeeze(1)

        batch_size = feats.size(0)
        num_wins = feats.size(1)
        num_bins = feats.size(2)
        win_len = feats.size(3)

        feats = feats.to(self.device)
        feats = feats.view(batch_size * num_wins, 1, num_bins, win_len)
        batch['feats'] = feats

        # Check if ground-truth was provided
        if 'tabs' in batch.keys():
            tabs = batch['tabs']
            tabs = tabs.transpose(-1, -2)
            tabs = tabs.transpose(-2, -3)
            batch['tabs'] = tabs.to(self.device)

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
        out = self.fc2(x)

        return out

    def post_proc(self, batch):
        # TODO - if this will be the same for other transcription models, abstract it to a function and just call that
        out = batch['out']
        batch_size = batch['audio'].size(0)
        out = out.view(batch_size, -1, NUM_STRINGS, NUM_FRETS+2)

        preds = torch.argmax(torch.softmax(out, dim=-1), dim=-1)

        loss = None
        if 'tabs' in batch.keys():
            tabs = batch['tabs']
            tabs = tabs.view(-1, NUM_FRETS + 2)

            out = out.view(-1, NUM_FRETS + 2)
            loss = F.cross_entropy(out, torch.argmax(tabs, dim=-1), reduction='none')
            loss = torch.sum(loss.view(-1, NUM_STRINGS), dim=-1)

        return preds, loss

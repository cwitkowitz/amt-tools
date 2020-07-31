# My imports

from models.common import *

# Regular imports

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

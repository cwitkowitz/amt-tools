# My imports

from models.common import *

# Regular imports


class AcousticModel(TranscriptionModel):
    def __init__(self, dim_in, dim_out, model_complexity=2, device='cpu'):
        super().__init__(dim_in, dim_out, model_complexity, device)

        # Number of filters for each stage
        nf1 = 16 * self.model_complexity
        nf2 = nf1
        nf3 = 32 * self.model_complexity

        # Kernel size for each stage
        ks1 = (3, 3)
        ks2 = ks1
        ks3 = ks1

        # Reduction size for each stage
        rd1 = (1, 2)
        rd2 = rd1

        # Dropout percentages for each stage
        dp1 = 0.25
        dp2 = dp1
        dp3 = 0.50

        # Number of neurons for each fully-connected stage
        nn1 = 256 * self.model_complexity
        nn2 = dim_out

        self.layer1 = nn.Sequential(
            # 1st convolution
            nn.Conv2d(1, nf1, ks1),
            # 1st batch normalization
            nn.BatchNorm2d(nf1),
            # Activation function
            nn.ReLU())

        self.layer2 = nn.Sequential(
            # 1st convolution
            nn.Conv2d(nf1, nf2, ks2),
            # 1st batch normalization
            nn.BatchNorm2d(nf2),
            # Activation function
            nn.ReLU(),
            # 1st reduction
            nn.MaxPool2d(rd1),
            # 1st dropout
            nn.Dropout(dp1))

        self.layer3 = nn.Sequential(
            # 1st convolution
            nn.Conv2d(nf2, nf3, ks3),
            # 1st batch normalization
            nn.BatchNorm2d(nf3),
            # Activation function
            nn.ReLU(),
            # 1st reduction
            nn.MaxPool2d(rd2),
            # 1st dropout
            nn.Dropout(dp2))

        feat_map_height = (dim_in - 6) // 4
        feat_map_width = (sample_width - 6)
        self.feat_map_size = nf3 * feat_map_height * feat_map_width

        # 1st fully-connected
        self.fc1 = nn.Linear(self.feat_map_size, nn1)

    def forward(self, feats):
        # Stage 1 convolution
        x = F.relu(self.cn1(feats))
        # Stage 2 convolution
        x = F.relu(self.cn2(x))
        # Stage 3 convolution
        x = F.relu(self.cn3(x))
        # Stage 1 reduction
        x = self.mp1(x).flatten()
        # Stage 1 dropout
        x = self.dp1(x)
        # Stage 1 fully-connected
        x = x.view(-1, self.feat_map_size)
        x = F.relu(self.fc1(x))
        # Stage 2 dropout
        x = self.dp2(x)
        # Stage 2 fully-connected
        out = self.fc2(x)

        return out

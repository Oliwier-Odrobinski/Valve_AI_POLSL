import torch
import torch.nn as nn

class Gen(nn.Module):
    def __init__(self, input_dim=2, norm_type='instance',     embedding_activation='relu', conv_activation='leaky_relu'):
        super(Gen, self).__init__()
        
        self.init_h = 2
        self.init_w = 16
        self.init_channels = 512

        flat_dim = self.init_channels * self.init_h * self.init_w

        if embedding_activation == 'relu':
            embedding_act = nn.ReLU(inplace=True)
        elif embedding_activation == 'leaky_relu':
            embedding_act = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unknown embedding activation: {embedding_activation}")

        self.param_embed = nn.Sequential(
        nn.Linear(input_dim, 256),
        embedding_act,
        nn.Linear(256, flat_dim)


        def get_norm(channels):
            if norm_type == 'instance':
                return nn.InstanceNorm2d(channels, affine=True)
            else: 
                return nn.BatchNorm2d(channels, affine=True)

        def get_act():
            if conv_activation == 'relu':
                return nn.ReLU(inplace=True)
            else:
                return nn.LeakyReLU(0.2, inplace=True)


        layers = [
            get_norm(self.init_channels),
            get_act(),
            nn.ConvTranspose2d(self.init_channels, 256, 4, stride=2, padding=1),
            get_norm(256),
            get_act(),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            get_norm(128),
            get_act(),


            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            get_norm(64),
            get_act(),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), 
            get_norm(32),
            get_act(),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), 
            get_norm(16),
            get_act(),

            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), 
            get_norm(8),
            get_act(),

            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1), 
            nn.Tanh()  
        ]


        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.param_embed(x)
        out = x.view(x.shape[0], self.init_channels, self.init_h, self.init_w)
        img = self.model(out)
                
        return img
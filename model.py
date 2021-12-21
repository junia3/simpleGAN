import torch
import torch. nn as nn
import torch.nn.functional as F
import numpy as np

#generator : image generation
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.noise = params['noise']  #input noise vecors
        self.image_size = params["image_size"]  #input image size (1 x 28 x 28)

        self.fc1 = self.make_fc_layer(self.noise, 128, normalize=False)
        self.fc2 = self.make_fc_layer(128, 256)
        self.fc3 = self.make_fc_layer(256, 512)
        self.fc4 = self.make_fc_layer(512, 1024)
        self.fc_out = nn.Linear(1024, int(np.prod(self.image_size))).apply(self.initialize_weight)

    def make_fc_layer(self, in_channels, out_channels, normalize=True):
        layers = []
        layers.append(nn.Linear(in_channels, out_channels))
        if normalize:
            layers.append(nn.BatchNorm1d(out_channels, 0.8))

        Sequential_layer = nn.Sequential(*layers)
        Sequential_layer.apply(self.initialize_weight)
        return Sequential_layer

    def initialize_weight(self, submodule):
        if isinstance(submodule, torch.nn.Linear):
            nn.init.normal_(submodule.weight.data, 0.0, 0.02)
            nn.init.constant_(submodule.bias.data, 0)
        elif isinstance(submodule, torch.nn.BatchNorm1d):
            nn.init.normal_(submodule.weight.data, 1.0, 0.02)
            nn.init.constant_(submodule.bias.data, 0)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.leaky_relu(self.fc4(x), 0.2)
        out = torch.tanh(self.fc_out(x))
        image = out.view(out.size(0), *self.image_size)
        return image


#discriminator : discriminate real image and fake image
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.image_size = params["image_size"]

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.image_size)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).apply(self.initialize_weight)

    def initialize_weight(self, submodule):
        if isinstance(submodule, torch.nn.Linear):
            nn.init.normal_(submodule.weight.data, 0.0, 0.02)
            nn.init.constant_(submodule.bias.data, 0)
        elif isinstance(submodule, torch.nn.BatchNorm1d):
            nn.init.normal_(submodule.weight.data, 1.0, 0.02)
            nn.init.constant_(submodule.bias.data, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x
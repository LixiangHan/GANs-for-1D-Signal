import torch.nn as nn
import torchvision.datasets as dataset


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
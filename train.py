import torch
import torchvision.datasets as dataset
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
from preprocessing import Dataset

epoch_num = 128
batch_size = 8


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainset = Dataset('./data', device)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    for epoch in range(epoch_num):
        for step, (x, _) in enumerate(trainloader):
            print("Epoch: %-3d | Step: %-3d | G Loss: %-.4f | D Loss: %-.4f" %
                  (epoch, step, 0, 0))


if __name__ == '__main__':
    main()

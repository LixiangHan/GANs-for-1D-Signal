import torch
import numpy as np
import os
import matplotlib.pyplot as plt


class Dataset():
    def __init__(self, root):
        self.root = root
        self.dataset = self.build_dataset()
        self.length = self.dataset.shape[1]
        self.minmax_normalize()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        step = self.dataset[:, idx]
        step = torch.unsqueeze(step, 0)
        # target = self.label[idx]
        target = 0  # only one class
        return step, target

    def build_dataset(self):
        '''get dataset of signal'''
        dataset = []
        for _file in os.listdir(self.root):
            sample = np.loadtxt(os.path.join(self.root, _file)).T
            dataset.append(sample)
        dataset = np.vstack(dataset).T
        dataset = torch.from_numpy(dataset).float()

        return dataset

    def minmax_normalize(self):
        '''return minmax normalize dataset'''
        for index in range(self.length):
            self.dataset[:, index] = (self.dataset[:, index] - self.dataset[:, index].min()) / (
                self.dataset[:, index].max() - self.dataset[:, index].min())


if __name__ == '__main__':
    dataset = Dataset('./data')
    plt.plot(dataset.dataset[:, 0].T)
    plt.show()

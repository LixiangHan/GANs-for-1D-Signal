import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from dcgan import Discriminator, Generator, weights_init
from preprocessing import Dataset


n_critic = 5
clip_value = 0.01
lr = 1e-4
epoch_num = 64
batch_size = 8
nz = 100  # length of noise
ngpu = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # load training data
    trainset = Dataset('./data')

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    
    # init netD and netG
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    netG = Generator(nz).to(device)
    netG.apply(weights_init)

    # used for visualizing training process
    fixed_noise = torch.randn(16, nz, 1, device=device)

    # optimizers
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

    for epoch in range(epoch_num):
        for step, (data, _) in enumerate(trainloader):
            # training netD
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            netD.zero_grad()

            noise = torch.randn(b_size, nz, 1, device=device)
            fake = netG(noise)

            loss_D = -torch.mean(netD(real_cpu)) + torch.mean(netD(fake))
            loss_D.backward()
            optimizerD.step()

            for p in netD.parameters():
                p.data.clamp_(-clip_value, clip_value)

            if step % n_critic == 0:
                # training netG
                noise = torch.randn(b_size, nz, 1, device=device)

                netG.zero_grad()
                fake = netG(noise)
                loss_G = -torch.mean(netD(fake))

                netD.zero_grad()
                netG.zero_grad()
                loss_G.backward()
                optimizerG.step()
            
            if step % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                    % (epoch, epoch_num, step, len(trainloader), loss_D.item(), loss_G.item()))

        # save training process
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            f, a = plt.subplots(4, 4, figsize=(8, 8))
            for i in range(4):
                for j in range(4):
                    a[i][j].plot(fake[i * 4 + j].view(-1))
                    a[i][j].set_xticks(())
                    a[i][j].set_yticks(())
            plt.savefig('./img/wgan_epoch_%d.png' % epoch)
            plt.close()
    # save model
    torch.save(netG, './nets/wgan_netG.pkl')
    torch.save(netD, './nets/wgan_netD.pkl')


if __name__ == '__main__':
    main()

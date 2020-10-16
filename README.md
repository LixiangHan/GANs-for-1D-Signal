# GANs-for-1D-Signal
## Introduction
This repo contains my pytorch implementations of several types of GANs, including DCGAN, WGAN and WGAN-GP, for 1-D signal. It was used to generate fake data of Raman spectra, which are typically used in Chemometrics as the fingerprints of materials.

<div align=center><img width="320" height="240" src="https://github.com/LixiangHan/GANs-for-1D-Signal/blob/main/img/Brilliant%20Blue.png"></div>

If you use these codes, please kindly cite the  this repository.

## Requirements

- python 3.7.8
- pytorch 1.6.0
- numpy 1.19.2
- matplotlib 3.3.0

## Experiment Result

### DCGAN

<div align=center><img width="320" height="320" src="https://github.com/LixiangHan/GANs-for-1D-Signal/blob/main/img/dcgan.gif"></div>

### WGAN

<div align=center><img width="320" height="320" src="https://github.com/LixiangHan/GANs-for-1D-Signal/blob/main/img/wgan.gif"></div>

### WGAN-GP

**NOTE:** RMSprop was used in the implementation of wgan-gp, rather than Adam, which was used in its original version, as it seems like Adam didn't work well in my applications.

<div align=center><img width="320" height="320" src="https://github.com/LixiangHan/GANs-for-1D-Signal/blob/main/img/wgan_gp.gif"></div>

## Usage

 

## Reference

[1] Nathan Inkawhich. DCGAN Tutorial [EB/OL]. https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html, 2020-10-14.

[2] Yangyangji. GAN-Tutorial [DB/OL]. https://github.com/Yangyangii/GAN-Tutorial, 2020-10-15.

[3]  mcclow12. wgan-gp-pytorch [DB/OL]. https://github.com/mcclow12/wgan-gp-pytorch


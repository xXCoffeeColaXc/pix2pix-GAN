# pix2pix-GAN
This repository hosts an implementation of the Pix2Pix Generative Adversarial Network (GAN) aimed at transforming semantic segmentation labels into realistic-looking images. 


## Original paper: link

## Architecture:

Generator:
- The encoder-decoder architecture consists of:
- encoder:
    C64-C128-C256-C512-C512-C512-C512
- decoder:
    CD512-CD512-CD512-C512-C256-C128-C64

Discriminator:
    C64-C128-C256-C512
- After last layer, a convulotion is applied to map to a 1-dimensional output, followed by a Sigmoid funtion.
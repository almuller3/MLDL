# Variational Autoencoder (VAE) on MNIST

This repository contains the implementation of a Variational Autoencoder (VAE) using PyTorch. The model is trained on the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits. VAEs are a class of generative models that learn to encode data into a latent space and then decode it back to reconstruct the original data.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Improving the Model](#improving-the-model)
- [Results](#results)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The Variational Autoencoder (VAE) is a generative model that learns a probabilistic mapping of input data to a latent space, and from this latent space, it reconstructs the input data. VAEs are particularly useful for generating new data samples, dimensionality reduction, and learning interpretable latent representations.

This project includes:

- A PyTorch implementation of the VAE.
- Training on the MNIST dataset.
- Visualization of the latent space and generated samples.

## Model Architecture

### Encoder
- **Input**: 784-dimensional vector (flattened 28x28 image).
- **Layers**:
  - Fully connected layer (784 -> 512).
  - Two separate fully connected layers for the mean (`mu`) and log-variance (`logvar`) of the latent space.
- **Output**: Mean and log-variance vectors for the latent space.

### Reparameterization Trick
- Samples a latent vector `z` from the distribution defined by the mean (`mu`) and log-variance (`logvar`), ensuring that the network can backpropagate through the stochastic sampling process.

### Decoder
- **Input**: Latent vector `z`.
- **Layers**:
  - Fully connected layer (latent_dim -> 512).
  - Fully connected layer (512 -> 784).
- **Output**: Reconstructed image (784-dimensional vector).

### Loss Function
- **Reconstruction Loss**: Binary Cross-Entropy (BCE) loss between the original and reconstructed images.
- **KL Divergence Loss**: Regularization term that forces the learned latent distribution to be close to a standard normal distribution.

## Prerequisites

- Python 3.7+
- PyTorch 1.6+
- torchvision
- numpy
- matplotlib

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/vae-mnist.git
    cd vae-mnist
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can run the VAE model on the MNIST dataset by executing the following command:

```bash
python train_vae.py

# Model Architecture: StyleGAN3

## Overview
This project implements a **StyleGAN3-based** Generative Adversarial Network (GAN) for high-fidelity face synthesis. StyleGAN3 introduces rotation and translation equivariance, eliminating "texture sticking" artifacts found in previous GANs.

## 1. Generator (`src/models/generator.py`)
The `StyleGAN3Generator` maps a latent vector $z$ to a generated image $G(z)$.

### Key Components:
-   **Mapping Network**: An 8-layer MLP that maps the input latent code $z  $ (512-dim) to an intermediate latent code $w  $ (512-dim). This disentangles the feature space.
-   **Synthesis Network**:
    -   **Fourier Features**: Replaces the constant learned input of StyleGAN2 with continuous Fourier features to support coordinate-based generation.
    -   **Synthesis Layers**: A series of modulated convolution layers. In StyleGAN3, these operations are designed to be alias-free using sophisticated signal processing (low-pass filtering/LRA).
    -   **ToRGB**: Converts the high-dimensional feature map to a 3-channel RGB image.
-   **Output**: A `(3, 256, 256)` tensor with values in `[-1, 1]`.

## 2. Discriminator (`src/models/discriminator.py`)
The `StyleGAN3Discriminator` tries to distinguish between real images from the dataset and fake images from the Generator.

### Key Components:
-   **Standard CNN**: A deep convolutional network that downsamples the image.
-   **Mini-batch Standard Deviation**: Increases variation by computing statistics across the batch.
-   **Output**: A scalar score (logit) indicating realness.

## 3. Loss Functions (`src/models/losses.py`)
We use the standard **Non-Saturating Logistic Loss** with **R1 Regularization**.

### Generator Loss ($L_G$)
-   **Objective**: Fool the discriminator.
-   **Formula**: $L_G = - _{z  } [ (D(G(z)))]$ (Softplus version)

### Discriminator Loss ($L_D$)
-   **Objective**: Correctly classify real vs. fake images.
-   **Formula**: $L_D =  _{x  _{data}} [ (1 + e^{-D(x)})] +  _{z  } [ (1 + e^{D(G(z))})]$ 

### R1 Regularization (Penalty)
-   **Objective**: Penalize the gradient of the discriminator on real data to ensure stability.
-   **Applied**: Every $d_{reg}$ steps (lazy regularization).
-   **Formula**: $rac{}{2}  _{x  _{data}} [||
abla D(x)||^2]$

## 4. Inference Pipeline
When deployed (`src/apis/inference.py`), we use the exported **ONNX** version of the Generator:
1.  **Input**: Latent vector `z` (1, 512).
2.  **Execution**: ONNX Runtime performs the forward pass.
3.  **Post-processing**:
    -   **Watermarking**: Invisible noise added for provenance (`src/apis/postprocessing.py`).
    -   **Denormalization**: `[-1, 1] -> [0, 255]`.

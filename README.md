# Basic Generative Adversarial Network (GAN) on Fashion-MNIST
**TensorFlow / Keras**

This repository demonstrates how to build and train a **basic Generative Adversarial Network (GAN)** from scratch using **TensorFlow/Keras** on the **Fashion-MNIST** dataset.

Unlike convolutional GANs (DCGANs), this project uses **fully connected (Dense) layers**, making it ideal for understanding **core GAN mechanics** before moving to more advanced architectures.

---

## 🚀 What This Project Covers

- Loading the Fashion-MNIST dataset
- Image preprocessing and normalization to `[-1, 1]`
- Building a `tf.data` pipeline with shuffling and batching
- Implementing:
  - Generator network (noise → image)
  - Discriminator network (image → real/fake)
  - Adversarial loss using Binary Cross Entropy
- Alternating training of generator and discriminator
- Visualizing generated images after each epoch

---

## 🧠 Why Start with a Basic GAN?

This project helps you:

- Understand GAN training dynamics
- Learn how generators and discriminators compete
- Debug GAN instability in a simple setting
- Build intuition before using CNN-based GANs (DCGAN, StyleGAN)

---

## 📦 Requirements

pip install tensorflow matplotlib

## 🏗️ Training Architecture

### Generator

Input: Random noise vector (latent space)

Dense + ReLU layers

Output: 28×28 image (flattened to 784)

Activation: tanh

### Discriminator

Input: Flattened image (784)

Dense + ReLU layers

Output: Single logit (real vs fake)

## 📉 Loss Functions

Binary Cross Entropy (from logits)

Generator Loss

Encourages discriminator to classify fake images as real

Discriminator Loss

Real images → 1

Generated images → 0

Optimizers:

Adam (1e-4) for both networks

## 🔍 Key Concepts Explained in Code

Adversarial training

Latent space sampling

Generator vs discriminator objectives

Binary Cross Entropy with logits

TensorFlow GradientTape

GAN instability & convergence behavior

## 🖼️ Visualization

After each epoch, the script displays:

A 4×4 grid of generated Fashion-MNIST images

Fixed noise seed for consistent visual comparison

This makes it easy to track training progress.

## 🧑‍🎓 Who Should Use This Repo?

Beginners learning GAN fundamentals

Students studying deep generative models

TensorFlow users wanting a clean GAN reference

Anyone preparing to move to DCGANs or diffusion models

## ⚠️ Important Notes

This is a Dense-only GAN (no convolutions)

Designed for education, not image quality

Training stability may vary (normal for GANs)

For better results, explore DCGANs or add batch normalization

## 📜 License

MIT License

## ⭐ Support

If this repository helped you understand GANs:

⭐ Star the repo

🧠 Share it with other ML learners

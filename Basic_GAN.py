"""
===========================================================
Basic GAN on Fashion-MNIST using TensorFlow/Keras
===========================================================

Overview:
- Implements a simple Generative Adversarial Network (GAN)
- Generator: Fully connected (Dense) layers that map random noise
             vectors (latent space) into 28x28 grayscale images.
- Discriminator: Fully connected (Dense) layers that classify
                 images as real (from Fashion-MNIST) or fake
                 (from Generator).

Dataset:
- Fashion-MNIST (Zalando Research)
- 70,000 grayscale images of clothing items (28x28 pixels)
- 10 categories: T-shirt/top, trouser, pullover, dress, coat,
                 sandal, shirt, sneaker, bag, ankle boot.

Training:
- Generator learns to produce realistic clothing images.
- Discriminator learns to distinguish real vs fake.
- Loss: Binary Cross Entropy
- Optimizer: Adam

Usage:
- Run the script to train the GAN.
- After each epoch, generated clothing images are displayed.
- This is a basic GAN (no convolutions), intended for learning
  and experimentation before moving to DCGANs or advanced models.

===========================================================
"""

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ---------------------------
# Dataset loading & prep
# ---------------------------
# Load Fashion-MNIST dataset from Keras datasets.
# The dataset returns tuples: (train_images, train_labels), (test_images, test_labels).
# We discard labels because GAN training is unsupervised here.
(train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

# Flatten 28x28 images to vectors of length 784 for Dense-based models.
# Convert to float32 for TensorFlow operations.
train_images = train_images.reshape(train_images.shape[0], 784).astype("float32")

# Normalize pixel values from [0, 255] to [-1, 1].
# Using tanh on the generator output commonly pairs with target scaling to [-1,1].
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]

# Create a tf.data.Dataset for batching and shuffling.
BUFFER_SIZE = 60000
BATCH_SIZE = 256
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# ---------------------------
# Generator model
# ---------------------------
def make_generator_model():
    """
    Build a simple generator model:
    - Input: latent vector of size 100 (noise).
    - Dense -> ReLU -> Dense -> ReLU -> Dense -> tanh
    - Output: 784 values representing a 28x28 image scaled to [-1,1].
    """
    model = tf.keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=(100,)),  # Project latent space to 256-dim
        layers.Dense(512, activation="relu"),                      # Increase capacity
        layers.Dense(784, activation="tanh")                       # Output image vector, tanh -> [-1,1]
    ])
    return model

# ---------------------------
# Discriminator model
# ---------------------------
def make_discriminator_model():
    """
    Build a simple discriminator model:
    - Input: flattened image vector of length 784.
    - Dense -> ReLU -> Dense -> ReLU -> Dense (logit)
    - Output: single logit value (higher -> more 'real').
    """
    model = tf.keras.Sequential([
        layers.Dense(512, activation="relu", input_shape=(784,)),  # First hidden layer
        layers.Dense(256, activation="relu"),                      # Second hidden layer
        layers.Dense(1)                                           # Final logit (no activation)
    ])
    return model

# ---------------------------
# Losses and optimizers
# ---------------------------
# Use Binary Cross Entropy loss with logits (safer numerical properties).
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Generator loss aims to fool the discriminator: wants discriminator(fake) -> ones.
def generator_loss(fake_output):
    # Compare discriminator outputs for generated images to label '1' (real).
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Discriminator loss: classify real as ones, fake as zeros.
def discriminator_loss(real_output, fake_output):
    # Loss on real images vs ones + loss on fake images vs zeros.
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Instantiate models and optimizers.
generator = make_generator_model()
discriminator = make_discriminator_model()
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ---------------------------
# Training parameters
# ---------------------------
EPOCHS = 50           # Number of training epochs
noise_dim = 100       # Latent vector (noise) dimension
num_examples = 16     # How many examples to generate for preview
# Fixed seed for visualization so generated examples are comparable across epochs.
seed = tf.random.normal([num_examples, noise_dim])

# ---------------------------
# Training step (single batch)
# ---------------------------
@tf.function
def train_step(images):
    """
    One training step:
    - Sample noise for the generator.
    - Generate fake images.
    - Compute discriminator outputs on real and fake images.
    - Compute generator and discriminator losses.
    - Compute gradients and update both networks' weights.
    Note: Using two GradientTapes to compute gradients independently.
    """
    # Sample random noise for the generator input.
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    # Record operations for automatic differentiation.
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate images from noise (generator forward pass).
        generated = generator(noise, training=True)

        # Discriminator forward passes on real and generated images.
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated, training=True)

        # Compute losses for generator and discriminator.
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Compute gradients for generator and discriminator.
    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply gradients to update model weights.
    generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

# ---------------------------
# Utility: generate and show images
# ---------------------------
def generate_and_save_images(model, epoch, test_input):
    """
    Given a generator `model` and a fixed `test_input` (noise seed),
    generate images and display them in a 4x4 grid (for num_examples=16).
    The generator output is in [-1,1], so convert back to [0,255] for display.
    """
    # Generate images from the test noise.
    predictions = model(test_input, training=False)

    # Setup matplotlib figure sized to show a 4x4 grid.
    fig = plt.figure(figsize=(4, 4))

    # Loop over generated examples and plot them.
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        # Reshape the flat vector back to 28x28 and rescale to [0,255].
        plt.imshow(predictions[i].numpy().reshape(28, 28) * 127.5 + 127.5, cmap='gray')
        plt.axis('off')  # Hide axis ticks for clearer visuals

    # Show the image grid. In a notebook, this will display inline.
    plt.show()

# ---------------------------
# Full training loop
# ---------------------------
def train(dataset, epochs):
    """
    Iterate through the dataset for `epochs` epochs:
    - For each batch, run `train_step`.
    - After each epoch, generate and display sample images.
    """
    for epoch in range(epochs):
        # Iterate over batches; dataset yields arrays shaped [batch_size, 784].
        for batch in dataset:
            train_step(batch)

        # After the epoch completes, generate sample images to observe progress.
        generate_and_save_images(generator, epoch + 1, seed)

# ---------------------------
# Entrypoint: run training
# ---------------------------
if __name__ == "__main__":
    # Start training when script is executed directly.
    train(dataset, EPOCHS)
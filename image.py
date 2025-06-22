import tensorflow as tf
from .keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
import os # For saving the model

# --- Configuration ---
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 100 # You can adjust this. More epochs for better quality, but takes longer.
NOISE_DIM = 100 # Dimension of the random noise vector
NUM_CLASSES = 10 # Digits 0-9

# --- 1. Load and Preprocess the MNIST Dataset ---
def load_and_preprocess_mnist():
    """Loads MNIST, normalizes images, and prepares labels."""
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # Reshape images to (batch, 28, 28, 1) for convolutional layers
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    # Normalize images to [-1, 1] for GANs (common practice)
    train_images = (train_images - 127.5) / 127.5 

    # One-hot encode labels for conditional input
    train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)

    # Create TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset

# --- 2. Model Architecture: Generator ---
# The Generator takes noise and a class label, outputs an image.
def make_generator_model():
    """Defines the Generator model architecture."""
    # Input for random noise
    noise_input = layers.Input(shape=(NOISE_DIM,))
    # Input for class label (one-hot encoded)
    label_input = layers.Input(shape=(NUM_CLASSES,))

    # Combine noise and label
    # Use a Dense layer to project and concatenate
    noise_dense = layers.Dense(7 * 7 * 128)(noise_input)
    noise_reshaped = layers.Reshape((7, 7, 128))(noise_dense)

    label_dense = layers.Dense(7 * 7 * 1)(label_input) # Project label to a small feature map
    label_reshaped = layers.Reshape((7, 7, 1))(label_dense)

    # Concatenate along the channel dimension
    merged_input = layers.Concatenate()([noise_reshaped, label_reshaped])

    # Deconvolutional layers to upsample to 28x28
    x = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(merged_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Output layer: 28x28x1 image with tanh activation for [-1, 1] range
    output_image = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)

    model = Model(inputs=[noise_input, label_input], outputs=output_image, name='Generator')
    return model

# --- 3. Model Architecture: Discriminator ---
# The Discriminator takes an image and a class label, outputs a probability (real/fake).
def make_discriminator_model():
    """Defines the Discriminator model architecture."""
    # Input for image
    image_input = layers.Input(shape=(28, 28, 1))
    # Input for class label (one-hot encoded)
    label_input = layers.Input(shape=(NUM_CLASSES,))

    # Combine image and label
    # Project label to a small feature map that matches image dimensions (approximately)
    label_dense = layers.Dense(28 * 28 * 1)(label_input) # Project label to a flat vector
    label_reshaped = layers.Reshape((28, 28, 1))(label_dense) # Reshape to image-like dimensions

    # Concatenate along the channel dimension
    merged_input = layers.Concatenate()([image_input, label_reshaped])

    # Convolutional layers to downsample and classify
    x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(merged_input)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    # Output layer: single probability (real/fake) with sigmoid activation
    output_decision = layers.Dense(1, activation='sigmoid')(layers.Flatten()(x))

    model = Model(inputs=[image_input, label_input], outputs=output_decision, name='Discriminator')
    return model

# --- Loss Functions and Optimizers ---
# Binary Cross-Entropy is standard for GANs
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False) # from_logits=False because Discriminator uses sigmoid

def discriminator_loss(real_output, fake_output):
    """Calculates the discriminator's loss."""
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    """Calculates the generator's loss (tries to fool the discriminator)."""
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers for Generator and Discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# --- Training Step ---
# Use tf.function for faster execution
@tf.function
def train_step(images, labels):
    """Performs one training step for both Generator and Discriminator."""
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        generated_images = generator([noise, labels], training=True)

        # Get discriminator output for real and fake images
        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        # Calculate losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Calculate gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# --- Training Loop ---
def train(dataset, epochs):
    """Main training loop."""
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for image_batch, label_batch in dataset:
            train_step(image_batch, label_batch)
        
        # Optionally, save and display generated images at intervals
        # (This part is often done to visually inspect training progress,
        # but can be omitted for speed if not required by specific criteria)
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, np.random.normal([10, NOISE_DIM]), tf.keras.utils.to_categorical(np.arange(10), NUM_CLASSES))

    # Save the trained generator model weights
    generator.save_weights('cgan_generator_weights.h5')
    print("\nGenerator model weights saved to 'cgan_generator_weights.h5'")

# --- Image Generation and Saving (for visualization during training) ---
def generate_and_save_images(model, epoch, test_noise, test_labels):
    """Generates and saves a grid of images."""
    predictions = model([test_noise, test_labels], training=False)
    # Denormalize images from [-1, 1] to [0, 1]
    predictions = (predictions * 0.5) + 0.5

    fig = plt.figure(figsize=(10, 10))
    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i + 1) # Adjust subplot based on number of generated images
        plt.imshow(predictions[i, :, :, 0] * 255, cmap='gray') # Display as grayscale 0-255
        plt.title(f"Digit: {np.argmax(test_labels[i])}")
        plt.axis('off')
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.close(fig) # Close figure to prevent memory leak if not displaying

# --- Main Execution ---
if __name__ == '__main__':
    # Initialize Generator and Discriminator
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # Load and prepare dataset
    train_dataset = load_and_preprocess_mnist()

    print("Starting CGAN training...")
    train(train_dataset, EPOCHS)
    print("CGAN training complete.")

    # You can then download 'cgan_generator_weights.h5' from your Colab environment
    # and use it in your Streamlit app.

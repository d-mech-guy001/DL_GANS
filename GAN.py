import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Creating the generator model
latent_dim = 100

generator = keras.Sequential([
    layers.Dense(7 * 7 * 256, input_dim=latent_dim),
    layers.Reshape((7, 7, 256)),
    layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Conv2D(1, (7, 7), activation='sigmoid', padding='same')
])

# Creating the discriminator model
discriminator = keras.Sequential([
    layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=(28, 28, 1)),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.4),
    layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
    layers.LeakyReLU(alpha=0.2),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

# Compiling the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

# Combining the GAN model
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.Model(gan_input, gan_output)

# Compiling the GAN
gan.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0002, beta_1=0.5))

# Loading the MNIST dataset
(train_images, _), (_, _) = keras.datasets.mnist.load_data()

# Preprocessing the data
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

# Training parameters
batch_size = 64
epochs = 5000
sample_interval = 50

# Training loop
for epoch in range(epochs):
    # Select a random batch of images
    idx = np.random.randint(0, train_images.shape[0], batch_size)
    real_images = train_images[idx]

    # Generate fake images
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_images = generator.predict(noise)

    # Train the discriminator
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real_labels)

    # Print the progress and save generated samples
    if epoch % sample_interval == 0:
        print(f"Epoch: {epoch} - D loss: {d_loss:.4f} - G loss: {g_loss:.4f}")

        # Generate and save a sample of generated images
        rows, cols = 5, 5
        noise = np.random.normal(0, 1, (rows * cols, latent_dim))
        generated_images = generator.predict(noise)

        fig, axs = plt.subplots(rows, cols)
        count = 0
        for i in range(rows):
            for j in range(cols):
                axs[i, j].imshow(generated_images[count, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                count += 1
        plt.show()

from random import randint

import numpy as np
from keras.datasets import mnist
from keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense, Reshape, Conv2DTranspose
from keras.models import Sequential
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from numpy import expand_dims, ones, zeros, vstack
from numpy.random import randn

from Project.gan import dataset as ds

trainX, trainY, testX, testY = ds.load_dataset("training_dataset.npz", "validation_dataset.npz")

# for i in range(9):
#     plt.subplot(3, 3, 1 + i)
#     plt.axis("off")
#     plt.imshow(trainX[i], cmap='gray')
#     plt.title(trainY[i])
# plt.show()


def define_discriminator(in_shape=(28, 28, 1)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same", input_shape=in_shape))
    model.add(LeakyReLU(alpha=.2))
    model.add(Dropout(.4))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Dropout(.4))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    opt = Adam(learning_rate=.0002, beta_1=.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # model.summary()
    # plot_model(model, to_file="discriminator_plot.png", show_shapes=True, show_layer_names=True)

    return model


def define_generator(latent_dim):
    model = Sequential()
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=.2))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=.2))
    model.add(Conv2D(1, (7, 7), activation="sigmoid", padding="same"))

    # model.summary()
    # plot_model(model, to_file="generator_plot.png", show_shapes=True, show_layer_names=True)

    return model


def define_gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(learning_rate=.0002, beta_1=.5)
    model.compile(loss="binary_crossentropy", optimizer=opt)

    return model


def load_real_samples():
    (X_train, _), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X = expand_dims(X_train, axis=-1)
    # X = X / 255
    return X


def generate_real_samples(dataset, n_samples):
    ix = [randint(0, dataset.shape[0] - 1) for _ in range(n_samples)]
    X = dataset[ix]
    y = ones((n_samples, 1))
    return X, y


def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_point(latent_dim, n_samples)
    X = g_model.predict(x_input)
    y = zeros((n_samples, 1))
    return X, y


def generate_latent_point(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


def train_gan(gan_model, latent_dim, n_epochs=100, n_batch=256):
    for i in range(n_epochs):
        x_gan = generate_latent_point(latent_dim, n_batch)
        y_gan = ones((n_batch, 1))

        gan_model.train_on_batch(x_gan, y_gan)


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
    # bat_per_epoch = int(dataset.shape[0] / n_batch)
    bat_per_epoch = 128
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):
        for j in range(bat_per_epoch):
            X_real, y_real = generate_real_samples(dataset, half_batch)
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            d_loss, _ = d_model.train_on_batch(X, y)
            X_gan = generate_latent_point(latent_dim, n_batch)
            y_gan = ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print(f'{i + 1}, {j + 1} {bat_per_epoch} d={d_loss:.3f}% g={g_loss:.3}%')
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim)


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print(f"{epoch + 1}: Accuracy real: {acc_real * 100:.0f}%, fake: {acc_fake * 100:.0f}%")
    save_plot(x_fake, epoch)
    filename = f"generator/generator_model_{epoch + 1}.h5"
    g_model.save(filename)


def save_plot(examples, epoch, n=10):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis("off")
        plt.imshow(examples[i, :, :, 0], cmap="gray_r")
        filename = f"figures/generated_plot_e{epoch + 1}.png"
        plt.savefig(filename)
    plt.close()


dim = 100
discriminator_model = define_discriminator()
generator_model = define_generator(dim)
gan_model = define_gan(generator_model, discriminator_model)
data = load_real_samples()
train(generator_model, discriminator_model, gan_model, data, dim)

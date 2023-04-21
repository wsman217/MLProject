import dis

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import Input
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.optimizers import Adam
from numpy.random import normal, randint
from tqdm import tqdm
from keras.layers import LeakyReLU, Dense, Dropout

from Project.gan import dataset


def build_generator():
    generator = Sequential()
    generator.add(Dense(units=256, input_dim=100))
    generator.add(LeakyReLU(.2))
    generator.add(Dense(units=512))
    generator.add(Dense(units=1024))
    generator.add(LeakyReLU(.2))
    generator.add(Dense(units=784, activation="tanh"))
    generator.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=.0002, beta_1=.5))
    return generator


def build_discriminator():
    discriminator = Sequential()
    discriminator.add(Dense(units=1024, input_dim=784))
    discriminator.add(LeakyReLU(.2))
    discriminator.add(Dropout(.2))
    discriminator.add(Dense(units=512))
    discriminator.add(LeakyReLU(.2))
    discriminator.add(Dropout(.3))
    discriminator.add(Dense(units=256))
    discriminator.add(LeakyReLU(.2))
    discriminator.add(Dropout(.3))
    discriminator.add(Dense(units=128))
    discriminator.add(LeakyReLU(.2))
    discriminator.add(Dense(units=1, activation="sigmoid"))
    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=.0002, beta_1=.5))

    return discriminator


def gan_net(generator, discriminator):
    # discriminator.trainable = False
    # inp = Input(shape=(100,))
    # X = generator(inp)
    # out = discriminator(X)
    # gan = Model(input=inp, outputs=out)
    # gan.compile(loss="binary_crossentropy", optimizer="adam")

    gan = Sequential()
    gan.add(generator)
    gan.add(discriminator)
    gan.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=.0002, beta_1=.5))

    return gan


def plot_images(generator, dim=(5, 5), figsize=(5, 5)):
    noise = normal(loc=0, scale=1, size=[dim[0] * dim[1], 100])
    generated_images = generator.predict(noise)

    generated_images = generated_images.reshape(dim[0] * dim[1], 28, 28)
    plt.figure(figsize=figsize)

    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.tight_layout()
    plt.show()


def train(X_train, epochs=5, batch_size=128):
    generator = build_generator()
    discriminator = build_discriminator()
    gan = gan_net(generator, discriminator)

    for epoch in range(1, epochs + 1):
        print("##### @ Epoch", epoch)

        for i in tqdm(range(batch_size)):
            noise = normal(0, 1, [batch_size, 100])
            generated_images = generator.predict(noise)
            image_batch = X_train[randint(low=0, high=X_train.shape[0], size=batch_size)].reshape(-1, 784)
            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 1.0

            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            noise = normal(0, 1, [batch_size, 100])
            y_gen = np.ones(batch_size)

            discriminator.trainable = False

            gan.train_on_batch(noise, y_gen)
            if i % 10 == 0:
                plot_images(generator)


# X_train, _, _, _ = dataset.load_dataset("training_dataset.npz", "validation_dataset.npz")
# X_train = X_train / 255
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train.reshape(-1, 784)

train(X_train)

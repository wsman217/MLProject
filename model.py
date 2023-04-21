import os

import tensorflow as tf
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import to_categorical

checkpoint_dir = "checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


def load_data():
    (x_train, y_train), (x_valid, y_valid) = mnist.load_data()
    x_train = x_train.reshape(-1, 784) / 255
    x_valid = x_valid.reshape(-1, 784) / 255
    y_train = to_categorical(y_train, 10)
    y_valid = to_categorical(y_valid, 10)
    return x_train, y_train, x_valid, y_valid


def define_model():
    model = Sequential()
    model.add(Dense(units=512, activation="relu", input_shape=(784,)))
    model.add(Dense(units=512, activation="relu"))
    model.add(Dense(units=10, activation="softmax"))
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def fit(model, x_train, y_train, validation_data, checkpoint, epochs, verbose):
    model.fit(x_train, y_train, validation_data=validation_data, epochs=epochs, verbose=verbose)
    checkpoint.save(checkpoint_prefix)


def define_checkpoint(model):
    checkpoint = tf.train.Checkpoint(model=model)
    return checkpoint


def load_checkpoint(checkpoint):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def restore_model():
    model = define_model()
    load_checkpoint(define_checkpoint(model))
    return model


def train(model, epochs=5, verbose=1):
    x_train, y_train, x_valid, y_valid = load_data()
    checkpoint = define_checkpoint(model)
    fit(model, x_train, y_train, (x_valid, y_valid), checkpoint, epochs, verbose)
    return model


def train_new():
    model = define_model()
    train(model)
    return model


def predict(model, data):
    return model.predict(data, verbose=0)


if __name__ == "__main__":
    mod = train_new()
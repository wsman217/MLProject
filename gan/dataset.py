import os

import numpy as np
from tensorflow import keras


def generate_dataset(path_to_root, output_name):
    filesCounter = 0
    labelCounter = 0
    images = []
    path_to_root = os.getcwd() + "/" + path_to_root + "/"
    for label in os.listdir(path_to_root):
        images_dir = path_to_root + label
        for image_file in os.listdir(images_dir):
            img = keras.utils.load_img(os.path.join(images_dir, image_file), grayscale=True)
            imgArr = keras.utils.img_to_array(img).reshape(-1)
            imgArr[imgArr == 0] = 1
            imgArr[imgArr == 255] = 0
            imgArr[imgArr == 1] = 255
            imgArr = np.insert(imgArr, 0, label)
            filesCounter += 1
            labelCounter += 1
            images.append(imgArr)
    np.savez(output_name + ".npz", *images)
    print("Found ", filesCounter, "files with ", labelCounter, "labels.")


def load_dataset(training_dataset, validation_dataset):
    training_labels, training_data = load_data(training_dataset)
    validation_labels, validation_data = load_data(validation_dataset)
    return training_data, training_labels, validation_data, validation_labels


def load_data(dataset):
    dataLabels = []
    dataArr = []
    with np.load(dataset) as data:
        for imgArr in data:
            dataLabels.append(data[imgArr][0])
            dataArr.append(data[imgArr][1:].reshape(28, 28))
    dataArr = np.array(dataArr)
    np.random.shuffle(dataArr)
    return dataLabels, dataArr


# generate_dataset("training", "training_dataset")
# generate_dataset("validation", "validation_dataset")
# training_data, training_labels, validation_data, validation_labels = load_dataset("training_dataset.npz",
#                                                                                   "validation_dataset.npz")
# print(training_data.shape, training_labels.shape, validation_data.shape, validation_labels.shape)

from PIL.Image import Image
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
import gzip
from numpy.lib.type_check import imag
from sklearn.preprocessing import OneHotEncoder
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools


#downgrade python version: virtualenv venv --python=python3.7.1


def getTrainLabels():
    with gzip.open('train-labels-idx1-ubyte.gz') as train_labels:
        data_from_train_file = train_labels.read()

    label_data = data_from_train_file[8:]
    assert len(label_data)==60000

    labels = [int(label_byte) for label_byte in label_data]
    assert min(labels) == 0 and max(labels) == 9
    assert len(labels) == 60000


def getTrainImages(images,SIZE_OF_ONE_IMAGE):
    with gzip.open('train-images-idx3-ubyte.gz') as train_images:
        train_images.read(4*4)
        ctr = 0
        for _ in range(60000):
            image = train_images.read(size=SIZE_OF_ONE_IMAGE)
            assert len(image) == SIZE_OF_ONE_IMAGE

            image_np = np.frombuffer(image,dtype='uint8')/255
            images.append(image_np)

    images = np.array(images)
    return images.shape


def plotImage(pixels:np.array):
    plt.imshow(pixels.reshape((28,28)),cmap = 'gray')
    plt.show()


if __name__ == '__main__':
    SIZE_OF_ONE_IMAGE = 28**2
    images = []
    getTrainLabels()
    getTrainImages(images,SIZE_OF_ONE_IMAGE)
    plotImage(images[24])

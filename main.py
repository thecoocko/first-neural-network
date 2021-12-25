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

    labels_np = np.array(labels).reshape((-1,1))
    encoder = OneHotEncoder(categories='auto')
    labels_np_onehot =  encoder.fit_transform(labels_np).toarray()

    return labels_np_onehot




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


def trainingCheck(imges,labels_np_onehot):
    x_train, x_test, y_train, y_test = train_test_split(images, labels_np_onehot)
    model = keras.Sequential()
    model.add(keras.layers.Dense(input_shape=(SIZE_OF_ONE_IMAGE,), units = 128, activation='relu'))
    model.add(keras.layers.Dense(10,activation='softmax'))
    
    model.summary()
    
    model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(np.array(x_train),np.array(y_train),epochs=20, batch_size=128)
    print(model.evaluate(np.array(x_test),np.array(y_test)))

    predicted_results = model.predict(x_test[1010].reshape((1, -1)))
    print(predicted_results)

    # predicted_outputs = np.argmax(model.predict(x_test),axis=1)
    # expected_outputs = np.argmax(y_test, axis=1)

    # predicted_confusion_matrix = confusion_matrix(expected_outputs,predicted_outputs)
    # print(f"matrix {predicted_confusion_matrix}")
    


def networkTrain(SIZE_OF_ONE_IMAGE,x_sample, y_sample):
    
    pass


if __name__ == '__main__':
    SIZE_OF_ONE_IMAGE = 28**2
    images = []
    print(getTrainLabels())
    getTrainImages(images,SIZE_OF_ONE_IMAGE)
    print(trainingCheck(images,getTrainLabels()))
    

    plotImage(images[1010])

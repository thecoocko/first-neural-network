import numpy as np
import matplotlib.pyplot as plt
import gzip
from sklearn.preprocessing import OneHotEncoder
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools


#virtualenv venv --python=python3.7.1


with gzip.open('train-labels-idx1-ubyte.gz') as train_labels:
    data_from_train_file = train_labels.read()

label_data = data_from_train_file[8:]
assert len(label_data)==60000

labels = [int(label_byte) for label_byte in label_data]
assert min(labels) == 0 and max(labels) == 9
assert len(labels) == 60000
print(labels)
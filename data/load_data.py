import os
import numpy as np
from tensorflow.keras.datasets import mnist 

def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)
    return train_images, train_labels, test_images, test_labels

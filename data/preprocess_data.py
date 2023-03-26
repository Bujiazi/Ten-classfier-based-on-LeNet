import numpy as np
import pickle
import os,sys
os.chdir(sys.path[0])
sys.path.append('../data')
sys.path.append('../../data')
os.getcwd()
from load_data import load_data

def preprocess_data(images, labels):
    # 对图像数据进行归一化操作
    images = images / 255.0

    # 对标签数据进行one-hot编码
    num_labels = len(np.unique(labels))
    labels_onehot = np.eye(num_labels)[labels]

    return images, labels_onehot

if __name__ == '__main__':
    # 载入原始的图像数据和标签数据
    train_images, train_labels, test_images, test_labels = load_data()

    # 对原始数据进行预处理
    train_images_preprocessed, train_labels_preprocessed = preprocess_data(train_images, train_labels)
    test_images_preprocessed, test_labels_preprocessed = preprocess_data(test_images, test_labels)

    # 将处理后的数据保存到文件中
    with open('data/train.pkl', 'wb') as f:
        pickle.dump((train_images_preprocessed, train_labels_preprocessed), f)
    with open('data/test.pkl', 'wb') as f:
        pickle.dump((test_images_preprocessed, test_labels_preprocessed), f)

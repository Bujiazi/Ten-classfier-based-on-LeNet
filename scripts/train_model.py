import numpy as np
import os,sys
os.chdir(sys.path[0])
sys.path.append('../data')
sys.path.append('../../data')
sys.path.append('../model')
sys.path.append('../../model')
os.getcwd()
import tensorflow as tf
from load_data import load_data
from lenet import create_model

def train_model():
    # 导入数据
    train_images, train_labels, test_images, test_labels = load_data()

    # 定义常量
    input_shape = train_images.shape[1:]
    num_classes = len(np.unique(train_labels))

    # 创建模型
    model = create_model(input_shape, num_classes)

    # 训练模型
    model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels))

    # 返回训练好的模型对象
    return model

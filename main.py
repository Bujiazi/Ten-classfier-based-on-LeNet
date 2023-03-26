import numpy as np
import os
import sys
os.chdir(sys.path[0])
sys.path.append('../data')
sys.path.append('../../data')
sys.path.append('../model')
sys.path.append('../../model')
current_dir = os.getcwd()

import tensorflow as tf
from data.load_data import load_data
from model.lenet import create_model

# 定义常量
input_shape = (28, 28, 1)
num_classes = 10
batch_size = 128
epochs = 10
save_dir = 'model/model_saver'

# 导入数据
train_images, train_labels, test_images, test_labels = load_data()
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# 创建模型
model = create_model(input_shape, num_classes)

# 训练模型
model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_labels))

# 保存模型
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
tf.keras.models.save_model(model, save_dir)

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

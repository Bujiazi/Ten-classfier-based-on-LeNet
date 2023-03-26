import numpy as np
import os
import sys
import tensorflow as tf
os.chdir(sys.path[0])
sys.path.append('../data')
sys.path.append('../../data')
sys.path.append('../model')
sys.path.append('../../model')
os.getcwd()
from load_data import load_data
from lenet import create_model

# 导入数据
train_images, train_labels, test_images, test_labels = load_data()
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
# 定义常量
input_shape = train_images.shape[1:]
num_classes =10

# 创建模型
model = create_model(input_shape, num_classes)

# 训练模型
model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_data=(test_images, test_labels))

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# 保存模型
save_dir = 'model_saver'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model.save(os.path.join(save_dir, 'my_model.h5'))

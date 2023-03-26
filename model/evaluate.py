import numpy as np
import os
import sys
import tensorflow as tf
from keras.models import load_model
os.chdir(sys.path[0])
sys.path.append('../data')
sys.path.append('../../data')
sys.path.append('../model/model_saver')
sys.path.append('../../model')
os.getcwd()

from load_data import load_data

# 导入数据
_, _, x_test, y_test = load_data()
y_test = tf.keras.utils.to_categorical(y_test)

# 加载模型
model = load_model('model_saver/my_model.h5', compile=False)

# 在测试集上评估模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
loss, accuracy = model.evaluate(x_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

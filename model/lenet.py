import tensorflow as tf

def create_model(input_shape, num_classes=10):
    # 定义一个顺序模型
    model = tf.keras.Sequential()

    # 添加一个卷积层
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

    # 添加一个池化层
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # 添加一个卷积层
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))

    # 添加一个池化层
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # 添加一个全连接层
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    # 添加一个输出层
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    # 编译模型
    model.compile(loss=tf.nn.softmax_cross_entropy_with_logits,
                  optimizer=tf.keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # 返回编译后的模型对象
    return model

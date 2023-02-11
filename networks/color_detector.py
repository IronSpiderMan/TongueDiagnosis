import os
import argparse

import numpy as np
from tensorflow import keras
from keras import layers
from datasets.tongue import get_data

# 设置命令行参数
parse = argparse.ArgumentParser()
parse.add_argument('--data', type=str, default='../dataset/color')
parse.add_argument('--savepath', type=str, default='../static/models/color_detector.keras')
parse.add_argument('--epoch', type=int, default=20)
# 解析命令行参数
args = parse.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ##表示使用GPU编号为0的GPU进行计算
# 获取数据
images, labels, cls2idx, idx2cls = get_data(args.data)
np.random.seed(12)
np.random.shuffle(images)
np.random.seed(12)
np.random.shuffle(labels)
# 数据增强
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2)
])
# 预训练模型
conv_base = keras.applications.resnet50.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(180, 180, 3)
)
conv_base.trainable = False
# 构建模型
inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = keras.applications.resnet50.preprocess_input(x)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(len(cls2idx), activation='softmax')(x)
model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
callbacks = [
    keras.callbacks.ModelCheckpoint(args.savepath, save_best_only=True)
]
model.fit(
    images,
    labels,
    batch_size=64,
    epochs=args.epoch,
    validation_split=0.1,
    callbacks=callbacks
)

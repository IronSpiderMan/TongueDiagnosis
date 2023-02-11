import argparse
import os
import cv2
import keras
import random
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras import layers

# 设置命令行参数
parse = argparse.ArgumentParser()
parse.add_argument('--data', type=str, default='../datasets/indentation')
parse.add_argument('--savepath', type=str, default='../static/models/indentation_detector.keras')
parse.add_argument('--epoch', type=int, default=20)
# 解析命令行参数
args = parse.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  ##表示使用GPU编号为0的GPU进行计算
# 获取数据
# 1、准备数据
positive_dir = f"{args.data}/1"
negative_dir = f"{args.data}/0"
positives = [os.path.join(positive_dir, file) for file in os.listdir(positive_dir)]
negatives = [os.path.join(negative_dir, file) for file in os.listdir(negative_dir)]
negatives = random.sample(negatives, len(positives))
X = []
y = []
for idx, x in enumerate(positives + negatives):
    if idx >= 1472:
        y.append(0)
    else:
        y.append(1)
    img = cv2.resize(cv2.imread(x), (180, 180))
    X.append(img)
X = np.array(X)
y = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)
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
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
callbacks = [
    keras.callbacks.ModelCheckpoint(args.savepath, save_best_only=True)
]
model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=args.epoch,
    validation_data=[X_test, y_test],
    callbacks=callbacks
)

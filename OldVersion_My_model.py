import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as K

import matplotlib.pyplot as plt
import numpy as np
from random import randint

import cv2

import string
import Levenshtein

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# =================函数=================


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# def gen(batch_size=64):
#     # 魔改版
#     # 每次从训练集中拿出batch_size个图片扔进去
#     X_part = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
#     y_part = np.zeros((batch_size, n_len), dtype=np.uint8)
#
#     while True:
#         for i in range(batch_size):
#             X_part[i] = X_train[randint(0, train_size - 1)]
#             y_part[i] = y_train[randint(0, train_size - 1)]
#         # 使用 yield 产生一个生成器，每次都生成一个X，y，rnn的长度，输入的长度，以及一个问号矩阵
#         yield [X_part, y_part, np.ones(batch_size) * rnn_length, np.ones(batch_size) * n_len], np.ones(batch_size)


def evaluate():
    # 读入验证集的数据
    val_file = open(VAL_PATH, encoding='utf-8')
    lines = val_file.readlines()
    val_size = len(lines)

    X_val = np.zeros((val_size, width, height, 3), dtype=np.uint8)
    y_val = np.zeros((val_size, n_len), dtype=np.uint8)
    label_len = np.zeros((val_size,), dtype=np.uint8)

    t = 0
    for line in lines:
        path = line.split(' ')[0]
        path = IMG_PATH + path[1:]
        s1 = path.rfind('_')
        s2 = path.rfind('_', 0, s1)
        label = path[s2 + 1:s1]
        label_len[t] = len(label)
        y_val[t][0:len(label)] = [characters.find(x) for x in label]
        img = cv2.resize(cv2.imread(path), (width, height))
        X_val[t] = np.array(img).transpose(1, 0, 2)
        t += 1

    y_pred = base_model.predict(X_val)
    y_pred = y_pred[:, 2:, :]
    shape = y_pred.shape
    ctc_decode = K.ctc_decode(y_pred, input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = K.get_value(ctc_decode)[:, :n_len]
    # out 即为目前预测的值
    all_acc = 0  # 用来存储每个字符串的匹配度之和

    for i in range(val_size):
        str1 = out[i][:label_len[i]]
        str2 = y_val[i][:label_len[i]]
        dis = Levenshtein.distance(str1, str2)
        all_acc += (1.0 - dis / len(str1))

    return all_acc / val_size


class Evaluator(Callback):
    def __init__(self):
        super().__init__()
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate() * 100
        self.accs.append(acc)
        print('')
        print('acc: %f%%' % acc)


# ================基本参数===============

# 开始为 170 80
width, height = 256, 64
characters = string.digits + string.ascii_lowercase + string.ascii_uppercase
n_len = 18
# len 最长为18
n_class = len(characters) + 1

EPOCH = 20
IMG_PATH = "/Users/rune/PycharmProjects/RNN_book/90kDICT32px"
TRAIN_PATH = "/Users/rune/PycharmProjects/RNN_book/90kDICT32px/annotation_train.txt"
VAL_PATH = "/Users/rune/PycharmProjects/RNN_book/90kDICT32px/annotation_val.txt"

rnn_size = 128

# =================读入数据=================

f = open(TRAIN_PATH, encoding='utf-8')
lines = f.readlines()
train_size = len(lines)

X_train = np.zeros((train_size, width, height, 3), dtype=np.uint8)
y_train = np.zeros((train_size, n_len), dtype=np.uint8)

t = 0
# 训练数据
for line in lines:
    path = line.split(' ')[0]
    path = IMG_PATH + path[1:]
    s1 = path.rfind('_')
    s2 = path.rfind('_', 0, s1)
    label = path[s2 + 1:s1]
    y_train[t][0:len(label)] = [characters.find(x) for x in label]
    img = cv2.resize(cv2.imread(path), (width, height))
    X_train[t] = np.array(img).transpose(1, 0, 2)
    t += 1

rnn_size = 128

input_tensor = Input((width, height, 3))
x = input_tensor
x = Lambda(lambda x: (x - 127.5) / 127.5)(x)
for i in range(3):
    for j in range(2):
        x = Conv2D(32 * 2 ** i, 3, kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    if i == 0 or i == 1:
        x = MaxPooling2D((2, 2))(x)
    else:
        x = MaxPooling2D((1, 2))(x)

conv_shape = x.get_shape().as_list()
rnn_length = conv_shape[1]
rnn_dimen = conv_shape[2] * conv_shape[3]
print(conv_shape, rnn_length, rnn_dimen)
x = Reshape(target_shape=(rnn_length, rnn_dimen))(x)
rnn_length -= 2

x = Dense(rnn_size, kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)

gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_uniform', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, kernel_initializer='he_uniform',
             go_backwards=True, name='gru1_b')(x)
x = add([gru_1, gru_1b])

gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_uniform', name='gru2')(x)
gru_2b = GRU(rnn_size, return_sequences=True, kernel_initializer='he_uniform',
             go_backwards=True, name='gru2_b')(x)
x = concatenate([gru_2, gru_2b])

x = Dropout(0.2)(x)
x = Dense(n_class, activation='softmax')(x)
base_model = Model(inputs=input_tensor, outputs=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                  name='ctc')([x, labels, input_length, label_length])

model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

model.summary()

evaluator = Evaluator()

Train_data = [X_train, y_train, np.ones(train_size) * rnn_length, np.ones(train_size) * n_len]

h = model.fit(Train_data, np.ones(train_size), epochs=10, batch_size=64, callbacks=[evaluator])

# h = model.fit(gen(64), steps_per_epoch=32, epochs=10,
#               callbacks=[evaluator],
#               # validation_data=gen(64),
#               # validation_steps=20
#               )

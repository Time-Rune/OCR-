import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as K
from sklearn.model_selection import train_test_split

from keras.layers.wrappers import Bidirectional

import matplotlib.pyplot as plt
import numpy as np
from random import randint

import cv2

import string
import Levenshtein
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# =================函数=================


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# ================基本参数===============

width, height = 224, 32
characters = string.digits + string.ascii_letters
n_len = 22
n_class = len(characters) + 1

EPOCH = 20
Data_path = '/Users/rune/PycharmProjects/RNN_book/90kDICT32px/dataset'

rnn_size = 256

# =================读入数据=================

input_tensor = Input((width, height, 3))
x = input_tensor

# VGG 16 part
x = Conv2D(64, 3, padding='same', activation='relu', name='conv1')(x)
x = MaxPool2D(pool_size=2, padding='same', name='pool1')(x)

x = Conv2D(128, 3, padding='same', activation='relu', name='conv2')(x)
x = MaxPool2D(pool_size=2, padding='same', name='pool2')(x)

x = Conv2D(256, 3, padding='same', use_bias=False, name='conv3')(x)
x = BatchNormalization(name='bn3')(x)
x = Activation('relu', name='relu3')(x)
x = Conv2D(256, 3, padding='same', activation='relu', name='conv4')(x)
x = MaxPool2D(pool_size=2, strides=(1, 2), padding='same', name='pool4')(x)

x = Conv2D(512, 3, padding='same', use_bias=False, name='conv5')(x)
x = BatchNormalization(name='bn5')(x)
x = Activation('relu', name='relu5')(x)
x = Conv2D(512, 3, padding='same', activation='relu', name='conv6')(x)
x = MaxPool2D(pool_size=2, strides=(1, 2), padding='same', name='pool6')(x)

x = Conv2D(512, 2, use_bias=False, name='conv7')(x)
x = BatchNormalization(name='bn7')(x)
x = Activation('relu', name='relu7')(x)

conv_shape = x.get_shape().as_list()
rnn_length = conv_shape[1]
rnn_dimen = conv_shape[2] * conv_shape[3]
# print(conv_shape, rnn_length, rnn_dimen)
x = Reshape(target_shape=(rnn_length, rnn_dimen))(x)
rnn_length -= 2

x = Dense(rnn_size, kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)

# 双向GRU层
gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_uniform', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, kernel_initializer='he_uniform',
             go_backwards=True, name='gru1_b')(x)
x = add([gru_1, gru_1b])

gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_uniform', name='gru2')(x)
gru_2b = GRU(rnn_size, return_sequences=True, kernel_initializer='he_uniform',
             go_backwards=True, name='gru2_b')(x)
x = concatenate([gru_2, gru_2b])

# x = Bidirectional(LSTM(rnn_size, return_sequences=True), input_shape=(rnn_length, rnn_dimen), merge_mode='concat')(x)

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

model.load_weights("/Users/rune/PycharmProjects/RNN_book/Adam_100epoch.h5")


def Predict(img):

    # img = cv2.resize(img, (width, height))

    ratio = height / len(img)
    img = cv2.resize(img, (int(len(img[0]) * ratio), height))

    if len(img[0]) >= width:
        img = cv2.resize(img, (width, height))
    else:
        img = cv2.copyMakeBorder(img, top=0, bottom=0, left=0, right=width - len(img[0]),
                                 borderType=cv2.BORDER_CONSTANT)

    # plt.imshow(img)
    # plt.show()

    X = np.ones((1, width, height, 3), dtype=np.uint8)
    X[0] = np.array(img).transpose(1, 0, 2)

    y_pred = base_model.predict(X)
    y_pred = y_pred[:, 2:, :]
    shape = y_pred.shape
    ctc_decode = K.ctc_decode(y_pred, input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = K.get_value(ctc_decode)[:, :n_len]
    Answer = ''
    now_len = 19

    # print(out[0])

    while out[0][now_len] == 0:
        now_len -= 1
    for i in range(now_len + 1):
        Answer += characters[out[0][i]]
    return Answer


if __name__ == '__main__':
    path = "/Users/rune/PycharmProjects/RNN_book/img.png"
    print(Predict(cv2.imread(path)))

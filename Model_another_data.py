import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as K
from sklearn.model_selection import train_test_split

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


def evaluate():
    # 读入验证集的数据
    print("开始预测")
    y_pred = base_model.predict(X)
    print("预测完成")
    y_pred = y_pred[:, 2:, :]
    shape = y_pred.shape
    ctc_decode = K.ctc_decode(y_pred, input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = K.get_value(ctc_decode)[:, :n_len]
    # out 即为目前预测的值
    all_acc = 0  # 用来存储每个字符串的匹配度之和
    val_size = len(y)

    for i in range(val_size):
        str1 = out[i][:z[i]]
        str2 = y[i][:z[i]]
        print("Ori: ", end=' ')
        for j in range(z[i]):
            print(characters[str2[j]], end='')
        print("\n Pre: ", end=' ')
        for j in range(z[i]):
            print(characters[str1[j]], end='')
        print('\n')

        # if str1 == str2:
        #     all_acc += 1

        dis = Levenshtein.distance(str1, str2)
        if dis == 0:
            all_acc += 1

    print("Acc: ")
    print(str(all_acc) + '/' + str(val_size))


# ================基本参数===============

# 开始为 170 80
width, height = 224, 32
characters = string.digits + string.ascii_lowercase + string.ascii_uppercase
n_len = 22
# len 最长为18
n_class = len(characters) + 1

EPOCH = 20
Data_path = '/Users/rune/PycharmProjects/RNN_book/90kDICT32px/dataset'

rnn_size = 256

# ===========读入需要预测的数据============

Test_path = '/Users/rune/PycharmProjects/RNN_book/ch4_training_word_images_gt/gt.txt'

f = open(Test_path, encoding='UTF-8-sig')
lines = f.readlines()
size = len(lines)

X = np.ones((size, 224, 32, 3), dtype=np.uint8)
y = np.ones((size, 20), dtype=np.uint8)
z = np.ones((size,), dtype=np.uint8)

t = 0
for line in lines:
    name, word = line.split(' ')[0][:-1], line.split(' ')[1][1:-2]
    img_path = '/Users/rune/PycharmProjects/RNN_book/ch4_training_word_images_gt/' + name

    y[t][0:len(word)] = [characters.find(x) for x in word]
    img = cv2.resize(cv2.imread(img_path), (width, height))
    X[t] = np.array(img).transpose(1, 0, 2)
    z[t] = len(word)
    t += 1


# =================构建模型=================

input_tensor = Input((width, height, 3))
x = input_tensor
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

model.load_weights("/Users/rune/PycharmProjects/RNN_book/NewBorn_model2.h5")

evaluate()

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
    y_pred = base_model.predict(X_val[:20])
    y_pred = y_pred[:, 2:, :]
    shape = y_pred.shape
    ctc_decode = K.ctc_decode(y_pred, input_length=np.ones(shape[0]) * shape[1])[0][0]
    out = K.get_value(ctc_decode)[:, :n_len]
    # out 即为目前预测的值
    all_acc = 0  # 用来存储每个字符串的匹配度之和
    val_size = len(y_val)

    val_size = 20  # test

    fig, ax = plt.subplots(5, 4)

    for i in tqdm(range(val_size)):
        print("original string: ", end=' ')
        now_len = 19
        while y_val[i][now_len] == 0:
            now_len -= 1
        s2 = ''
        for j in range(0, now_len + 1):
            print(characters[y_val[i][j]], end='')
            s2 += characters[y_val[i][j]]
        print(' ')
        print("Predict string: ", end=' ')
        now_len = 19
        while out[i][now_len] == 0:
            now_len -= 1
        s1 = ''
        for j in range(0, now_len + 1):
            print(characters[out[i][j]], end='')
            s1 = s1 + characters[out[i][j]]
        print(' ')

        ax[i // 4, i % 4].imshow(X_val[i].transpose(1, 0, 2))
        ax[i // 4, i % 4].set_title("Pre:" + s1 + "\n Ori:" + s2)

        str1 = out[i][:z_val[i]]
        str2 = y_val[i][:z_val[i]]
        dis = Levenshtein.distance(str1, str2)
        all_acc += (1.0 - dis / len(str1))
        # if dis == 0:
        #     all_acc += 1

    plt.savefig("Final_result.png")
    plt.show()

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
width, height = 224, 32
characters = string.digits + string.ascii_lowercase + string.ascii_uppercase
n_len = 22
# len 最长为18
n_class = len(characters) + 1

EPOCH = 20
Data_path = '/Users/rune/PycharmProjects/RNN_book/90kDICT32px/dataset'

rnn_size = 256

# =================读入数据=================

size = 0

for i in range(30, 33):
    files = os.listdir(Data_path + r"/%d" % i)
    for file in files:
        if file == '.DS_Store':
            continue
        f2 = os.listdir(Data_path + r"/%d/" % i + file)
        size += len(f2)

X = np.zeros((size, width, height, 3), dtype=np.uint8)
y = np.zeros((size, n_len), dtype=np.uint8)
z = np.zeros((size, ), dtype=np.uint8)

t = 0
# 训练数据

for i in tqdm(range(30, 33)):
    files = os.listdir(Data_path + r"/%d" % i)
    for file in files:
        if file == '.DS_Store':
            continue
        f2 = os.listdir(Data_path + r"/%d/" % i + file)
        for f3 in f2:
            img_path = Data_path + r"/%d" % i + "/" + file + "/" + f3
            s1 = f3.rfind("_")
            s2 = f3.rfind("_", 0, s1)
            label = f3[s2 + 1: s1]
            # print(img_path)
            # print(f3[s2 + 1: s1])
            y[t][0:len(label)] = [characters.find(x) for x in label]
            z[t] = len(label)

            img = cv2.imread(img_path)
            ratio = height / len(img)
            img = cv2.resize(img, (int(len(img[0]) * ratio), height))
            if len(img[0]) >= width:
                img = cv2.resize(img, (width, height))
            else:
                img = cv2.copyMakeBorder(img, top=0, bottom=0, left=0, right=width - len(img[0]),
                                         borderType=cv2.BORDER_CONSTANT)
            X[t] = np.array(img).transpose(1, 0, 2)

            # plt.imshow(X[t])
            # plt.show()
            # if t == 6:
            #     exit()

            t += 1

# X = np.load('X.npy')
# y = np.load('y.npy')
# z = np.load('z.npy')

X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(X, y, z, test_size=0.2)
# X_val = X
# y_val = y
# z_val = z

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

# train_size = len(X_train)
# Train_data = [X_train, y_train, np.ones(train_size) * rnn_length, np.ones(train_size) * n_len]

model.load_weights("/Users/rune/PycharmProjects/RNN_book/Adam_100epoch.h5")

# h = model.fit(Train_data, np.ones(train_size), epochs=20, batch_size=64, callbacks=[evaluator])
#
# model.save_weights("NewBorn_model.h5")

print(evaluate())

# h = model.fit(gen(64), steps_per_epoch=32, epochs=10,
#               callbacks=[evaluator],
#               # validation_data=gen(64),
#               # validation_steps=20
#               )


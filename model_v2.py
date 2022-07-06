# from pynvml import *
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import string

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

digits = string.digits
operators = '+-*'
characters = digits + operators + '()'
print(characters)

width, height, n_len, n_class = 180, 60, 7, len(characters) + 1
print(n_class)


def generate():
    seq = ''
    k = random.randint(0, 2)

    if k == 1:
        seq += '('
    seq += random.choice(digits)
    seq += random.choice(operators)
    if k == 2:
        seq += '('
    seq += random.choice(digits)
    if k == 1:
        seq += ')'
    seq += random.choice(operators)
    seq += random.choice(digits)
    if k == 2:
        seq += ')'

    return seq


generate()
# 定义
# CTC
# Loss
from keras import backend as K


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# 定义网络结构
from keras.layers import *
from keras.models import *

# from make_parallel import make_parallel

rnn_size = 128

input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(3):
    x = Conv2D(32 * 2 ** i, (3, 3), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32 * 2 ** i, (3, 3), kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

conv_shape = x.get_shape()
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

x = Dense(128, kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
             name='gru1_b')(x)
gru1_merged = add([gru_1, gru_1b])

gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal',
             name='gru2_b')(gru1_merged)
x = concatenate([gru_2, gru_2b])
x = Dropout(0.25)(x)
x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)
base_model = Model(inputs=input_tensor, outputs=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=(1,), dtype='int64')
label_length = Input(name='label_length', shape=(1,), dtype='int64')
loss_out = Lambda(ctc_lambda_func, name='ctc')([base_model.output, labels, input_length, label_length])

model = Model(inputs=(input_tensor, labels, input_length, label_length), outputs=loss_out)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')


# 定义数据生成器

def gen(batch_size=128):
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.int32)
    label_length = np.ones(batch_size)

    generator = ImageCaptcha(width=width, height=height)

    while True:
        for i in range(batch_size):
            random_str = generate()
            X[i] = np.array(generator.generate_image(random_str)).transpose(1, 0, 2)
            y[i, :len(random_str)] = [characters.find(x) for x in random_str]
            y[i, len(random_str):] = -1
            label_length[i] = len(random_str)
        yield [X, y, np.ones(batch_size) * int(conv_shape[1] - 2), label_length], np.ones(batch_size)


[X_test, y_test, _, label_length_test], _ = next(gen(1))
plt.imshow(X_test[0].transpose(1, 0, 2))
plt.title(''.join([characters[x] for x in y_test[0]]))
# 验证函数和回调函数
from tqdm import tqdm
import pandas as pd
import cv2

# df = pd.read_csv('../../image_contest_level_1/labels.txt', sep=' ', header=None)
# n_test = 100000


# X_test = np.zeros((n_test, width, height, 3), dtype=np.uint8)
# y_test = np.zeros((n_test, n_len), dtype=np.int32)
# label_length_test = np.zeros((n_test, 1), dtype=np.int32)

# for i in tqdm(range(n_test)):
#     img = cv2.imread('../../image_contest_level_1/%d.png' % i)
#     X_test[i] = img[:, :, ::-1].transpose(1, 0, 2)
#     random_str = df[0][i]
#     y_test[i, :len(random_str)] = [characters.find(x) for x in random_str]
#     y_test[i, len(random_str):] = -1
#     label_length_test[i] = len(random_str)


def evaluate(model):
    y_pred = base_model2.predict(X_test, batch_size=1024)
    shape = y_pred[:, 2:, :].shape
    out = K.get_value(K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:, :n_len]
    if out.shape[1] > 4:
        return (y_test == out).all(axis=-1).mean()
    return 0


from keras.callbacks import *


class Evaluate(Callback):
    def __init__(self):
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model) * 100
        self.accs.append(acc)
        print('val_acc: %f%%' % acc)


evaluator = Evaluate()
# 训练
from keras.optimizers import *
from keras.callbacks import *

# batch_size = 1024
batch_size = 128
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
h = model.fit(gen(batch_size),
              validation_data=gen(32),
              validation_steps=20,
              steps_per_epoch=10000 / batch_size, epochs=50,
              callbacks=[evaluator])
model.save('New_model.h5')
# plt.plot(range(len(h.history['loss'][10:])), h.history['loss'][10:])
# plt.plot(range(len(h.history['loss'][10:])), h.history['val_loss'][10:])

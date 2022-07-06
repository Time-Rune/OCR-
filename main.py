import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras import backend as K

from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random

import string

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def gen(batch_size=64):
    # 每次生成一个 batch_size 大小的数据，送入训练集
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    # 使用numpy矩阵存放数据
    # 可以参考这个样式制作数据集

    generator = ImageCaptcha(width=width, height=height)

    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(n_len)])
            # 生成一段随机的序列
            X[i] = np.array(generator.generate_image(random_str)).transpose(1, 0, 2)
            # 将X[i]转置为竖着的图片
            y[i] = [characters.find(x) for x in random_str]
        # 使用 yield 产生一个生成器，每次都生成一个X，y，rnn的长度，输入的长度，以及一个问号矩阵
        # return [X, y, np.ones(batch_size) * rnn_length, np.ones(batch_size) * n_len], np.ones(batch_size)
        yield [X, y, np.ones(batch_size) * rnn_length, np.ones(batch_size) * n_len], np.ones(batch_size)
        # 前四个一起组成了X, 后面一个空矩阵构成y
        #


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # y_pred = y_pred[:, 2:, :]
    # keras 自导ctc loss函数，需要用lambda层进行层封装
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# 重写evaluate函数，对预判结果进行评估
def evaluate(batch_size=10, steps=4):
    batch_acc = 0
    generator = gen(batch_size)
    for i in range(steps):
        [X_test, y_test, _, _], _ = next(generator)
        # 用生成器获取当10个新的数据
        y_pred = base_model.predict(X_test)
        shape = y_pred.shape
        # pred为模型对此X的预测效果
        ctc_decode = K.ctc_decode(y_pred, input_length=np.ones(shape[0]) * shape[1])[0][0]
        # 使用内置的decode 对y_pred 进行编码

        out = K.get_value(ctc_decode)[:, :n_len]
        ctc = K.get_value(ctc_decode)
        print(ctc)

        # print("生成的y_test")
        # print(y_test)
        #
        # print("模型预测的y")
        # print(out)
        #
        # print("y_test == out的结果")
        # print(y_test == out)
        # print('\n')

        if out.shape[1] == n_len:
            # 长度对上了才能算
            # out.shape 就是长度
            # 这里好像是全部匹配上才算ok
            batch_acc += (y_test == out).all(axis=1).mean()
    return batch_acc / steps


class Evaluator(Callback):
    def __init__(self):
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(steps=64) * 100
        self.accs.append(acc)
        print('')
        print('acc: %f%%' % acc)


rnn_size = 128

characters = string.digits + string.ascii_uppercase + string.ascii_lowercase

width, height, n_len, n_class = 170, 80, 4, len(characters) + 1

# 将evaluator类实例化，进而作为callback的参数传进去
evaluator = Evaluator()


input_tensor = Input((width, height, 3))
x = input_tensor

# 一个简单的预处理
x = Lambda(lambda X: (X - 127.5) / 127.5)(x)

# 搭建VGG16模型
for i in range(3):
    for j in range(2):
        x = Conv2D(32 * 2 ** i, 3, kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)

conv_shape = x.get_shape().as_list()
rnn_length = conv_shape[1]
rnn_dimen = conv_shape[2] * conv_shape[3]
print(conv_shape, rnn_length, rnn_dimen)

"""
从 Input 到 最后一个 MaxPooling2D，是一个很深的卷积神经网络，它负责学习字符的各个特征，尽可能区分不同的字符。

它输出 shape 是 [None, 17, 6, 128]，这个形状相当于把一张宽为 170，高为 80 的彩色图像 (170, 80, 3)，压缩为宽为 17，高为 6 的 128维特征的特征图 (17, 6, 128)。

然后我们把图像 reshape 成 (17, 768)，也就是把高和特征放在一个维度，然后降维成 (17, 128)，也就是从左到右有17条特征，每个特征128个维度。

这128个维度就是这一条图像的非常高维，非常抽象的概括，然后我们将17个特征向量依次输入到 GRU 中，GRU 有能力学会不同特征向量的组合会代表什么字符，即使是字符之间有粘连也不会怕。这里使用了双向 GRU，

最后 Dropout 接一个全连接层，作为分类器输出每个字符的概率。

这个是 base_model 的结构，也是我们模型的结构。那么后面的 labels, input_length, label_length 和 loss_out 都是为了输入必要的数据来计算 CTC Loss 的。

"""

# 将x reshape为 length*dimen的矩阵
x = Reshape(target_shape=(rnn_length, rnn_dimen))(x)
rnn_length -= 2

x = Dense(rnn_size, kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.2)(x)

# RNN部分
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

# base_model 用dense进行输出，用来训练参数
# 输入是x

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
# shape = [1] 表示输入的是一个1*1的值
# shape = [n_len] 表示输入的是一个长为n_len的一维数组

loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                  name='ctc')([x, labels, input_length, label_length])


model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
# model 以loss作为输出，用于训练参数

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

# h = model.fit_generator(gen(64), steps_per_epoch=80, epochs=20,
#                         callbacks=[evaluator],
#                         validation_data=gen(64), validation_steps=20)
# 暂时不fit
h = model.fit(gen(64), steps_per_epoch=128, epochs=20,
              callbacks=[evaluator], validation_data=gen(64), validation_steps=20)

model.save("Rnn_model.h5")
# lis, non = gen(1)
#
# for cha in lis[1][0]:
#     print(characters[cha], end=' ')
#
# plt.figure()
# plt.imshow(lis[0][0])
# plt.show()

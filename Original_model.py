from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from keras import backend as K
import string
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
from keras.callbacks import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

characters = string.digits + string.ascii_uppercase
print(characters)

width, height, n_len, n_class = 170, 80, 4, len(characters) + 1

generator = ImageCaptcha(width=width, height=height)
random_str = ''.join([random.choice(characters) for j in range(4)])
img = generator.generate_image(random_str)

plt.imshow(img)
plt.title(random_str)


# 使用循环神经网络，所以默认丢掉前面两个输出，因为它们通常无意义，且会影响模型的输出。


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# 构建模型

rnn_size = 128

input_tensor = Input((width, height, 3))
x = input_tensor
x = Lambda(lambda x: (x - 127.5) / 127.5)(x)
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


def gen(batch_size=128):
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)
    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(n_len)])
            X[i] = np.array(generator.generate_image(random_str)).transpose(1, 0, 2)
            y[i] = [characters.find(x) for x in random_str]
        yield [X, y, np.ones(batch_size) * rnn_length, np.ones(batch_size) * n_len], np.ones(batch_size)


def evaluate(batch_size=2, steps=1):
    batch_acc = 0
    generator = gen(batch_size)
    for i in range(steps):
        [X_test, y_test, _, _], _ = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:, 2:, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
        out = K.get_value(ctc_decode)[:, :n_len]
        if out.shape[1] == n_len:
            batch_acc += (y_test == out).all(axis=1).mean()
    return batch_acc / steps


class Evaluator(Callback):
    def __init__(self):
        super().__init__()
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(steps=20) * 100
        self.accs.append(acc)
        print('')
        print('acc: %f%%' % acc)

    def get_config(self):  # 在有自定义网络层时，需要保存模型时，重写get_config函数
        config = {"kernel_size": self.kernel_size, "filter": self.filter}
        base_config = super(MyLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


evaluator = Evaluator()


h = model.fit(gen(128), steps_per_epoch=2, epochs=1,
              callbacks=[evaluator],
              validation_data=gen(128), validation_steps=1)

# model.save('Origain_model.h5')

# model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(1e-4))
# h2 = model.fit_generator(gen(128), steps_per_epoch=400, epochs=20,
#                          callbacks=[evaluator],
#                          validation_data=gen(128), validation_steps=20)
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.plot(h.history['loss'] + h2.history['loss'])
# plt.plot(h.history['val_loss'] + h2.history['val_loss'])
# plt.legend(['loss', 'val_loss'])
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.ylim(0, 1)
#
# plt.subplot(1, 2, 2)
# plt.plot(evaluator.accs)
# plt.ylabel('acc')
# plt.xlabel('epoch')

# y_pred = base_model.predict(X_vis)
# shape = y_pred[:, 2:, :].shape
# ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
# out = K.get_value(ctc_decode)[:, :4]
#
# plt.figure(figsize=(16, 8))
# for i in range(12):
#     plt.subplot(3, 4, i + 1)
#     plt.imshow(X_vis[i].transpose(1, 0, 2))
#     plt.title('pred:%s\nreal :%s' % (''.join([characters[x] for x in out[i]]),
#                                      ''.join([characters[x] for x in y_vis[i]])))
# (X_vis, y_vis, input_length_vis, label_length_vis), _ = next(gen(10000))
#
# y_pred = base_model.predict(X_vis, verbose=1)
# shape = y_pred[:, 2:, :].shape
# ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
# out = K.get_value(ctc_decode)[:, :4]
#
# (y_vis == out).all(axis=1).mean()
# from collections import Counter
#
# Counter(''.join([characters[i] for i in y_vis[y_vis != out]]))
# base_model.save('model.h5')
# characters2 = characters + ' '
#
# generator = ImageCaptcha(width=width, height=height)
# random_str = '0O0O'
# X_test = np.array(generator.generate_image(random_str))
# X_test = X_test.transpose(1, 0, 2)
# X_test = np.expand_dims(X_test, 0)
#
# y_pred = base_model.predict(X_test)
# shape = y_pred[:, 2:, :].shape
# ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
# out = K.get_value(ctc_decode)[:, :4]
# out = ''.join([characters[x] for x in out[0]])
#
# plt.imshow(X_test[0].transpose(1, 0, 2))
# plt.title('pred:' + str(out))
#
# argmax = np.argmax(y_pred, axis=2)[0]
# list(zip(argmax, ''.join([characters2[x] for x in argmax])))

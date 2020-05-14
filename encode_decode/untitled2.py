#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:54:49 2020

@author: lg
"""

from random import randint
from numpy import array, argmax, array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, LSTM, Dense

# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(1, n_unique-1) for _ in range(length)]


# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
    X1, X2, y = list(), list(), list()
    for _ in range(n_samples):
        # generate source sequence
        source = generate_sequence(n_in, cardinality)
        # define padded target sequence
        target = source[:n_out]
        target.reverse()
        # create padded input target sequence
        target_in = [0] + target[:-1]
        # encode
        src_encoded = to_categorical(source, num_classes=cardinality)
        tar_encoded = to_categorical(target, num_classes=cardinality)
        tar2_encoded = to_categorical(target_in, num_classes=cardinality)
        # store
        X1.append(src_encoded)
        X2.append(tar2_encoded)
        y.append(tar_encoded)
    return array(X1), array(X2), array(y)


# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
    # 定义训练编码器
    encoder_inputs = Input(shape=(None, n_input))  # n_input表示特征这一维(维的大小即特征的数目，如图像的feature map)
    encoder = LSTM(n_units, return_state=True)  # 编码器的特征维的大小dimension(n),即单元数。
    encoder_outputs, state_h, state_c = encoder(encoder_inputs) # 取出输入生成的隐藏状态和细胞状态，作为解码器的隐藏状态和细胞状态的初始化值。
    # 定义训练解码器
    decoder_inputs = Input(shape=(None, n_output))   # n_output：输出响应序列的特征维的大小。
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)  # 因解码器用编码器的隐藏状态和细胞状态，所以n_units必等
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])  # 这个解码层在后面推断中会被共享！！

    decoder_dense = Dense(n_output, activation='softmax')    # 这个full层在后面推断中会被共享！！
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)  # 得到以输入序列和目标序列作为输入，以目标序列的移位为输出的训练模型

    # 定义推断编码器  根据输入序列得到隐藏状态和细胞状态的路径图，得到模型，使用的输入到输出之间所有层的权重，与tf的预测签名一样
    encoder_model = Model(encoder_inputs, [state_h, state_c])   # 层编程模型很简单，只要用Model包住其输入和输出即可。
    #encoder_outputs, state_h, state_c = encoder(encoder_inputs)  # ？ 似乎就是上面的
    # 定义推断解码器，由于循环网络的性质，由输入状态(前)推理出输出状态(后)。
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))

    decoder_outputs, state_hd, state_cd = decoder_lstm(decoder_inputs, initial_state=[decoder_state_input_h, decoder_state_input_c])
    decoder_outputs = decoder_dense(decoder_outputs)
    # 由老状态更新出新的状态
    decoder_model = Model([decoder_inputs, decoder_state_input_h, decoder_state_input_c], [decoder_outputs, state_hd, state_cd])
    # return all models
    return model, encoder_model, decoder_model


# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # encode
    h_state, c_state = infenc.predict(source)  # 根据输入计算该原输入在状态空间的取值
    # start of sequence input
    target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)   # shape (1, 1, 51) [[[0,0,..]]] 一步
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char  这是递归网络的序列预测过程
        yhat,h_state, c_state = infdec.predict([target_seq, h_state, c_state])   # 获得循环地推的target_seq初始值，不停迭代产生新值
        # store prediction
        output.append(yhat[0, 0, :])
        # update state
        # update target sequence
        target_seq = yhat
    return array(output)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]

# configure problem
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
# define model
train, infenc, infdec = define_models(n_features, n_features, 128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])  # 这一层需要被编译
# generate training dataset
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 100000)
print(X1.shape,X2.shape,y.shape)
# train model
train.fit([X1, X2], y, epochs=1)
# evaluate LSTM
total, correct = 100, 0
# for _ in range(total):
#     X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
#     target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
#     if array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
#         correct += 1
# print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))
# spot check some examples
for _ in range(10):
    X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
    target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
    print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:21:28 2020

@author: lg
"""

from __future__ import print_function
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
 
BASE_DIR = "./"
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
 
TEXT_DATA_DIR = os.path.join(BASE_DIR, 'news20/20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000  # 每个文本或者句子的截断长度，只保留1000个单词
MAX_NUM_WORDS = 20000  # 用于构建词向量的词汇表数量
EMBEDDING_DIM = 100  # 词向量维度
VALIDATION_SPLIT = 0.2
 
"""
基本步骤：
1.数据准备：
预训练的词向量文件:下载地址：http://nlp.stanford.edu/data/glove.6B.zip
用于训练的新闻文本文件:下载地址:http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
2.数据预处理
1)生成文本文件词汇表:这里词汇表长度为20000，只取频数前20000的单词
2)将文本文件每行转为长度为1000的向量，多余的截断，不够的补0。向量中每个值表示单词在词汇表中的索引
3）将文本标签转换为one-hot编码格式
4）将文本文件划分为训练集和验证集
3.模型训练和保存
1）构建网络结构
2）模型训练
3）模型保存
"""
# 构建词向量索引
print("Indexing word vectors.")
embeddings_index = {}
with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]  # 单词
        coefs = np.asarray(values[1:], dtype='float32')  # 单词对应的向量
        embeddings_index[word] = coefs  # 单词及对应的向量
 
# print('Found %s word vectors.'%len(embeddings_index))#400000个单词和词向量
 
 
print('预处理文本数据集')
texts = []  # 训练文本样本的list
labels_index = {}  # 标签和数字id的映射
labels = []  # 标签list
 
# 遍历文件夹，每个子文件夹对应一个类别
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    # print(path)
    if os.path.isdir(path):
        labels_id = len(labels_index)
        labels_index[name] = labels_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                args = {} if sys.version_info < (3,) else {'encoding': 'latin-1'}
                with open(fpath, **args) as f:
                    t = f.read()
                    i = t.find('\n\n')  ##屏蔽文件头获取'\n\n'的位置
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                labels.append(labels_id)
 
print("Found %s texts %s label_id." % (len(texts), len(labels)))  # 19997个文本文件
 
# 向量化文本样本
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
# fit_on_text(texts) 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档。就是对文本单词进行去重后
tokenizer.fit_on_texts(texts)
# texts_to_sequences(texts) 将多个文档转换为word在词典中索引的向量形式,shape为[len(texts)，len(text)] -- (文档数，每条文档的长度)
sequences = tokenizer.texts_to_sequences(texts)
print(sequences[0])
print(len(sequences))  # 19997
 
word_index = tokenizer.word_index  # word_index 一个dict，保存所有word对应的编号id，从1开始
print("Founnd %s unique tokens." % len(word_index))  # 174074个单词
# ['the', 'to', 'of', 'a', 'and', 'in', 'i', 'is', 'that', "'ax"] [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(list(word_index.keys())[0:10], list(word_index.values())[0:10])  #
 
#空位补零
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  # 长度超过MAX_SEQUENCE_LENGTH则截断，不足则补0
 
labels = to_categorical(np.asarray(labels))
print("训练数据大小为：", data.shape)  # (19997, 1000)
print("标签大小为:", labels.shape)  # (19997, 20)
 
# 将训练数据划分为训练集和验证集
indices = np.arange(data.shape[0])
np.random.shuffle(indices)  # 打乱数据
data = data[indices]
labels = labels[indices]
 
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
 
# 训练数据
x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
 
# 验证数据
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
 
# 准备词向量矩阵

#注意，这里的　len(word_index) + 1) 多了一个１　,第一个位置的０留给不认识的字0 ,这个０　
#大部分来自　pad_sequences
num_words = min(MAX_NUM_WORDS, len(word_index) + 1)  # 词汇表数量
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))  # 20000*100
 



for word, i in word_index.items():
    if i >= MAX_NUM_WORDS:  # 过滤掉根据频数排序后排20000以后的词
        continue
    embedding_vector = embeddings_index.get(word)  # 根据词向量字典获取该单词对应的词向量
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
 
# 加载预训练的词向量到Embedding layer
embedding_layer = Embedding(input_dim=num_words,  # 词汇表单词数量
                            output_dim=EMBEDDING_DIM,  # 词向量维度
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,  # 文本或者句子截断长度
                            trainable=False)  # 词向量矩阵不进行训练
 
print("开始训练模型.....")
# 使用
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')  # 返回一个张量，长度为1000，也就是模型的输入为batch_size*1000

#这种用法比较经典，embed一个矩阵，按照矩阵的行进行索引vect
embedded_sequences = embedding_layer(sequence_input)  # 返回batch_size*1000*100
x = Conv1D(128, 5, activation='relu')(embedded_sequences)  # 输出的神经元个数为128，卷积的窗口大小为5
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)
 
model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
 
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))
model.summary()
model.save("../data/textClassifier.h5")
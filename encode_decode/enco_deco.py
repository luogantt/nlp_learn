
# 学习encode decode 方法

import numpy as np
from keras.models import Model
from keras.models import load_model
from keras.layers import Input,LSTM,Dense


#设置一些参数
batch_size = 64  # Batch size for training.
epochs = 10 # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'fra.txt'

input_texts = []
target_texts = []
input_characters = set()
target_characters = set()

#对数据进行梳理

#打开文件逐条读取数据
lines = open(data_path,encoding='utf-8').read().split('\n')
for index,line in enumerate(lines[: min(num_samples, len(lines) - 1)]):
    #输入       #目标
    input_text, target_text = line.split('\t')[:2]  #keras 官网代码有问题，这里我改了

    target_text = '\t' + target_text + '\n'

    #将输入的英文和中文分别放在不同的列表
    input_texts.append(input_text)
    target_texts.append(target_text)

    # 分别建立字符集合
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

#输入的字符集合
input_characters = sorted(list(input_characters))
#输出的字符集合
target_characters = sorted(list(target_characters))
# 统计source和target的字符数
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
# 取出最长的句子的长度：
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
# 打印具体的信息
print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
# 将它们转化为id的形式存储（char-to-id）
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])
# 初始化
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
print(encoder_input_data.shape)
# 训练测试
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data比decoder_input_data提前一个时间步长
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.





#########################################################################################################################################################
# 定义训练编码器
#None表示可以处理任意长度的序列
# num_encoder_tokens表示特征的数目,三维张量的列数

encoder_inputs = Input(shape=(None, num_encoder_tokens),name='encoder_inputs')  

# 编码器，要求其返回状态，lstm 公式理解https://blog.csdn.net/qq_38210185/article/details/79376053  
encoder = LSTM(latent_dim, return_state=True,name='encoder_LSTM')               # 编码器的特征维的大小latent_dim,即单元数,也可以理解为lstm的层数

#lstm 的输出状态，隐藏状态，候选状态
encoder_outputs, state_h, state_c = encoder(encoder_inputs) # 取出输入生成的隐藏状态和细胞状态，作为解码器的隐藏状态和细胞状态的初始化值。

#上面两行那种写法很奇怪，看了几天没看懂，可以直接这样写
#encoder_outputs, state_h, state_c= LSTM(latent_dim, return_state=True,name='encoder_LSTM')(encoder_inputs)

# 我们丢弃' encoder_output '，只保留隐藏状态，候选状态
encoder_states = [state_h, state_c]  
#########################################################################################################################################################





#########################################################################################################################################################
# 定义解码器的输入
# 同样的，None表示可以处理任意长度的序列
# 设置解码器，使用' encoder_states '作为初始状态
# num_decoder_tokens表示解码层嵌入长度,三维张量的列数
decoder_inputs = Input(shape=(None, num_decoder_tokens),name='decoder_inputs')  

# 接下来建立解码器，解码器将返回整个输出序列
# 并且返回其中间状态，中间状态在训练阶段不会用到，但是在推理阶段将是有用的
# 因解码器用编码器的隐藏状态和细胞状态，所以latent_dim必等
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,name='decoder_LSTM')   
# 将编码器输出的状态作为初始解码器的初始状态
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)

# 添加全连接层
# 这个full层在后面推断中会被共享！！
decoder_dense = Dense(num_decoder_tokens, activation='softmax',name='softmax')  
decoder_outputs = decoder_dense(decoder_outputs)
#########################################################################################################################################################










# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
####################################################################################################################
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)    #原始模型  
####################################################################################################################          
#model.load_weights('s2s.h5')
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# 保存模型
model.save('s2s.h5')


# 定义推断编码器  根据输入序列得到隐藏状态和细胞状态的路径图，得到模型，使用的输入到输出之间所有层的权重，与tf的预测签名一样
####################################################################################################################
encoder_model = Model(encoder_inputs, encoder_states)                #编码模型 ，注意输出是  encoder_states = [state_h, state_c]  
####################################################################################################################  
     

encoder_model.save('encoder_model.h5')



#定义解码模型

#解码的隐藏层
decoder_state_input_h = Input(shape=(latent_dim,))
#解码的候选门
decoder_state_input_c = Input(shape=(latent_dim,))
#解码的输入状态
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]


decoder_outputs, state_hd, state_cd = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_hd, state_cd]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
# 反向查找令牌索引，将序列解码回可读的内容。

decoder_model.save('decoder_model.h5')
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # 将输入编码为状态向量
    #注意输出是  encoder_states = [state_h, state_c]  
    # states_value  是一个有两元素的列表，每个元素的维度是256
    states_value = encoder_model.predict(input_seq)             
    # 生成长度为1的空目标序列
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # 用起始字符填充目标序列的第一个字符。
    target_seq[0, 0, target_token_index['\t']] = 1.
    # 对一批序列的抽样循环(为了简化，这里我们假设批大小为1)
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        # 退出条件:到达最大长度或找到停止字符。
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        # 更新目标序列(长度1)
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        
        print(sampled_token_index)
        # 更新状态
        states_value = [h, c]
    return decoded_sentence

for seq_index in range(200):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index : seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)

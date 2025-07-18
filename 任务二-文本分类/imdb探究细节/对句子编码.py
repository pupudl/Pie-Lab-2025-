import tensorflow as tf

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',  #文本矢量层会去掉标点符号，dog!同dog
    'Do you think my dog is amazing?'
]

vectorize_layer = tf.keras.layers.TextVectorization()  #文本矢量化层
vectorize_layer.adapt(sentences)  #根据句子中的单词自适应的生成词汇表

vocabulary = vectorize_layer.get_vocabulary(include_special_tokens=False)  #include_special_tokens=False，默认为true，会多出两个token：'', '[UNK]'
print(vocabulary)
#print token index
for index,word in enumerate(vocabulary):
    print( index,word)

vocabulary = vectorize_layer.get_vocabulary()

# sequence = vectorize_layer(sentences)  #将句子编码成整数列表，用token代替单词，eg.I love my dog->6 3 2 4,长度为最长句子的长度（填充0）
# print(sequence)
# for index,word in enumerate(vocabulary):
#     print( index,word)

sentences_dataset = tf.data.Dataset.from_tensor_slices(sentences)
sequences = sentences_dataset.map(vectorize_layer)  #转换为一个map数据集，并可对其进行迭代,每个序列长度对应于原始句子长度
# print(sequences)
# for sentence,sequence in zip(sentences,sequences):
#     print(f'{sentence} ---> {sequence}')
#
# maxlen参数决定最长序列长度，truncating='post'则从后面截断
# sequences_pre = tf.keras.utils.pad_sequences(sequences,padding='pre')  #pre预填充，将0填充到最前面，post填充到最后
# print(sequences_pre)

# vectorize_layer = tf.keras.layers.TextVectorization(ragged=True)  #文本矢量化层,输出参差不齐的张量，即可能包含不同形状的张量，也就是不填充
# vectorize_layer.adapt(sentences)  #根据句子中的单词自适应的生成词汇表
# vocabulary = vectorize_layer.get_vocabulary()
# ragged_sequences = vectorize_layer(sentences)
# for index,word in enumerate(vocabulary):
#     print(index,word)
# print(ragged_sequences)

# test_data = [
#     'i really love my dog',
#     'my dog loves my manatee'
# ]
# test_seq = vectorize_layer(test_data)
# print(test_seq)   #因为loves、really与manatee都不在词汇表中，所以被标记为[UNK]即1


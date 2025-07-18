import json

TRAINING_SIZE = 20000
VOCAB_SIZE= 10000
MAX_LENGTH = 32
EMBEDDING_DIM = 16

with open("C:/Users/78636/Desktop/output.json",'r') as f:
    datastore = json.load(f)

# print(datastore[0])
# print(datastore[20000])
sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# 划分训练集和测试集
training_sentences = sentences[0:TRAINING_SIZE]  # 训练文本（前training_size个样本）
testing_sentences = sentences[TRAINING_SIZE:]    # 测试文本（剩余样本）
training_labels = labels[0:TRAINING_SIZE]        # 训练标签
testing_labels = labels[TRAINING_SIZE:]          # 测试标签

import tensorflow as tf

vectorize_layer = tf.keras.layers.TextVectorization(       #文本矢量化层
                        max_tokens=VOCAB_SIZE,             #词汇表大小（如10000）
                        output_sequence_length=MAX_LENGTH  #统一序列长度（如120）
)

# vectorize_layer.adapt(sentences)  #根据句子中的单词自适应的生成词汇表
# vocabulary = vectorize_layer.get_vocabulary()
# post_padded_sequences = vectorize_layer(sentences)
# print(f'padded sequence: {post_padded_sequences[2]}')
# print(f'shape of padded sequences: {post_padded_sequences.shape}')

#适配词汇表到训练文本
vectorize_layer.adapt(training_sentences)
# 文本向量化转换
train_sequences = vectorize_layer(training_sentences)  # 训练集向量化
test_sequences = vectorize_layer(testing_sentences)    # 测试集向量化

# 构建TensorFlow Dataset对象
train_dataset_vectorized = tf.data.Dataset.from_tensor_slices(
    (train_sequences, training_labels))  # (向量化文本, 标签)
test_dataset_vectorized = tf.data.Dataset.from_tensor_slices(
    (test_sequences, testing_labels))

#定义常量参数
SHUFFLE_BUFFER_SIZE = 1000               #打乱数据时使用的缓冲区大小
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE  #自动调整预取缓冲区大小
BATCH_SIZE = 32                          #每个批次的样本数量

#构建最终训练数据集（含数据增强）
train_dataset_final = (
    train_dataset_vectorized
    .cache()                        #缓存数据到内存/磁盘避免重复计算
    .shuffle(SHUFFLE_BUFFER_SIZE)   #随机打乱数据（缓冲区大小1000）
    .prefetch(PREFETCH_BUFFER_SIZE) #异步预取数据提升吞吐量
    .batch(BATCH_SIZE)              #按32个样本/批次的规格组织数据
)

#构建最终测试数据集（不含打乱）
test_dataset_final = (
    test_dataset_vectorized
    .cache()                        #同样需要缓存
    .prefetch(PREFETCH_BUFFER_SIZE) #预取
    .batch(BATCH_SIZE)              #批次化
)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LENGTH,)),
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
    tf.keras.layers.GlobalAveragePooling1D(),  #tf.keras.layers.Flatten()
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 30
history = model.fit(train_dataset_final,epochs=num_epochs,validation_data=test_dataset_final,verbose=2)

import matplotlib.pyplot as plt
# 定义训练曲线绘制函数
def plot_graphs(history, metric):
    # 绘制训练集指标曲线
    plt.plot(history.history[metric])
    # 绘制验证集指标曲线（自动添加'val_'前缀）
    plt.plot(history.history['val_' + metric])
    # 设置坐标轴标签
    plt.xlabel("Epochs")  # x轴显示训练轮次
    plt.ylabel(metric)    # y轴显示指标名称
    # 添加图例（自动区分训练/验证曲线）
    plt.legend([metric, 'val_' + metric])
    # 显示图形
    plt.show()

# 使用示例：绘制准确率和损失曲线
plot_graphs(history, "accuracy")  # 绘制准确率变化曲线
plot_graphs(history, "loss")       # 绘制损失变化曲线
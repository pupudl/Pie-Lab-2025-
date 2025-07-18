import tensorflow_datasets as tfds
import tensorflow as tf

imdb,info = tfds.load("imdb_reviews",with_info=True,as_supervised=True)  #加载指定名称的数据集"imdb_reviews"
#with_info=True:返回数据集的元信息（如数据集描述、特征结构、样本数量等），存储在info变量中,as_supervised=True:以监督学习的格式（即(input,label)对）加载数据
#对于IMDB数据集，每条数据是(评论文本,情感标签)
#imdb：包含训练集和测试集的字典（键为'train'和'test'）
#info：数据集描述对象（可通过info.features查看特征详情）

#single_example = list(imdb['train'].take(1))[0]
#imdb['train']：获取训练集（共25,000条带标签的评论）,.take(1)：从数据集中取出第一条样本（返回一个tf.data.Dataset对象）
#list(...)[0]：将取出的样本转换为Python列表并索引第一个元素（即(文本,标签)元组）。
#输出结构：single_example是一个元组，格式为(文本张量,标签张量)
#print(single_example[1])

train_data,test_data = imdb['train'],imdb['test']

train_reviews = train_data.map(lambda review,label  :  review) #从包含(review, label)对的原始数据集中提取出所有评论文本（review），形成一个新的仅包含文本的数据集
train_labels = train_data.map(lambda review,label  :  label)

test_reviews = test_data.map(lambda review,label:review)
test_labels = test_data.map(lambda review,label:label)

vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=10000)  #文本矢量化层（将原始文本转换为整数序列）,max_tokens=10000仅按出现次数取前10000个token(10000种单词)
vectorize_layer.adapt(train_reviews)  #根据句子中的单词自适应的生成词汇表
vocabulary = vectorize_layer.get_vocabulary()


#定义填充函数
def padding_func(sequences):
    # 将不规则张量转换为密集张量
    sequences = sequences.ragged_batch(batch_size=sequences.cardinality()) #将所有样本合并为一个批次，batch_size=sequences.cardinality()表示将所有样本合并为一个批次（cardinality()返回样本总数）
    #创建一个包含所有序列的参差不齐的批次，其产生一个不规则的张量，其序列长度不同

    sequences = sequences.get_single_element() #从批处理后的不规则张量中提取单个密集张量（tf.Tensor），此时形状为[num_samples, None]（None表示变长维度）

    # 使用Keras工具进行填充 (pre表示前端填充，post表示后端截断)
    padded_sequences = tf.keras.utils.pad_sequences(
        sequences.numpy(),  #将TensorFlow张量转换为NumPy数组，供pad_sequences处理
        maxlen=120,    #一个句子最长120个单词
        truncating='post',
        padding='pre'
    )
    padded_sequences = tf.data.Dataset.from_tensor_slices(padded_sequences) #将填充后的NumPy数组重新转换为tf.data.Dataset 对象，以便后续使用TensorFlow的数据管道（map前操作）
    # 转换回Dataset对象
    return padded_sequences

#处理训练和测试数据
#将train_reviews原始文本转换为整数序列,eg.very good->[1,2]，输出是一个不规则张量数据集(用了map)，每个样本的长度可能不同，调用padding_func对变长序列进行填充/截断/统一长度
train_sequences = train_reviews.map(lambda text: vectorize_layer(text)).apply(padding_func)
test_sequences = test_reviews.map(lambda text: vectorize_layer(text)).apply(padding_func)


#组合数据集（将特征序列与标签配对）
train_dataset_vectorized = tf.data.Dataset.zip((train_sequences, train_labels))
test_dataset_vectorized = tf.data.Dataset.zip((test_sequences, test_labels))

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

vocab_size, embedding_dim = 10000,16  #句子的长度与嵌入的维度

model = tf.keras.Sequential([
    tf.keras.Input(shape=(120,)),
    #将离散的文本数据转化为可学习的连续表示
    tf.keras.layers.Embedding(vocab_size, embedding_dim),  #嵌入层初始化一个形状为(vocab_size, embedding_dim)的矩阵,输入中的每个整数会被替换为一个长度为embedding_dim的向量,最终得到(batch_size,120,16)
    tf.keras.layers.Flatten(),  #tf.keras.layers.GlobalAveragePooling1D()：[mean(t1_1,t2_1,...,t120_1),mean(t1_2,...,t120_2),...,mean(t1_16,...,t120_16)]，（batch_size,embedding_dim）
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  #二分类只需1个输出
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])   #编译模型
model.summary()

num_epochs = 10
model.fit(train_dataset_final,epochs=num_epochs,validation_data=test_dataset_final)  #训练模型

#获取模型的第一层（嵌入层）
embedding_layer = model.layers[0]
#提取该层的权重矩阵（第一个可训练参数）
embedding_weights = embedding_layer.get_weights()[0]
#打印权重矩阵的形状
print(embedding_weights.shape)  # 输出形如：(vocab_size, embedding_dim)

import io
#打开两个TSV文件用于写入词向量和元数据
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')  # 存储词向量数值
out_m = io.open('meta.tsv', 'w', encoding='utf-8')  # 存储词汇文本
#从文本向量化层获取词汇表
vocabulary = vectorize_layer.get_vocabulary()
#遍历词汇表（从索引1开始跳过占位符）
for word_num in range(1, len(vocabulary)):
    word_name = vocabulary[word_num]  # 获取当前词
    word_embedding = embedding_weights[word_num]  # 获取对应词向量
    # 写入数据（每行一个词/向量）
    out_m.write(word_name + "\n")  # 元数据文件写入词汇文本
    out_v.write('\t'.join([str(x) for x in word_embedding]) + "\n")  # 向量文件写入数值
# 关闭文件句柄
out_v.close()
out_m.close()
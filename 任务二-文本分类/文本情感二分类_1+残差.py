import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import time
import matplotlib.pyplot as plt
import numpy as np


# from torchtext.models.roberta.bundler import xlmr_transform


# 数据预处理
class IMDBDataset(Dataset):
    def __init__(self, data_dir, vocab=None, max_len=120):
        self.texts = []  # 存储评论文本
        self.labels = []  # 存储对应标签(0=负面,1=正面)

        # 加载负面评价
        neg_dir = os.path.join(data_dir, 'neg')
        for fname in os.listdir(neg_dir):  # 遍历neg_dir目录下的文件
            with open(os.path.join(neg_dir, fname), 'r', encoding='utf-8') as f:
                self.texts.append(f.read())  # 将文本添加到texts后
                self.labels.append(0)  # 将对应标签(0)添加到labels后

        # 加载正面评价
        pos_dir = os.path.join(data_dir, 'pos')
        for fname in os.listdir(pos_dir):
            with open(os.path.join(pos_dir, fname), 'r', encoding='utf-8') as f:
                self.texts.append(f.read())
                self.labels.append(1)

        # 构建词汇表
        self.vocab = vocab if vocab else self._build_vocab()  # 词汇表为None时(训练集,因为此时未建表)建立词汇表;词汇表不为None时(验证集或测试集)选择使用训练集建立好的词汇表
        # 若测试集用自己的数据建立词汇表,可能与训练集词汇表对应单词的id不匹配,导致效果不好
        self.vocab_size = len(self.vocab)  # 保存词汇表长度

    def _build_vocab(self, max_size=10000):  # 取前10,000个词频最高的单词构建词汇表
        counter = Counter()  # 统计词频
        for text in self.texts:
            counter.update(text.lower().split())
        vocab = {word: i + 2 for i, (word, _) in enumerate(counter.most_common(max_size))}  # 只取单词,忽略计数,从2开始编号
        vocab['<pad>'] = 0  # 用于填充短文本
        vocab['<unk>'] = 1  # 用于标记不在词汇表中的单词
        return vocab

    def text_to_sequence(self, text):
        return [
            self.vocab.get(word.lower(), self.vocab['<unk>'])  # 将文本转为序号
            for word in text.split()[:120]  # 只取前120个单词,来截断长文本,评价大多110~130单词
        ]

    def __len__(self):
        return len(self.texts)  # 返回样本总数

    def __getitem__(self, idx):
        sequence = self.text_to_sequence(self.texts[idx])  # 把第idx条文本转换为序列
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(self.labels[idx],
                                                                      dtype=torch.float)  # 将评论文本,标签文本转换成张量


# LSTM基本单元
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size  # 输入向量的维度(词嵌入的维度)
        self.hidden_size = hidden_size  # 隐藏状态的维度

        # 输入门
        self.input_gate = nn.Sequential(
            # 将当前输入x和上一时刻的隐藏状态a拼接后,转换为一个维度与隐藏层相同的门控信号
            nn.Linear(input_size + hidden_size, hidden_size),  # 创建权重矩阵W、偏置向量b、自动初始化参数
            # w为权重矩阵,形状为 (hidden_size, input_size + hidden_size)
            # b为偏置向量,形状为 (hidden_size,)
            # 由公式可知,拼接后的向量,形状 (batch_size, input_size + hidden_size)
            # 结果输出形状为 (batch_size, hidden_size)
            nn.Sigmoid()  # 将门控信号压缩到0~1(0=关闭,1=打开)
        )
        # 遗忘门
        self.forget_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        # 输出门
        self.output_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )

        self.c_hat = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()  # 输出范围[-1,1]
        )

    def forward(self, x_t, hc_t_1):
        h_t_1, c_t_1 = hc_t_1
        # 拼接输入和上一隐藏状态
        xh = torch.cat((x_t, h_t_1), dim=1)  # [batch_size, input_size + hidden_size]

        i_t = self.input_gate(xh)  # 进行线性运算W[x_t,h_t_1]+b
        f_t = self.forget_gate(xh)
        o_t = self.output_gate(xh)

        # 生成候选记忆
        c_hat = self.c_hat(xh)

        # 更新细胞状态
        c_t = f_t * c_t_1 + i_t * c_hat  # 旧记忆遗忘部分+新记忆添加部分

        # 生成新隐藏状态
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()

        # 输入：x (batch_size,seq_len,input_size)  eg.32个句子,每个句子50个单词,每个单词用128维向量表示
        # 输出：h (batch_size,seq_len,hidden_size)
        self.cell = LSTMCell(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x, hc_0=None):  # 默认h0与c0为None
        batch_size, seq_len, input_size = x.size()  # 获取输入形状
        device = x.device

        # 初始化隐藏状态和细胞状态
        if hc_0 is None:  # 零初始化,用于序列的第一个时间步
            h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            c_0 = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_0, c_0 = hc_0

        h_list = []  # 存储每个时间步的隐藏状态
        h_t_1, c_t_1 = h_0, c_0  # 初始化

        for t in range(seq_len):
            x_t = x[:, t, :]  # 获取当前时间步的输入x[:, t, :]->[batch_size,t, input_size]
            h_t, c_t = self.cell(x_t, (h_t_1, c_t_1))  # 调用LSTMCell->h_t, c_t为[batch_size,hidden_size]
            h_list.append(h_t)  # 保存当前隐藏状态
            h_t_1, c_t_1 = h_t, c_t  # 更新状态

        # 将seq_len个[batch_size, hidden_size]的张量拼接为3D张量
        h = torch.stack(h_list, dim=1)  # 将列表转为张量[batch_size, seq_len, hidden_size]
        return h, (h_t_1, c_t_1)  # 返回完整序列和最终状态


# 情感分类模型
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)  # 将单词ID转换为密集向量
        self.lstm = LSTM(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(0.7)  # 训练时随机将70%的神经元输出置零,防止模型过度依赖某些特征
        self.fc = nn.Linear(hidden_dim, 1)  # 将LSTM最终输出转换为单个预测值

    def forward(self, x):
        # x形状:[batch_size, seq_len],batch_size个长度为seq_len的句子
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim]
        last_out = self.dropout(lstm_out[:, -1, :])  # 取最后时刻输出->[batch_size, hidden_dim]
        return torch.sigmoid(self.fc(last_out)).squeeze()
        # fc将向量压缩为1维 [batch_size, 1],sigmoid将输出压缩到(0,1)之间,表示正面情感的概率,squeeze去掉多余的维度(从 [batch_size, 1] 变为 [batch_size])


class LSTMCellWithResidual(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCellWithResidual, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.forget_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.output_gate = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.c_hat = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )

        # 残差连接需要的线性变换
        if input_size != hidden_size:  # x_t与h_t维度不相同,x_t经过F(x_t)线性变换
            self.residual_transform = nn.Linear(input_size, hidden_size)
        else:  # x_t与h_t维度相同,x_t不用经过F(x_t)线性变换
            self.residual_transform = None

    def forward(self, x_t, hc_t_1):
        h_t_1, c_t_1 = hc_t_1
        xh = torch.cat((x_t, h_t_1), dim=1)

        i_t = self.input_gate(xh)
        f_t = self.forget_gate(xh)
        o_t = self.output_gate(xh)
        c_hat = self.c_hat(xh)

        c_t = f_t * c_t_1 + i_t * c_hat

        # 残差连接
        if self.residual_transform is not None:  # 维度不相同,线性变换
            x_t_res = self.residual_transform(x_t)
        else:
            x_t_res = x_t  # 维度相同,无需变换

        h_t = o_t * torch.tanh(c_t) + x_t_res  # 加入残差连接

        return h_t, c_t


class LSTMWithResidual(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMWithResidual, self).__init__()
        self.cell = LSTMCellWithResidual(input_size, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

    def forward(self, x, hc_0=None):
        batch_size, seq_len, input_size = x.size()
        device = x.device

        if hc_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            c_0 = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            h_0, c_0 = hc_0

        h_list = []
        h_t_1, c_t_1 = h_0, c_0

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t, c_t = self.cell(x_t, (h_t_1, c_t_1))
            h_list.append(h_t)
            h_t_1, c_t_1 = h_t, c_t

        h = torch.stack(h_list, dim=1)
        return h, (h_t_1, c_t_1)


class SentimentClassifierWithResidual(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = LSTMWithResidual(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        last_out = self.dropout(lstm_out[:, -1, :])
        return torch.sigmoid(self.fc(last_out)).squeeze()


# 训练和评估函数
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()  # 设置为训练模式
    total_loss = 0
    correct = 0
    total = 0

    for texts, labels in loader:  # 遍历数据加载器
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()  # 清空梯度缓存,防止梯度累加
        outputs = model(texts)  # 前向传播 [batch_size, 1]
        loss = criterion(outputs, labels)  # 计算损失

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        total_loss += loss.item()  # 累加损失
        preds = (outputs > 0.5).float()  # 将Sigmoid输出转为二分类预测(阈值0.5)
        correct += (preds == labels).sum().item()  # 正确数
        total += labels.size(0)  # 总样本数

    return total_loss / len(loader), correct / total  # 平均损失和准确率


def evaluate(model, loader, criterion, device):
    model.eval()  # 设置为评估模式
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算,减少内存占用(不保存中间梯度)
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


# 可视化训练过程
def plot_training(lstm_train_losses, lstm_val_losses, lstm_train_accs, lstm_val_accs,
                  res_train_losses, res_val_losses, res_train_accs, res_val_accs, num_epochs):
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(15, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, lstm_train_losses, 'b-', label='LSTM Train')
    plt.plot(epochs, lstm_val_losses, 'b--', label='LSTM Val')
    plt.plot(epochs, res_train_losses, 'r-', label='ResLSTM Train')
    plt.plot(epochs, res_val_losses, 'r--', label='ResLSTM Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, lstm_train_accs, 'b-', label='LSTM Train')
    plt.plot(epochs, lstm_val_accs, 'b--', label='LSTM Val')
    plt.plot(epochs, res_train_accs, 'r-', label='ResLSTM Train')
    plt.plot(epochs, res_val_accs, 'r--', label='ResLSTM Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_comparisontest.png')
    plt.show()


# 主函数
def main():
    # 设置路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "DataSet", "aclImdb")

    # 加载训练集并构建词汇表
    train_full = IMDBDataset(os.path.join(data_dir, "train"))
    vocab = train_full.vocab

    train_size = int(0.8 * len(train_full))  # 80%训练
    val_size = len(train_full) - train_size  # 20%验证
    train_dataset, val_dataset = random_split(train_full, [train_size, val_size])

    # 加载测试集（强制使用训练集的词汇表）
    test_dataset = IMDBDataset(os.path.join(data_dir, "test"), vocab=vocab)

    # 创建DataLoader
    def collate_fn(batch):  # 处理变长序列（填充到相同长度）
        texts, labels = zip(*batch)
        texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
        return texts_padded, torch.stack(labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # LSTM
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SentimentClassifier(vocab_size=train_full.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # 优化器：Adam（学习率0.001）
    criterion = nn.BCELoss()  # 损失函数：BCELoss（二分类交叉熵）

    # 训练循环
    num_epochs = 20
    best_val_acc = 0

    # 初始化记录列表
    lstm_train_losses, lstm_val_losses = [], []
    lstm_train_accs, lstm_val_accs = [], []

    for epoch in range(num_epochs):
        start_time = time.time()
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        lstm_train_losses.append(train_loss)  # 累积训练损失
        lstm_train_accs.append(train_acc)  # 累积训练准确率

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        lstm_val_losses.append(val_loss)  # 累积验证损失
        lstm_val_accs.append(val_acc)  # 累积验证准确率

        epoch_time = time.time() - start_time

        # 保存最佳模型(基于验证集性能)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_1test.pt")
            print(f"New best model saved with val_acc: {val_acc:.4f}")

        print(f"Epoch {epoch + 1}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")
        print("-" * 60)
    # 测试集
    print("\nEvaluating BEST model on TEST set...")
    best_model = SentimentClassifier(vocab_size=train_full.vocab_size).to(device)
    best_model.load_state_dict(torch.load("best_model_1test.pt", weights_only=True))  # 加载最佳参数
    lstm_test_loss, lstm_test_acc = evaluate(best_model, test_loader, criterion, device)
    print(f"Final Test Performance: Loss={lstm_test_loss:.4f}, Acc={lstm_test_acc * 100:.2f}%")

    # LSTMWithResidual
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SentimentClassifierWithResidual(vocab_size=train_full.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()

    # 训练循环
    num_epochs = 20
    best_val_acc = 0

    resnet_train_losses, resnet_val_losses = [], []
    resnet_train_accs, resnet_val_accs = [], []

    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        resnet_train_losses.append(train_loss)
        resnet_train_accs.append(train_acc)

        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        resnet_val_losses.append(val_loss)
        resnet_val_accs.append(val_acc)

        epoch_time = time.time() - start_time

        # 保存最佳模型(基于验证集性能)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model_WithResidual_1test.pt")
            print(f"New best model with residual saved with val_acc: {val_acc:.4f}")

        print(f"Epoch {epoch + 1}/{num_epochs} | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")
        print("-" * 60)

    # 测试集
    print("\nEvaluating BEST model with residual on TEST set...")
    best_model = SentimentClassifierWithResidual(vocab_size=train_full.vocab_size).to(device)
    best_model.load_state_dict(torch.load("best_model_WithResidual_1test.pt", weights_only=True))  # 加载最佳参数
    resnet_test_loss, resnet_test_acc = evaluate(best_model, test_loader, criterion, device)
    print(f"Final Test with residual Performance: Loss={resnet_test_loss:.4f}, Acc={resnet_test_acc * 100:.2f}%")

    plot_training(
        lstm_train_losses,  # 训练损失列表（长度=num_epochs）
        lstm_val_losses,  # 验证损失列表
        lstm_train_accs,  # 训练准确率列表
        lstm_val_accs,  # 验证准确率列表
        resnet_train_losses,
        resnet_val_losses,
        resnet_train_accs,
        resnet_val_accs,
        num_epochs
    )

    # 打印最终结果
    print("\n===== 对比结果 =====")
    print(f'LSTM最终准确率: {lstm_test_acc:.2f}%')
    print(f'残差LSTM最终准确率: {resnet_test_acc:.2f}%')
    print(f'准确率提升: {resnet_test_acc - lstm_test_acc:.2f}%')

    # 分析收敛速度
    lstm_best_epoch = np.argmax(lstm_val_accs) + 1
    resnet_best_epoch = np.argmax(resnet_val_accs) + 1
    print(f'CNN最佳准确率出现在第 {lstm_best_epoch} 个epoch')
    print(f'残差LSTM最佳准确率出现在第 {resnet_best_epoch} 个epoch')


if __name__ == "__main__":
    main()
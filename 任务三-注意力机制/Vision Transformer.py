import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import freeze_support


#手动实现Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k   #缩放点积结果，防止梯度消失

    def forward(self, Q, K, V):

        #输入Q:[batch_size, num_heads, seq_len, d_k]
        #输入K:[batch_size, num_heads, seq_len, d_k]
        #K.transpose(-2, -1):转置最后两个维度->[batch_size, num_heads, d_k, seq_len]
        #输出scores:[batch_size, num_heads, seq_len, seq_len],为相关系数矩阵
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)  #计算（Q*K）/sqrt（d_k）

        #对每行的分数进行归一化
        attn = F.softmax(scores, dim=-1)

        #用注意力权重对值向量V进行加权求和，得到每个位置的新表示
        output = torch.matmul(attn, V)

        #输出output:[batch_size, num_heads, seq_len, d_k]
        return output


#实现Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim   #输入/输出的特征维度（每个token的向量长度）
        self.num_heads = num_heads   #注意力头的数量（将 embed_dim 分割为多个头）
        self.d_k = embed_dim // num_heads  #将256维向量拆分为8个32维的子空间，独立计算注意力

        self.W_q = nn.Linear(embed_dim, embed_dim)  #查询（Query）变换，x*W_q
        self.W_k = nn.Linear(embed_dim, embed_dim)  #键（Key）变换，x*W_k
        self.W_v = nn.Linear(embed_dim, embed_dim)  #值（Value）变换，x*W_v，维度不变，但是内容改变了
        self.W_o = nn.Linear(embed_dim, embed_dim)  #输出投影
        self.attention = ScaledDotProductAttention(self.d_k)  #Scaled Dot-Product Attention模块

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        #求得KQV，再多头分割与维度变换
        #[batch_size, seq_len, embed_dim]中embed_dim拆分为num_heads个d_k维的子空间->[batch_size, seq_len, num_heads, d_k]
        #交换seq_len和num_heads维度,将num_heads维度前置，便于并行计算每个头的注意力->[batch_size, seq_len, num_heads, d_k]
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 执行Scaled Dot-Product Attention，按照公式进行计算
        attn_output = self.attention(Q, K, V)

        #将8个头的32维输出拼接回256维
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        #将拼接后的结果，通过线性投影进行融合
        output = self.W_o(attn_output)
        return output


#实现Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)   #多头自注意力机制，8个头

        #先升维（增加模型容量）后降维（保持输入输出维度一致）
        self.linear1 = nn.Linear(embed_dim, ff_dim)  #MLP block，先升维
        self.linear2 = nn.Linear(ff_dim, embed_dim)  #MLP block，再降维

        #对每个token的特征向量进行归一化
        self.norm1 = nn.LayerNorm(embed_dim)  #自注意力后的层归一化
        self.norm2 = nn.LayerNorm(embed_dim)  #MLP block后的层归一化
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        #先经过多头自注意力机制
        attn_output = self.self_attn(x)

        #残差连接，保留原始输入信息，缓解梯度消失
        x = x + self.dropout(attn_output)

        #层归一化
        x = self.norm1(x)

        #MLP block
        ff_output = self.linear2(F.gelu(self.linear1(x)))

        # 残差连接，保留原始输入信息，缓解梯度消失
        x = x + self.dropout(ff_output)

        # 层归一化
        x = self.norm2(x)
        return x


#简化版ViT
class SimpleViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10,
                 embed_dim=256, num_heads=8, ff_dim=512, num_layers=6, dropout=0.3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2  #(32/4)^2 = 64
        self.embed_dim = embed_dim
        #原始图像，[batch_size, 3, 32, 32]，输入

        #Patch Embedding，[batch_size, 64, 256]，nn.Conv2d(3→256, kernel=4, stride=4)
        #32*32的图像被分成64个patch（每个patch有4*4*3=48个像素），每个patch（有rgb三维）被嵌入为256维度(每个4x4x3的patch与256个核做点积,使得每个patch→256维向量)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        #[class]令牌，用于最终分类
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  #第一个1为占位符，会自动广播成batch_size

        #为每个patch添加位置信息（+1是为[class]令牌预留位置）
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  #第一个1为占位符，会自动广播成batch_size

        #执行6次Transformer Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        #最终层归一化
        self.norm = nn.LayerNorm(embed_dim)

        #分类层
        self.fc = nn.Linear(embed_dim, num_classes)

        #初始化可学习参数
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]

        #Patch Embedding
        x = self.patch_embed(x)   #[batch_size, 3, 32, 32]->[batch_size, 256, 8, 8]
        x = x.flatten(2).transpose(1, 2)   #[batch_size, 256, 8, 8]展平->[batch_size, 256, 64]转置为适合Transformer处理的序列格式->[batch_size, 64, 256]

        class_token = self.class_token.expand(batch_size, -1, -1)  #初始化令牌[batch_size, 1, 256]
        x = torch.cat([class_token, x], dim=1)   #将令牌拼接到x前面
        x = x + self.pos_embed  #添加位置编码

        #执行6次Transformer Encoder
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.norm(x)  #层归一化
        class_token_final = x[:, 0]  #取[class]令牌的输出[batch_size, 256]
        out = self.fc(class_token_final)  #[batch_size, 10]
        return out


#数据准备（划分训练/验证/测试集）
def prepare_data():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  #随机水平翻转，50%概率增强数据多样性
        transforms.RandomCrop(32, padding=4),  #随机裁剪，32x32输入，四周填充4像素后随机裁剪
        transforms.ToTensor(),   #转化数据类型,归一化范围,通道顺序匹配
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  #归一化，将像素值从[0,1]映射到[-1,1]，RGB三个通道的均值与标准差均设为0.5
    ])

    #完整训练集（5万张）
    full_trainset = torchvision.datasets.CIFAR10(
        root='./DataSet', train=True, download=True, transform=transform)

    #划分训练集（4万）和验证集（1万）
    train_size = 40000
    val_size = 10000
    train_set, val_set = torch.utils.data.random_split(
        full_trainset, [train_size, val_size])

    #测试集（1万张）
    test_set = torchvision.datasets.CIFAR10(
        root='./DataSet', train=False, download=True, transform=transform)

    #创建 DataLoader,将数据集分成多个小批次(batch),用于小批量梯度下降(Mini-batch GD)
    #分成多个小批量,可以使得从原本1个epoch进行一次梯度下降,变为1个epoch进行n次梯度下降(n为小批量个数)
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=0)  #每次从数据集中取出 128 张图像和对应的标签，训练时,随机打乱数据顺序
    valloader = torch.utils.data.DataLoader(
        val_set, batch_size=128, shuffle=False, num_workers=0)   #测试时,不随机打乱数据顺序
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=0)

    return trainloader, valloader, testloader


#训练和验证函数
def train_and_validate(model, trainloader, valloader, testloader, epochs=30):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #多分类任务标准损失函数（内置softmax）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(trainloader, desc=f'Train Epoch {epoch + 1}/{epochs}'):

            #数据迁移
            inputs, labels = inputs.to(device), labels.to(device)

            #梯度清零，防止梯度累积
            optimizer.zero_grad()

            #前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #反向传播，计算梯度
            loss.backward()

            #参数更新，根据梯度更新参数
            optimizer.step()

            #统计指标
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_loss /= len(trainloader)  #平均损失
        train_acc = 100. * correct / total  #准确率百分比
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        #无梯度计算（torch.no_grad）无参数更新（无optimizer.step()）
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(valloader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()

        scheduler.step()

        print(f'Epoch {epoch + 1}/{epochs}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # 保存最佳模型
    torch.save(best_model_state, 'best_vit_model.pth')

    # 最终测试
    model.load_state_dict(best_model_state)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total
    print(f'Final Test Accuracy: {test_acc:.2f}%')

    return train_losses, val_losses, train_accs, val_accs


#可视化训练曲线
def plot_training(train_losses, val_losses, train_accs, val_accs, epochs):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Train Loss')
    plt.plot(range(1, epochs + 1), val_losses, 'b--', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_accs, 'r-', label='Train Acc')
    plt.plot(range(1, epochs + 1), val_accs, 'r--', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('vit_training.png')
    plt.show()


#主函数
def main():
    freeze_support()  #Windows多进程支持

    #准备数据
    trainloader, valloader, testloader = prepare_data()

    model = SimpleViT(
        image_size=32,  #输入图像尺寸（CIFAR-10为32x32）
        patch_size=4,   #每个图像块的大小（4x4像素）
        num_classes=10, #分类类别数（CIFAR-10有10类）
        embed_dim=256,  #嵌入向量维度（每个patch编码后的特征长度）
        num_heads=8,    #多头注意力机制的头数
        ff_dim=512,     #前馈网络（FFN）的隐藏层维度
        num_layers=6,   #Transformer编码器的层数
        dropout=0.3     #Dropout概率（正则化强度）
    )

    #训练和验证、最优参数测试
    train_losses, val_losses, train_accs, val_accs = train_and_validate(
        model, trainloader, valloader, testloader, epochs=30)

    #可视化
    plot_training(train_losses, val_losses, train_accs, val_accs, epochs=30)


if __name__ == '__main__':
    main()
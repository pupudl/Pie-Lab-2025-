import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import time

# 设置随机种子以确保结果可复现
t.manual_seed(42)
np.random.seed(42)

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #对图像进行随机裁剪
    transforms.RandomHorizontalFlip(),          #以50%的概率对图像进行水平翻转
    transforms.ToTensor(),  # 转化数据类型,归一化范围,通道顺序匹配
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化,RGB 三个通道的均值与标准差均设为 0.5
])

# 训练集
trainset = tv.datasets.CIFAR10(
    root='DataSet/',  # 数据集路径
    train=True,
    download=False,
    transform=transform
)

# 测试集
testset = tv.datasets.CIFAR10(
    root='DataSet/',
    train=False,
    download=False,
    transform=transform
)

# 创建 DataLoader,将数据集分成多个小批次(batch),用于小批量梯度下降(Mini-batch GD)
# 分成多个小批量,可以使得从原本1个epoch进行一次梯度下降,变为1个epoch进行n次梯度下降(n为小批量个数)
trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=128, #每次从数据集中取出 128 张图像和对应的标签
    shuffle=True,   #训练时,随机打乱数据顺序
    num_workers=2
)

testloader = t.utils.data.DataLoader(
    testset,
    batch_size=128,
    shuffle=False,   #测试时,不随机打乱数据顺序
    num_workers=2
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 原始CNN模型
#输入(3*32*32)->卷积层1->ReLU->池化层->卷积层2->ReLU->池化层->卷积层3->ReLU->池化层->展平->全连接层1->ReLU->全连接层2->ReLU->全连接层3->输出(10类)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #311的卷积层 输入:3通道(RGB 图像),输出:16通道,卷积后尺寸为32*32(因padding=1) (32+2-3)/1+1=32
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        #311的卷积层 输入:16通道(conv1 的输出),输出:32通道,卷积后尺寸16*16(池化了)->卷积后仍为16*16(同上311卷积尺寸不变)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        #311的卷积层 输入:32通道(conv2 的输出),输出:64通道,卷积后尺寸8*8(池化了)->卷积后仍为8*8(同上311卷积尺寸不变)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        #22的池化层 步长2(无重叠采样),每次池化后尺寸缩小为原来的1/2
        self.pool = nn.MaxPool2d(2, 2)

        #全连接层 输入:展平后的卷积特征(64通道*4*4尺寸),输出:512维特征
        self.fc1 = nn.Linear(64 * 4 * 4, 512)

        #全连接层 输入:512维特征,输出:128维特征
        self.fc2 = nn.Linear(512, 128)

        #全连接层 输入:128维特征,输出:10维(对应CIFAR-10的10个类别,最终分类得分)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):

        #尺寸从32*32缩小到16*16,通道数从3增加到16
        x = self.pool(F.relu(self.conv1(x)))

        #尺寸缩小到8*8,通道数增加到32
        x = self.pool(F.relu(self.conv2(x)))
        
        #尺寸缩小到4*4,通道数增加到64(最终卷积特征为64*4*4)
        x = self.pool(F.relu(self.conv3(x)))

        #将三维卷积特征(64*4*4)展平为一维向量(64*4*4=1024),view(-1, 1024)中-1表示自动计算批次维度(保持batch_size不变)
        x = x.view(-1, 64 * 4 * 4)

        #全连接层传播:w*x+b,eg.x为（400,1）,w为（120,400）,b为（120，1）->（120,1）
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #输出10个类别的得分,无需激活,因损失函数含SoftMax
        x = self.fc3(x)
        return x


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        #第一个卷积层:可能改变通道数和尺寸
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)   #二维批量归一化

        #第二个卷积层:保持通道数和尺寸不变(311)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)   #二维批量归一化

        #捷径连接,解决在跨层传递时,输入与输出通道数不一致或特征图尺寸不匹配的问题
        #将输入x直接加到卷积输出F(x)上,eg.a[l+2] = g(z[l+2]+a[l]) = g(w[l+2]*a[l+1]+b[l+2]+a[l])
        #有时输入x与卷积输出F(x)的形状不兼容,无法直接相加,需要通过self.shortcut对输入x进行维度匹配

        #默认初始化
        self.shortcut = nn.Sequential()

        #当stride!=1(需要下采样)或in_channels!=out_channels(通道数变化)时,需要调整输入x的维度
        if stride != 1 or in_channels != out_channels:

            #1*1卷积调整维度(改变通道数)(使用目标通道数(out_channels)个1*1滤波器处理)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)  #标准化调整后的特征,稳定训练
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))   #第一个卷积+BN+ReLU
        out = self.bn2(self.conv2(out))         #第二个卷积+BN(没有ReLU)
        out += self.shortcut(x)                 #关键:残差连接,将输入直接加到输出
        out = F.relu(out)                       #最终ReLU激活
        return out


# 残差网络,堆叠多个残差块构建深层网络
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16

        #初始卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        #三个残差模块(每个模块包含多个残差块)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)  # 特征图尺寸不变
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)  # 特征图尺寸减半
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)  # 特征图尺寸减半

        #全局平均池化+全连接分类器
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))    #无论输入尺寸如何,输出1*1特征图
        self.fc = nn.Linear(64, num_classes)  #最终分类

    #残差模块构建
    def _make_layer(self, block, out_channels, num_blocks, stride):

        #第一个残差块使用指定的stride(可能为2,用于下采样),后续残差块的stride均为1(保持特征图尺寸不变)
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            
            #第一个残差块:可能通过stride=2进行下采样,若self.in_channels != out_channels,使用1*1卷积调整捷径连接的维度。
            #后续残差块:stride=1,不改变特征图尺寸,由于self.in_channels已更新为out_channels,输入输出通道数一致,无需调整捷径连接。
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels  #更新输入通道数为当前输出通道数

        #将所有残差块按顺序组合成一个完整的模块
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  #初始卷积+BN+ReLU

        #通过三个残差模块,逐步提取特征
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # 全局平均池化+展平+全连接分类
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)  #展平为一维向量
        out = self.fc(out)               #最终分类得分
        return out


# 训练函数
def train_model(model, criterion, optimizer, epochs, device):

    model.to(device)       #将模型移动到指定设备(GPU/CPU)
    train_losses = []      #记录每轮训练集损失
    val_losses = []        #记录每轮验证集损失
    train_accuracies = []  #记录每轮训练集准确率
    val_accuracies = []    #记录每轮验证集准确率
    best_acc = 0.0         #跟踪验证集最佳准确率(用于保存最优模型)

    for epoch in range(epochs):

        model.train()       #切换模型到训练模式
        running_loss = 0.0  #累计当前epoch的训练损失
        correct = 0         #累计当前epoch的正确预测数
        total = 0           #累计当前epoch的总样本数
        start_time = time.time()  #记录当前epoch的开始时间

        for i, (inputs, labels) in enumerate(trainloader):
            # 数据准备:将输入和标签移动到指定设备
            inputs, labels = inputs.to(device), labels.to(device)

            #梯度清零:避免上一批次的梯度影响当前批次
            optimizer.zero_grad()

            #前向传播:模型预测输出
            outputs = model(inputs)   #输出形状: [batch_size, 10](10类别的得分)

            #计算损失:预测值与真实标签的差异
            loss = criterion(outputs, labels)

            #反向传播:计算损失对所有参数的梯度
            loss.backward()

            #优化器进行参数更新:根据梯度调整模型参数
            optimizer.step()

            #跟踪训练指标
            running_loss += loss.item()  #累加损失(loss.item()获取标量值)

            #计算当前批次的准确率
            _, predicted = outputs.max(1)        #取每行最大值的索引(预测类别),shape: [batch_size],128张图片分别有10个数字代表十类事物,取最大数字的索引
            # 第一个返回值是最大值(置信度),用下划线忽略
            # 第二个返回值是最大值的索引(预测的类别)

            total += labels.size(0)              #累加总样本数(batch_size)
            correct += predicted.eq(labels).sum().item()   #累加正确预测数

        #计算当前epoch的平均训练损失和准确率
        train_loss = running_loss / len(trainloader)  #总损失/批次数量
        train_acc = 100.0 * correct / total           #正确数/总样本数(转为百分比)

        #记录指标
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 验证,验证阶段不更新模型参数,仅评估模型在未见过的数据上的表现
        model.eval()     #切换模型到验证模式(关闭 dropout、固定 BatchNorm 统计量)
        val_loss = 0     #累计当前epoch的验证损失
        val_correct = 0  #累计当前epoch的正确预测数
        val_total = 0    #累计当前epoch的总样本数

        #关闭梯度计算(节省内存,加速验证)
        with t.no_grad():
            for inputs, labels in testloader:
                #数据准备(同训练阶段)
                inputs, labels = inputs.to(device), labels.to(device)

                #前向传播(无反向传播和参数更新)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                #跟踪验证指标
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        #计算当前epoch的平均验证损失和准确率
        val_loss = val_loss / len(testloader)
        val_acc = 100.0 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        #打印当前epoch的训练时间和指标
        end_time = time.time()
        print(f'Epoch {epoch + 1}/{epochs} | Time: {end_time - start_time:.2f}s | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

        #保存验证集准确率最高的模型(避免保存过拟合的模型)
        if val_acc > best_acc:
            best_acc = val_acc
            t.save(model.state_dict(), 'best_model.pth')  #仅保存参数,不保存模型结构

    return train_losses, val_losses, train_accuracies, val_accuracies


# 测试函数
def test_model(model, device):
    model.to(device)  #将模型移动到指定设备
    model.eval()      #切换模型到评估模式
    correct = 0       #记录所有测试样本中预测正确的数量
    total = 0         #记录测试样本的总数量

    #关闭梯度计算
    with t.no_grad():
        for inputs, labels in testloader:  #迭代测试集的每个批次

            #将输入数据和标签移动到指定设备(与模型同设备)
            inputs, labels = inputs.to(device), labels.to(device)

            #前向传播:模型预测输出
            outputs = model(inputs)   #输出形状: [batch_size, 10](10类别的得分)

            #提取预测结果
            _, predicted = outputs.max(1)   #取每行最大值的索引(预测类别),shape: [batch_size],128张图片分别有10个数字代表十类事物,取最大数字的索引
            #第一个返回值是最大值(置信度),用下划线忽略
            #第二个返回值是最大值的索引(预测的类别)

            total += labels.size(0)  # 累加总样本数(batch_size)
            correct += predicted.eq(labels).sum().item()  # 累加正确预测数

    #计算整体准确率(正确数/总数,转为百分比)
    accuracy = 100.0 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')  #保留两位小数输出
    return accuracy    #返回准确率供后续对比分析


# 可视化训练过程
def plot_training(cnn_train_losses, cnn_val_losses, cnn_train_acc, cnn_val_acc,
                  resnet_train_losses, resnet_val_losses, resnet_train_acc, resnet_val_acc, epochs):
    plt.figure(figsize=(15, 5))

    #绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), cnn_train_losses, 'b-', label='CNN Train Loss')
    plt.plot(range(1, epochs + 1), cnn_val_losses, 'b--', label='CNN Val Loss')
    plt.plot(range(1, epochs + 1), resnet_train_losses, 'r-', label='ResNet Train Loss')
    plt.plot(range(1, epochs + 1), resnet_val_losses, 'r--', label='ResNet Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    #绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), cnn_train_acc, 'b-', label='CNN Train Acc')
    plt.plot(range(1, epochs + 1), cnn_val_acc, 'b--', label='CNN Val Acc')
    plt.plot(range(1, epochs + 1), resnet_train_acc, 'r-', label='ResNet Train Acc')
    plt.plot(range(1, epochs + 1), resnet_val_acc, 'r--', label='ResNet Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_comparison.png')
    plt.show()


def main():
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')

    epochs = 30
    lr = 0.001  #学习率

    # 训练CNN
    print("\n===== 训练CNN =====")
    cnn = CNN()

    #交叉熵损失函数,用于多分类问题,直接处理模型的原始输出,无需手动添加Softmax层
    cnn_criterion = nn.CrossEntropyLoss()

    #Adam优化器,自适应调整每个参数的学习率,自动为不同参数分配动态学习率
    cnn_optimizer = optim.Adam(cnn.parameters(), lr=lr)

    cnn_train_losses, cnn_val_losses, cnn_train_acc, cnn_val_acc = train_model(
        cnn, cnn_criterion, cnn_optimizer, epochs, device
    )

    cnn_accuracy = test_model(cnn, device)

    # 训练ResNet
    print("\n===== 训练ResNet =====")
    resnet = ResNet(ResidualBlock, [2, 2, 2])  # 创建一个11层的ResNet

    # 交叉熵损失函数,用于多分类问题,直接处理模型的原始输出,无需手动添加Softmax层
    resnet_criterion = nn.CrossEntropyLoss()

    # Adam优化器,自适应调整每个参数的学习率,自动为不同参数分配动态学习率
    resnet_optimizer = optim.Adam(resnet.parameters(), lr=lr)

    resnet_train_losses, resnet_val_losses, resnet_train_acc, resnet_val_acc = train_model(
        resnet, resnet_criterion, resnet_optimizer, epochs, device
    )

    resnet_accuracy = test_model(resnet, device)

    # 可视化训练过程
    plot_training(cnn_train_losses, cnn_val_losses, cnn_train_acc, cnn_val_acc,
                  resnet_train_losses, resnet_val_losses, resnet_train_acc, resnet_val_acc, epochs)

    # 打印最终结果
    print("\n===== 对比结果 =====")
    print(f'CNN最终准确率: {cnn_accuracy:.2f}%')
    print(f'ResNet最终准确率: {resnet_accuracy:.2f}%')
    print(f'准确率提升: {resnet_accuracy - cnn_accuracy:.2f}%')

    # 分析收敛速度
    cnn_best_epoch = np.argmax(cnn_val_acc) + 1
    resnet_best_epoch = np.argmax(resnet_val_acc) + 1
    print(f'CNN最佳准确率出现在第 {cnn_best_epoch} 个epoch')
    print(f'ResNet最佳准确率出现在第 {resnet_best_epoch} 个epoch')


if __name__ == '__main__':
    main()
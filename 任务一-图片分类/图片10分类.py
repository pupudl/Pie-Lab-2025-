import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

show = ToPILImage()  # 把 Tensor 转成 Image，方便可视化

# 定义对数据的预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为 Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
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

# 创建 DataLoader
trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

testloader = t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)  # 最后是一个十分类，所以最后的一个全连接层的神经元个数为10

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)  # 展平  x.size()[0]是batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test():
    # 检查单张图片
    data, label = trainset[66]
    print(f"图片维度: {data.size()}")  # 3x32x32
    print(f"类别: {classes[label]}")  # 输出类别名称

    # 显示单张图片
    show((data + 1) / 2).resize((100, 100)).show()

    # 检查一个 batch 的数据
    dataiter = iter(trainloader)
    images, labels = next(dataiter)  # 返回 4 张图片及标签

    # 打印 batch 的类别
    print('Batch 类别:', ' '.join('%11s' % classes[labels[j]] for j in range(4)))

    # 显示 batch 图片
    show(tv.utils.make_grid((images + 1) / 2)).resize((400, 100)).show()

def train():
    t.set_num_threads(4)
    for epoch in range(5):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):  # i 第几个batch     data：一个batch中的数据

            # 输入数据
            inputs, labels = data  # images：batch大小为4     labels：batch大小为4

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 打印log信息
            # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
            running_loss += loss.item()
            if i % 2000 == 1999:  # 每2000个batch打印一下训练状态
                print('[%d, %5d] loss: %.3f' \
                      % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    # 保存模型参数
    t.save(net.state_dict(), 'cifar_net.pth')
    print('模型已保存为 cifar_net.pth')
    print('Finished Training')

def prediction():
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数

    # 加载保存的模型参数
    net.load_state_dict(t.load('cifar_net.pth'))
    print('已加载预训练模型参数')

    # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
    with t.no_grad():
        for data in testloader:  # data是个tuple
            images, labels = data  # image和label 都是tensor
            outputs = net(images)
            _, predicted = t.max(outputs, 1)
            total += labels.size(0)  # labels tensor([3, 8, 8, 0])            labels.size: torch.Size([4])
            correct += (predicted == labels).sum()

    print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    #test()
    net = Net() #加载网络模型
    print(net)

    #定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #训练模型
    #train()

    #预测结果
    prediction()

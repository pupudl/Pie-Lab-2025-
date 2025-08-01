from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import batch_size

#数据加载
def get_dataloader():
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转化数据类型,归一化范围,通道顺序匹配
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化,将像素值从[0,1]映射到[-1,1],RGB三个通道的均值与标准差均设为0.5
    ])

    train_dataset = datasets.CIFAR10(
        root='./DataSet',
        train=True,
        download=True,
        transform=transform
    )

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
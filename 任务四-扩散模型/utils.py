import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import os
from PIL import Image

#将模型输出的张量转换为可显示的图像格式
def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),  #反归一化，将[-1,1]范围映射到[0,1]
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  #调整维度顺序：从(C,H,W)转为(H,W,C)
        transforms.Lambda(lambda t: t * 255.),  #缩放到0-255范围
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),  #转换为uint8类型
    ])
    if len(image.shape) == 4:  #如果输入是4D张量（batch,C,H,W）
        image = image[0, :, :, :]  #取batch中第一个样本
    plt.imshow(reverse_transforms(image))

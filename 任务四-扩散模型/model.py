import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import device


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        #生成一组从高频到低频变化的基频，用于构建不同周期的时间信号
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        ).to(device)
        self.register_buffer('inv_freq', inv_freq)

    #正弦余弦交织，将离散的时间步转换为连续的向量表示
    def forward(self, t):
        pos_enc = t[:, None] * self.inv_freq[None, :]
        return torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=-1)

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()

        #将时间步的嵌入向量（32）线性映射到与当前层特征通道数（out_ch）相同的维度,让时间信息能适配不同深度的特征图
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        #上采样：使用转置卷积进行2倍上采样
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1) #因为拼接跳跃连接，所以输入通道翻倍
            #插零，在输入特征图的每个元素周围插入stride-1个零；补边，在插零后的矩阵外围补kernel_sizepadding-1个零；卷积计算
            #(batch, 256, 16, 16)->(batch, 256, 31, 31)->(batch, 256, 35, 35)->(batch, 256, 32, 32)
            #上采样恢复分辨率,将低分辨率特征图放大到原始尺寸，转置卷积的权重会受时间嵌入向量调制，使上采样策略随扩散阶段动态调整,学习最优上采样方式，适应不同噪声水平的需求。
            #处理的特征会与编码器路径的跳跃连接特征拼接，保留多尺度信息
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1) #转置卷积上采样

        #下采样块：使用步幅2的普通卷积进行下采样
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1) #使用3*3卷积保持空间尺寸（padding=1），仅改变通道数（in_ch->out_ch）
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1) #对于输入32x32->输出16x16（2倍缩小），相比池化可保留可学习性

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)  #第二层卷积（保持尺寸）
        self.bnorm1 = nn.BatchNorm2d(out_ch)  #第一个批归一化层
        self.bnorm2 = nn.BatchNorm2d(out_ch)  #第二个批归一化层
        self.relu = nn.ReLU()  #激活函数

    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))    #根据上/下采样选择第一层卷积->relu激活->第一个批归一化层
        time_emb = self.relu(self.time_mlp(t))       #时间嵌入通过MLP->激活
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1) #广播相加
        h = self.bnorm2(self.relu(self.conv2(h)))    #第二层卷积->relu激活->第二个批归一化层，在时间条件引导后进一步提炼特征
        return self.transform(h)

#处理多尺度信息：通过下采样（收缩路径）捕获全局结构，上采样（扩展路径）恢复细节
#时间步感知：将时间步信息嵌入到每一层，使模型能动态调整去噪策略。
class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3  #输入图像的通道数（RGB）
        down_channels = (64, 128, 256, 512, 1024)  #下采样各层通道数
        up_channels = (1024, 512, 256, 128, 64)  #上采样各层通道数（对称）
        out_dim = 3  #输出维度（预测的噪声维度）
        time_emb_dim = 32  #时间嵌入的维度

        #将整数时间步t转换为高维向量并进一步非线性变换
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1) #将输入图像（3通道）映射到第一层特征空间（64通道）

        #4个下采样块（64->128->256->512->1024）,每个块包含卷积层、时间嵌入注入、下采样（ stride=2卷积）
        self.downs = nn.ModuleList([
            Block(down_channels[i], down_channels[i + 1], time_emb_dim)
            for i in range(len(down_channels) - 1)
        ])
        #4个上采样块(1024->512->256->128->64)，需要拼接来自编码器路径的跳跃连接特征
        self.ups = nn.ModuleList([
            Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True)
            for i in range(len(up_channels) - 1)
        ])

        #将64通道特征映射回3通道噪声预测（1x1卷积保持分辨率）
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        t = self.time_mlp(timestep)  #时间步->32维向量
        x = self.conv0(x)  #初始卷积

        residual_inputs = []
        for down in self.downs:
            x = down(x, t)  #下采样块处理
            residual_inputs.append(x)  #每层输出被存入residual_inputs用于跳跃连接

        #上采样，跳跃连接是指在网络结构中，绕过中间层，将某一层的输出直接与更深层的输入合并的短路连接
        #ResNet的残差连接是直接相加，U-Net的跳跃连接是拼接
        #解决梯度消失问题，还可以保留多尺度信息，因为连续下采样会丢失细节，因此将编码器的低层特征（富含细节）直接传递给解码器。
        for up in self.ups:
            residual_x = residual_inputs.pop()  #取出对应层特征（从后往前,通道数对应）
            x = torch.cat((x, residual_x), dim=1)  #将当前特征与跳跃特征拼接（通道数翻倍）
            x = up(x, t)  #上采样块处理

        return self.output(x)
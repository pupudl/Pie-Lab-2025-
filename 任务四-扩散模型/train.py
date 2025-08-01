from model import SimpleUnet
from data import get_dataloader
from diffusion import forward_diffusion_sample,sample_timestep
from utils import show_tensor_image
from config import *
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
import time

#从纯噪声开始，通过逐步去噪生成样本图像，并保存可视化结果
def generate_samples(model, epoch, num_samples=5):
    #保存到samples目录
    os.makedirs("samples", exist_ok=True)
    #切换为评估模式
    model.eval()
    #禁用梯度计算
    with torch.inference_mode():
        samples = []
        for _ in range(num_samples):
            #从标准高斯噪声开始
            img = torch.randn((1, 3, 32, 32), device=device)
            #从T到0逐步去噪
            for t in reversed(range(T)):
                timestep = torch.full((1,), t, device=device, dtype=torch.long)
                #调用反向采样函数预测并去噪一步
                img = sample_timestep(model, img, timestep)
            #保存生成结果
            samples.append(img.cpu())

    #可视化生成样本
    plt.figure(figsize=(15, 3))
    for i, sample in enumerate(samples):
        plt.subplot(1, num_samples, i + 1)
        show_tensor_image(sample)
        plt.axis('off')
    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(f"samples/epoch_{epoch}.png")
    plt.close()
    #恢复训练模式
    model.train()

#绘制训练曲线
def plot_training_curve(log_file="training_log.csv"):
    # 读取日志文件
    df = pd.read_csv(log_file)

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['loss'], 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()

    # 绘制时间曲线
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['time_per_epoch'], 'r-', label='Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curve.png')
    plt.close()
    print("训练曲线已保存为 training_curve.png")

#训练
def train():
    train_loader = get_dataloader()  #获取数据加载器
    model = SimpleUnet().to(device)  #初始化U-Net模型并移至GPU
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  #Adam优化器
    scaler = torch.cuda.amp.GradScaler()  #混合精度梯度缩放器

    #创建训练日志
    log_df = pd.DataFrame(columns=['epoch', 'loss', 'time_per_epoch'])

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        start_time = time.time()  #记录epoch开始时间
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch in progress_bar:
            #梯度归零
            optimizer.zero_grad()
            #图像数据移至GPU
            batch = batch[0].to(device, non_blocking=True)
            batch_size = batch.shape[0]
            #随机时间步，确保模型学习所有扩散阶段的去噪能力
            t = torch.randint(0, T, (batch_size,), device=device).long()

            with torch.cuda.amp.autocast():
                #获取时间步t时，加噪后的图像和所加噪声
                x_noisy, noise = forward_diffusion_sample(batch, t)
                #模型预测噪声
                noise_pred = model(x_noisy, t)
                #噪声预测损失（公式14）
                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()  #梯度缩放的反向传播
            scaler.step(optimizer)  #缩放后的梯度更新
            scaler.update()  #调整缩放系数

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        #计算epoch耗时
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

        # 记录日志
        log_df.loc[len(log_df)] = [epoch, avg_loss, epoch_time]
        log_df.to_csv("training_log.csv", index=False)

        #每10个epoch保存一次检查点
        if epoch % 10 == 0:
            generate_samples(model, epoch)
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

    #训练完成后绘制曲线
    plot_training_curve()

    return model

if __name__ == "__main__":
    trained_model = train()
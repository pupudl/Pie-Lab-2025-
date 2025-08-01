import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from model import SimpleUnet
from diffusion import sample_timestep
from utils import show_tensor_image
from config import device, T


def visualize_diffusion_generation(model_path="model_epoch_100.pth", output_dir="generation_process", num_steps=20):

    #创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    #初始化模型并加载训练好的模型权重
    model = SimpleUnet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  #切换到评估模式

    #从纯噪声开始
    img = torch.randn((1, 3, 32, 32), device=device)

    #选择要保存的时间步(从T到0均匀采样)，间隔时间步捕捉，扩散过程中关键变化阶段
    save_steps = np.linspace(T - 1, 0, num_steps, dtype=int)

    #创建大图用于显示整个过程
    plt.figure(figsize=(20, 10))

    #反向扩散过程
    for i, t in enumerate(tqdm(reversed(range(T)), desc="Generating")):
        with torch.no_grad():
            # 准备时间步张量
            timestep = torch.full((1,), t, device=device, dtype=torch.long)

            #调用反向采样函数预测并去噪一步
            img = sample_timestep(model, img, timestep)

            #如果是需要保存的步骤
            if t in save_steps:
                step_idx = np.where(save_steps == t)[0][0]

                #在大图中添加子图
                plt.subplot(2, num_steps // 2, step_idx + 1)
                show_tensor_image(img.cpu())
                plt.title(f"t={T - t}")
                plt.axis('off')

    #保存完整过程图
    plt.tight_layout()
    plt.savefig(f"{output_dir}/full_process.png")
    plt.close()
    print(f"生成过程已保存到 {output_dir} 目录")



if __name__ == "__main__":
    visualize_diffusion_generation()
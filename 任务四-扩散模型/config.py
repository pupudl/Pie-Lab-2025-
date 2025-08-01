import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#随机种子设置
torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed_all(42)

#扩散过程参数
T = 1000  #扩散步数
beta_start = 0.0001  #噪声起始系数
beta_end = 0.02  #噪声结束系数

#训练参数
batch_size = 128
epochs = 100
learning_rate = 1e-4

#计算扩散参数

#计算累积乘积α，用于控制不同时间步的噪声强度
betas = torch.linspace(beta_start, beta_end, T).to(device)   #噪声率betas是从0.0001到0.02的等差数列
alphas = 1. - betas  #信号保留率alpha= 1-噪声率beta
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)   #累计信号率是信号保留率alpha的累乘，表示累积信号衰减系数，用于一步计算任意时刻t的噪声图像

alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0).to(device)  #表示前一时间步的累计信号率，[1.0, ᾱ_{0}, ᾱ_{1}, ..., ᾱ_{T-1}],用于计算反向过程的后验方差σt^2
sqrt_recip_alphas = torch.sqrt(1.0 / alphas).to(device)  #在反向采样时用于计算去噪均值μθ

#用于前向扩散过程的闭式解公式计算
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)  #控制原始图的保留强度
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).to(device)  #控制添加噪声的强度

#计算扩散模型反向过程中的后验方差，即当已知x_t和x_0时，x_t−1的真实条件分布方差
posterior_variance = (betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)).to(device)
from config import *

#前向扩散过程（根据原始图像x_0和时间步t，生成加噪后的图像x_t）
def forward_diffusion_sample(x_0, t):
    #生成与x_0同形状的标准高斯噪声
    noise = torch.randn_like(x_0, device=device)

    #获取当前时间步的累积噪声系数
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

    #计算加噪后的图像（闭式解公式）
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

#反向去噪过程（使用训练好的模型model，从带噪图像x_t预测并去除噪声）
def sample_timestep(model, x, t):
    #禁用梯度计算
    with torch.inference_mode():
        #获计算所需要的参数
        betas_t = betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = sqrt_recip_alphas[t].reshape(-1, 1, 1, 1)

        #计算模型预测的均值（公式11）
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        #根据时间步添加随机噪声
        if t == 0:  #最后一步不添加噪声
            return model_mean
        else:       #添加随机性
            posterior_variance_t = posterior_variance[t].reshape(-1, 1, 1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
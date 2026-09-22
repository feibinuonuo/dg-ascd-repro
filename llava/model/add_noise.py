import torch

# def random_mask(image_features, mask_ratio, pad_method='zero'):
#     bs, img_tok, d = image_features.shape
#     mask_index = torch.randperm(img_tok)[:int(mask_ratio * img_tok)]
#     if pad_method == "zero":
#         image_features[:, mask_index, :] = torch.zeros(bs, 1, d).to(image_features.dtype).to(image_features.device)
#     elif pad_method == "mean":
#         image_features[:, mask_index, :] = image_features.mean(1, keepdim=True)
#     elif pad_method == "pad_token_embed":
#         image_features[:, mask_index, :] = self.language_model.get_input_embeddings()(torch.tensor([self.tokenizer.pad_token_id]).to(image_features.device))
#     else:
#         raise ValueError("The pad method can only be either zero or mean!")
#     return image_features

def add_gauss_noise(image_features, sigma):
    noise = torch.randn_like(image_features) * sigma
    return image_features + noise

def add_diffusion_noise(image_tensor, noise_step):
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd

def add_noise_to_feature(image_features,
                         mask_ratio_mean=0.0,
                         mask_ratio_zero=0.0,
                         mask_ratio_pad=0.0,
                         sigma_gauss_noise=0.0,
                         diff_noise_step=0):
    image_features = random_mask(image_features, mask_ratio=mask_ratio_mean, pad_method='mean') if mask_ratio_mean > 0.0 else image_features
    image_features = random_mask(image_features, mask_ratio=mask_ratio_zero, pad_method='zero') if mask_ratio_zero > 0.0 else image_features
    image_features = random_mask(image_features, mask_ratio=mask_ratio_pad, pad_method='pad_token_embed') if mask_ratio_pad > 0.0 else image_features
    image_features = add_gauss_noise(image_features, sigma_gauss_noise) if sigma_gauss_noise > 0.0 else image_features
    image_features = add_diffusion_noise(image_features, diff_noise_step) if diff_noise_step > 0 else image_features
    
    return image_features
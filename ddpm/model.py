import torch
from torch import nn


class DDPM(nn.Module):
    def __init__(self, denoiser: nn.Module, scheduler: str, denoising_steps: int):
        super().__init__()
        self.denoiser = denoiser
        self.denoising_steps = denoising_steps

        if scheduler == 'linear_beta':
            self._linear_beta_scheduler(denoising_steps)
        else:
            raise ValueError(f"Unsupported scheduler '{scheduler}'.")

    def _linear_beta_scheduler(self, denoising_steps, beta_start=1e-4, beta_end=2e-2) -> None:
        betas = torch.linspace(beta_start, beta_end, denoising_steps)
        alphas = 1 - betas
        sqrt_alphas = alphas.sqrt()
        alphas_cumprod = alphas.cumprod(0)
        sqrt_alphas_cumprod = alphas_cumprod.sqrt()
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod).sqrt()

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("sqrt_alphas", sqrt_alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)

    def noise(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dim_match = (x.size(0),) + (1,) * (x.dim() - 1)
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t].view(dim_match)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(dim_match)
        eps = torch.randn_like(x)
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * eps, eps

    def denoise(self, x: torch.Tensor, t: torch.Tensor, *args) -> tuple[torch.Tensor, torch.Tensor]:
        dim_match = (x.size(0),) + (1,) * (x.dim() - 1)
        betas = self.betas[t].view(dim_match)
        pred_eps = self.denoiser(x, t, *args)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(dim_match)
        sqrt_alphas = self.sqrt_alphas[t].view(dim_match)
        return (x - betas * pred_eps / sqrt_one_minus_alphas_cumprod) / sqrt_alphas, pred_eps
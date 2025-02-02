# [Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)

### Authors
| Name              | Affiliation  |
|-------------------|--------------|
| **Jonathon Ho**   | UC Berkeley  |
| **Ajay Jain**     | UC Berkeley  |
| **Pieter Abbeel** | UC Berkeley  |

### Model Description
A class of generative models that iteratively refine noisy data through a learned denoising process, effectively
reversing a predefined forward diffusion process. By gradually adding Gaussian noise to data and then training a neural
network to reverse this corruption step-by-step, DDPMs can generate high-quality samples from complex data
distributions. Their ability to model rich, multi-modal data makes them particularly powerful for image and audio
synthesis tasks.

## [Full Implementation](model.py)

#### Custom Classes
| Class                | Description     |
|----------------------|-----------------|
| [Args](#Args-Class)  | Model arguments |
| [DDPM](#DDPM-Module) | Model           |

#### Dimension Symbols
| Symbol | Description                   |
|--------|-------------------------------|
| $B$    | Batch size                    |
| $D$    | List of additional dimensions |


*The entire model implementation is detailed below in what is believed to be the most intuitive order...*

---

### Args Class
```python
@dataclass
class Args:
    beta_start: float =1e-4
    beta_end: float = 2e-2
```

### DDPM Module
```python
class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model (DDPM).

    This module implements a DDPM which progressively adds noise to input data
    and then learns to remove it using a provided denoiser network. The diffusion
    process is controlled by a scheduler (currently supporting a linear beta scheduler).

    Attributes:
        denoiser (nn.Module): The denoising network used to predict the noise component.
        denoising_steps (int): The number of diffusion timesteps.
        betas (torch.Tensor): Beta values for the noise schedule.
        alphas (torch.Tensor): Alpha values computed as 1 - beta.
        sqrt_alphas (torch.Tensor): Square root of the alpha values.
        alphas_cumprod (torch.Tensor): Cumulative product of the alpha values.
        sqrt_alphas_cumprod (torch.Tensor): Square root of the cumulative product of alphas.
        sqrt_one_minus_alphas_cumprod (torch.Tensor): Square root of (1 - cumulative product of alphas).
    """
    betas: torch.Tensor
    alphas: torch.Tensor
    sqrt_alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor

    def __init__(self, denoiser: nn.Module, scheduler: str, denoising_steps: int, **kwargs):
        """Initializes the DDPM module.

        This module progressively adds noise to input data and then learns to remove it using
        a provided denoiser network. The diffusion process is controlled by a scheduler, which
        currently supports a linear beta schedule.

        Args:
            denoiser (nn.Module): The denoising network that predicts the noise component.
            scheduler (str): The type of noise schedule to use. Currently, only "linear_beta" is supported.
            denoising_steps (int): The number of diffusion timesteps used for noise addition and removal.
            **kwargs: Additional keyword arguments for configuring the scheduler.

        Raises:
            ValueError: If an unsupported scheduler type is provided.
        """
        super().__init__()
        self.denoiser = denoiser
        self.denoising_steps = denoising_steps

        if scheduler == 'linear_beta':
            self._linear_beta_scheduler(denoising_steps, **kwargs)
        else:
            raise ValueError(f"Unsupported scheduler '{scheduler}'.")

    def _linear_beta_scheduler(self, denoising_steps, **kwargs) -> None:
        """Configures a linear beta scheduler for the diffusion process.

        This method computes linearly spaced beta values between `beta_start` and `beta_end`,
        derives the corresponding alpha values, and computes the cumulative products and their
        square roots. The computed tensors are registered as buffers so they become part of the module's
        state but are not considered parameters.

        Args:
            denoising_steps (int): The number of diffusion timesteps.
            **kwargs: Additional keyword arguments for configuring the scheduler. Supported keys are:
                - beta_start (float, optional): The starting beta value.
                - beta_end (float, optional): The ending beta value.
        """
        args = Args(**kwargs)
        beta_start = args.beta_start
        beta_end = args.beta_end

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
        """Adds noise to the input tensor based on the diffusion process at timestep `t`.

        The function applies a noise schedule to the input `x` using precomputed coefficients.
        It returns the noisy tensor along with the noise component that was added.

        Args:
            x (torch.Tensor): The original input tensor (e.g., an image or a batch of images).
            t (torch.Tensor): Timestep indices indicating the level of noise to add.
                Should be a tensor of indices corresponding to the diffusion steps.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - The noisy version of `x`.
                - The noise tensor (`eps`) that was added to `x`.
        """
        dim_match = (x.size(0),) + (1,) * (x.dim() - 1)
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[t].view(dim_match)                      # (B, *[1]*len(D))
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(dim_match)  # (B, *[1]*len(D))
        eps = torch.randn_like(x)                                                              # (B, *D)
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * eps, eps              # (B, *D), (B, *D)

    def denoise(self, x: torch.Tensor, t: torch.Tensor, *args) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoises the input tensor `x` at timestep `t` using the denoiser network.

        The function uses the denoiser network to predict the noise component from the noisy input `x`
        and then computes the denoised tensor using the inverse of the noise addition process.

        Args:
            x (torch.Tensor): The noisy input tensor.
            t (torch.Tensor): Timestep indices corresponding to the noise levels applied to `x`.
            *args: Additional arguments to be passed to the denoiser network.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - The denoised tensor.
                - The predicted noise tensor (`pred_eps`) output by the denoiser network.
        """
        dim_match = (x.size(0),) + (1,) * (x.dim() - 1)
        betas = self.betas[t].view(dim_match)                                                  # (B, *[1]*len(D))
        pred_eps = self.denoiser(x, t, *args)                                                  # (B, *D)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(dim_match)  # (B, *[1]*len(D))
        sqrt_alphas = self.sqrt_alphas[t].view(dim_match)                                      # (B, *[1]*len(D))
        return (x - betas * pred_eps / sqrt_one_minus_alphas_cumprod) / sqrt_alphas, pred_eps  # (B, *D), (B, *D)
```

---
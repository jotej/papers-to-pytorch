# [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://openreview.net/pdf?id=a-xFK8Ymz5J)

### Authors

| Name                 | Affiliation                             | Email                                                 |
|----------------------|-----------------------------------------|-------------------------------------------------------|
| **Zhifeng Kong**     | Computer Science and Engineering, UCSD  | [z4kong@eng.ucsd.edu](mailto:z4kong@eng.ucsd.edu)     |
| **Jiaji Huang**      | Baidu Research                          | [huangjiaji@baidu.com](mailto:huangjiaji@baidu.com)   |
| **Kexin Zhao**       | Baidu Research                          | [kexinzhao@baidu.com](mailto:kexinzhao@baidu.com)     |
| **Wei Ping**         | NVIDIA                                  | [wping@nvidia.com](mailto:wping@nvidia.com)           |
| **Bryan Catanzaro**  | NVIDIA                                  | [bcatanzaro@nvidia.com](mailto:bcatanzaro@nvidia.com) |

### Model Description
A non-autoregressive denoising diffusion probabilistic model for waveform generation directly in the
time domain, supporting both conditional and unconditional audio synthesis.

## [Full Implementation](#All-Modules)

#### Dimension Symbols
| Symbol         | Description                                                 |
|----------------|-------------------------------------------------------------|
| $B$            | Batch size                                                  |
| $C_{in}$       | Number of input channels                                    |
| $C_{res}$      | Number of residual channels                                 |
| $N$            | Number of mel frequency bins in conditional Mel spectrogram |
| $T_{spec}$     | Total time frames in conditional Mel spectrogram            |
| $T_{sample}$   | Total samples in waveform                                   |

#### Custom Modules
| Module                                       | Description                               |
|----------------------------------------------|-------------------------------------------|
| [DiffWaveDenoiser](#DiffWaveDenoiser-Module) | Model denoiser                            |
| [TimestepEmbedder](#TimestepEmbedder-Module) | Timestep embedder of denoiser             |
| [MelUpsampler](#MelUpsampler-Module)         | Upsampler for conditional Mel spectrogram |
| [ResidualLayer](#ResidualLayer-Module)       | Individual layers of denoiser             |
| [DiffWave](#DiffWave-Module)                 | Model                                     |


*The entire model implementation is detailed below in what is believed to be the most intuitive order...*

---

### DiffWaveDenoiser Module

```python
class _DiffWaveDenoiser(nn.Module):
    def __init__(self,
        in_channels,
        residual_channels,
        residual_layers,
        residual_blocks,
        max_denoising_steps,
        is_conditional,
        mel_bands,
    ):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, residual_channels, 1)
        self.timestep_embedder = _TimestepEmbedder(max_denoising_steps)
        self.spectrogram_upsampler = _MelUpsampler()
        self.res_blocks = nn.ModuleList([
            nn.ModuleList([
                _ResidualLayer(residual_channels, 2**i, mel_bands) if is_conditional
                else _ResidualLayer(residual_channels, 2**i, None)
                for i in range(residual_layers // residual_blocks)
            ])
            for _ in range(residual_blocks)
        ])
        self.out_conv1 = nn.Conv1d(residual_channels, residual_channels, 1)
        self.out_conv2 = nn.Conv1d(residual_channels, in_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.relu(x)
        t = self.timestep_embedder(t)
        if mel is not None: mel = self.spectrogram_upsampler(mel)
        skips = None
        for block in self.res_blocks:
            for layer in block:
                x, skip = layer(x, t, mel)
                skips = skip if skips is None else skips + skip
        out = self.out_conv1(skips)
        out = self.relu(out)
        out = self.out_conv2(out)
        return out
```

#### Class Attributes Notes
- **self.res_blocks:** The residual layers instantiated are divided evenly between the residual blocks. This is because
  all residual blocks have the same dilation cycle in the paper (e.g., if there are $n$ residual layers and $m$
  residual blocks, then there are **$\frac{n}{m}$** residual layers per block, each with a dilation cycle of
  $[2^0, 2^1, ..., 2^{\frac{n}{m}-1}]$). The paper also states that the receptive field of the output can be calculated
  as: $r = (k-1) \sum_i d_i + 1$, where $k$ is the kernel size of the dilated convolutions in the residual layers, and
  $d_i$ is the dilation rate at the $i$-th residual layer.

#### Forward Pass Shapes
| Variable | Initial Shape              | Final Shape                |
|----------|----------------------------|----------------------------|
| x        | $(B, C_{in}, T_{sample})$  | $(B, C_{res}, T_{sample})$ |
| t        | $(B,)$                     | $(B, 512)$                 |
| mel      | $(B, N, T_{spec})$         | $(B, N, T_{sample})$       | 
| skips    | $(B, C_{res}, T_{sample})$ | $(B, C_{res}, T_{sample})$ |
| skip     | $(B, C_{res}, T_{sample})$ | $(B, C_{res}, T_{sample})$ |
| out      | $(B, C_{res}, T_{sample})$ | $(B, C_{in}, T_{sample})$  |

---

### TimestepEmbedder Module

```python
class _TimestepEmbedder(nn.Module):
    def __init__(self, max_denoising_steps):
        super().__init__()
        self.register_buffer(
            "timestep_embeddings",
            self._get_timestep_embeddings(max_denoising_steps)
        )
        self.linear1 = nn.Linear(128, 512)
        self.linear2 = nn.Linear(512, 512)
        self.swish = nn.SiLU()

    @staticmethod
    def _get_timestep_embeddings(max_denoising_steps):
        channel_wise = torch.arange(64).unsqueeze(0)
        time_wise = torch.arange(1, max_denoising_steps + 1).unsqueeze(1)
        sinusoid_inputs = 10**(channel_wise * 4 / 63) * time_wise
        return torch.cat((sinusoid_inputs.sin(), sinusoid_inputs.cos()), 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = self.timestep_embeddings[t]
        t = self.linear1(t)
        t = self.swish(t)
        t = self.linear2(t)
        t = self.swish(t)
        return t
```

#### Class Attributes Notes
- #### self.timestep_embeddings:
  $t_{\text{embedding}} = \left[
  \sin\left(10^{\frac{0 \times 4}{63}} t\right), \dots,
  \sin\left(10^{\frac{63 \times 4}{63}} t\right),
  \cos\left(10^{\frac{0 \times 4}{63}} t\right), \dots,
  \cos\left(10^{\frac{63 \times 4}{63}} t\right)
  \right]$

#### Forward Pass Shapes
| Variable | Initial Shape | Final Shape |
|----------|---------------|-------------|
| t        | $(B, )$       | $(B, 512)$  |

---

### MelUpsampler Module

```python
class _MelUpsampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.convt1 = nn.ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16),
                                          padding=(1, 12))
        self.convt2 = nn.ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16),
                                          padding=(1, 8))
        self.leaky_relu = nn.LeakyReLU(0.4)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.unsqueeze(1)
        mel = self.convt1(mel)
        mel = self.leaky_relu(mel)
        mel = self.convt2(mel)
        mel = self.leaky_relu(mel)
        mel = mel.squeeze(1)
        return mel
```

#### Class Attributes Notes
- **self.convt1, self.convt2:** Expands the input Mel spectrogram's time frame dimension $T_{spec}$ to the total
  number of samples in the waveform $T_{sample}$.

#### Forward Pass Shapes
| Variable | Initial Shape      | Final Shape            |
|----------|--------------------|------------------------|
| mel      | $(B, N, T_{spec})$ | $(B, N, T_{sample})$   |

---

### ResidualLayer Module

```python
class _ResidualLayer(nn.Module):
    def __init__(self, residual_channels, dilation, mel_bands):
        super().__init__()
        self.timestep_linear = nn.Linear(512, residual_channels)
        self.bi_dilated_conv = nn.Conv1d(residual_channels, 2*residual_channels, 3, 1, dilation,
                                         dilation)
        self.mel_conv = None if mel_bands is None else nn.Conv1d(mel_bands, 2*residual_channels, 1)
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, t: torch.Tensor, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = x + self.timestep_linear(t).unsqueeze(2)
        x = self.bi_dilated_conv(x) if self.mel_conv is None else self.bi_dilated_conv(x) + self.mel_conv(mel)
        gates, filters = x.chunk(2, 1)
        x = self.sigmoid(gates) * self.tanh(filters)
        return self.res_conv(x) + residual, self.skip_conv(x)
```

#### Class Attributes Notes
- **self.bi_dilated_conv:** Expands the channel dimensions of **x** from $C_{res}$ to $2\cdot{C_{res}}$, and dilates
  according to the residual layer's part of the dilation cycle in the corresponding residual block in order to increase
  its receptive field.

#### Forward Pass Shapes
| Variable | Initial Shape              | Final Shape                |
|----------|----------------------------|----------------------------|
| x        | $(B, C_{res}, T_{sample})$ | $(B, C_{res}, T_{sample})$ |
| residual | $(B, C_{res}, T_{sample})$ | $(B, C_{res}, T_{sample})$ |
| t        | $(B, 512)$                 | $(B, 512)$                 |
| mel      | $(B, N, T_{sample})$       | $(B, N, T_{sample})$       |
| gates    | $(B, C_{res}, T_{sample})$ | $(B, C_{res}, T_{sample})$ |
| filters  | $(B, C_{res}, T_{sample})$ | $(B, C_{res}, T_{sample})$ |

---

### DiffWave Module

```python
class DiffWave(nn.Module):
    def __init__(self,
        scheduler: str,
        denoising_steps: int,
        in_channels: int,
        residual_channels: int,
        residual_layers: int,
        residual_blocks: int,
        max_denoising_steps: int,
        is_conditional: bool,
        mel_bands: int = None,
    ):
        super().__init__()
        if residual_layers % residual_blocks != 0:
            raise ValueError("`residual_layers` must be evenly divisible by `residual_blocks`.")
        if is_conditional and not isinstance(mel_bands, int):
            raise TypeError("`mel_bands` must be an integer.")
        self.is_conditional = is_conditional
        self.diffwave = DDPM(
            denoiser=_DiffWaveDenoiser(
                in_channels,
                residual_channels,
                residual_layers,
                residual_blocks,
                max_denoising_steps,
                is_conditional,
                mel_bands,
            ),
            scheduler=scheduler,
            denoising_steps=denoising_steps,
        )

    def noise(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.diffwave.noise(x, t)

    def denoise(self, x: torch.Tensor, t: torch.Tensor, mel: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_conditional:
            if not isinstance(mel, torch.Tensor):
                raise TypeError("`mel` must be torch.Tensor when `is_conditional` is True.")
        else:
            if mel is not None:
                raise ValueError("`mel` must be None when `is_conditional` is False.")
        return self.diffwave.denoise(x, t, mel)
```

#### Class Attributes Notes
- **self.ddpm:** An instance of DDPM with [DiffWaveDenoiser](#DiffWaveDenoiser-Module) as the denoiser.

#### Noise Pass Shapes
| Variable  | Initial Shape             | Final Shape               |
|-----------|---------------------------|---------------------------|
| x         | $(B, C_{in}, T_{sample})$ | $(B, C_{in}, T_{sample})$ |
| t         | $(B,)$                    | $(B,)$                    |

#### Denoise Pass Shapes
| Variable | Initial Shape               | Final Shape               |
|----------|-----------------------------|---------------------------|
| x        | $(B, C_{in}, T_{sample})$   | $(B, C_{in}, T_{sample})$ |
| t        | $(B,)$                      | $(B,)$                    |
| mel      | $(B, N, T_{spec})$          | $(B, N, T_{spec})$        |

---

### All Modules
```python
import torch
from torch import nn
from DDPM.model import DDPM


class _TimestepEmbedder(nn.Module):
    def __init__(self, max_denoising_steps):
        super().__init__()
        self.register_buffer(
            "timestep_embeddings",
            self._get_timestep_embeddings(max_denoising_steps)
        )
        self.linear1 = nn.Linear(128, 512)
        self.linear2 = nn.Linear(512, 512)
        self.swish = nn.SiLU()

    @staticmethod
    def _get_timestep_embeddings(max_denoising_steps):
        channel_wise = torch.arange(64).unsqueeze(0)
        time_wise = torch.arange(1, max_denoising_steps + 1).unsqueeze(1)
        sinusoid_inputs = 10**(channel_wise * 4 / 63) * time_wise
        return torch.cat((sinusoid_inputs.sin(), sinusoid_inputs.cos()), 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = self.timestep_embeddings[t]
        t = self.linear1(t)
        t = self.swish(t)
        t = self.linear2(t)
        t = self.swish(t)
        return t


class _MelUpsampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.convt1 = nn.ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16),
                                          padding=(1, 12))
        self.convt2 = nn.ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16),
                                          padding=(1, 8))
        self.leaky_relu = nn.LeakyReLU(0.4)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.unsqueeze(1)
        mel = self.convt1(mel)
        mel = self.leaky_relu(mel)
        mel = self.convt2(mel)
        mel = self.leaky_relu(mel)
        mel = mel.squeeze(1)
        return mel


class _ResidualLayer(nn.Module):
    def __init__(self, residual_channels, dilation, mel_bands):
        super().__init__()
        self.timestep_linear = nn.Linear(512, residual_channels)
        self.bi_dilated_conv = nn.Conv1d(residual_channels, 2*residual_channels, 3, 1, dilation,
                                         dilation)
        self.mel_conv = None if mel_bands is None else nn.Conv1d(mel_bands, 2*residual_channels, 1)
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, t: torch.Tensor, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = x + self.timestep_linear(t).unsqueeze(2)
        x = self.bi_dilated_conv(x) if self.mel_conv is None else self.bi_dilated_conv(x) + self.mel_conv(mel)
        gates, filters = x.chunk(2, 1)
        x = self.sigmoid(gates) * self.tanh(filters)
        return self.res_conv(x) + residual, self.skip_conv(x)


class _DiffWaveDenoiser(nn.Module):
    def __init__(self,
        in_channels,
        residual_channels,
        residual_layers,
        residual_blocks,
        max_denoising_steps,
        is_conditional,
        mel_bands,
    ):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, residual_channels, 1)
        self.timestep_embedder = _TimestepEmbedder(max_denoising_steps)
        self.spectrogram_upsampler = _MelUpsampler()
        self.res_blocks = nn.ModuleList([
            nn.ModuleList([
                _ResidualLayer(residual_channels, 2**i, mel_bands) if is_conditional
                else _ResidualLayer(residual_channels, 2**i, None)
                for i in range(residual_layers // residual_blocks)
            ])
            for _ in range(residual_blocks)
        ])
        self.out_conv1 = nn.Conv1d(residual_channels, residual_channels, 1)
        self.out_conv2 = nn.Conv1d(residual_channels, in_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.relu(x)
        t = self.timestep_embedder(t)
        if mel is not None: mel = self.spectrogram_upsampler(mel)
        skips = None
        for block in self.res_blocks:
            for layer in block:
                x, skip = layer(x, t, mel)
                skips = skip if skips is None else skips + skip
        out = self.out_conv1(skips)
        out = self.relu(out)
        out = self.out_conv2(out)
        return out


class DiffWave(nn.Module):
    def __init__(self,
        scheduler: str,
        denoising_steps: int,
        in_channels: int,
        residual_channels: int,
        residual_layers: int,
        residual_blocks: int,
        max_denoising_steps: int,
        is_conditional: bool,
        mel_bands: int = None,
    ):
        super().__init__()
        if residual_layers % residual_blocks != 0:
            raise ValueError("`residual_layers` must be evenly divisible by `residual_blocks`.")
        if is_conditional and not isinstance(mel_bands, int):
            raise TypeError("`mel_bands` must be an integer.")
        self.is_conditional = is_conditional
        self.diffwave = DDPM(
            denoiser=_DiffWaveDenoiser(
                in_channels,
                residual_channels,
                residual_layers,
                residual_blocks,
                max_denoising_steps,
                is_conditional,
                mel_bands,
            ),
            scheduler=scheduler,
            denoising_steps=denoising_steps,
        )

    def noise(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.diffwave.noise(x, t)

    def denoise(self, x: torch.Tensor, t: torch.Tensor, mel: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_conditional:
            if not isinstance(mel, torch.Tensor):
                raise TypeError("`mel` must be torch.Tensor when `is_conditional` is True.")
        else:
            if mel is not None:
                raise ValueError("`mel` must be None when `is_conditional` is False.")
        return self.diffwave.denoise(x, t, mel)
```

---
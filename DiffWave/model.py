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
        scheduler: str = 'linear_beta',
        denoising_steps: int = 50,
        in_channels: int = 1,
        residual_channels: int = 64,
        residual_layers: int = 30,
        residual_blocks: int = 3,
        is_conditional: bool = True,
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
                denoising_steps,
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
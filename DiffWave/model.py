import torch
from torch import nn


class TimestepEmbedder(nn.Module):
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


class ResidualLayer(nn.Module):
    def __init__(self, residual_channels, dilation):
        super().__init__()
        self.timestep_linear = nn.Linear(512, residual_channels)
        self.bi_dilated_conv = nn.Conv1d(residual_channels, 2*residual_channels, 3, 1, dilation, dilation)
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, 1, 1)
        self.skip_conv = nn.Conv1d(residual_channels, residual_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        x = x + self.timestep_linear(t).unsqueeze(2)
        x = self.bi_dilated_conv(x)
        gates, filters = x.chunk(2, 1)
        x = self.sigmoid(gates) * self.tanh(filters)
        return self.res_conv(x) + residual, self.skip_conv(x)


class DiffWaveDenoiser(nn.Module):
    def __init__(self,
        in_channels: int,
        residual_channels: int,
        residual_blocks: int,
        residual_layers: int,
        max_denoising_steps: int
    ):
        super().__init__()
        assert residual_layers % residual_blocks == 0, "residual_layers must be evenly divisible by residual_blocks"
        self.in_conv = nn.Conv1d(in_channels, residual_channels, 1)
        self.timestep_embedder = TimestepEmbedder(max_denoising_steps)
        self.res_blocks = nn.ModuleList([
            nn.ModuleList([
                ResidualLayer(residual_channels, 2**i)
                for i in range(residual_layers // residual_blocks)
            ])
            for _ in range(residual_blocks)
        ])
        self.out_conv_1 = nn.Conv1d(residual_channels, residual_channels, 1)
        self.out_conv_2 = nn.Conv1d(residual_channels, in_channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        x = self.relu(x)
        t = self.timestep_embedder(t)
        skips = None
        for block in self.res_blocks:
            for layer in block:
                x, skip = layer(x, t)
                skips = skip if skips is None else skips + skip
        out = self.out_conv_1(skips)
        out = self.relu(out)
        out = self.out_conv_2(out)
        return out
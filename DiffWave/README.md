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

## [Full Implementation](model.py)

#### Dimension Symbols
| $\text{Symbol}$ | $\text{Description}$                                     |
|-----------------|----------------------------------------------------------|
| $B$             | $\text{Batch size}$                                      |
| $C_{in}$        | $\text{Number of input channels}$                        |
| $C_{res}$       | $\text{Number of residual channels}$                     |
| $N$             | $\text{Number of mel frequency bins in conditional Mel}$ |
| $T_{spec}$      | $\text{Total time frames in conditional Mel}$            |
| $T_{samples}$   | $\text{Total samples in the waveform}$                   |

#### Custom Modules
| $\text{Module}$                                       | $\text{Description}$                               |
|-------------------------------------------------------|----------------------------------------------------|
| [$\text{DiffWaveDenoiser}$](#DiffWaveDenoiser-Module) | $\text{Model denoiser}$                            |
| [$\text{TimestepEmbedder}$](#TimestepEmbedder-Module) | $\text{Timestep embedder of denoiser}$             |
| MelUpsampler                                          | $\text{Upsampler for conditional Mel Spectrogram}$ |
| ResidualLayer                                         | $\text{Individual layers of denoiser}$             |
| DiffWave                                              | $\text{Entire model}$                              |


*The entire model implementation will be explained below, and in the order that I believe is most intuitive...*

---

### DiffWaveDenoiser Module

#### Full Denoiser Code:
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

#### Class Attributes Notes:
- **self.res_blocks:** The residual layers instantiated are divided evenly between the residual blocks. This is because
  all residual blocks have the same dilation cycle in the paper (e.g., if there are $n$ residual layers and $m$
  residual blocks, then there are **$\frac{n}{m}$** residual layers per block, each with a dilation cycle of
  $[2^0, 2^1, ..., 2^{\frac{n}{m}-1}]$). The paper also states that the receptive field of the output can be calculated
  as: $r = (k-1) \sum_i d_i + 1$, where $k$ is the kernel size of the dilated convolutions in the residual layers, and
  $d_i$ is the dilation rate at the $i$-th residual layer.

#### Forward Pass Notes:
| Variable | Initial Shape               | Final Shape                                                                                                                                                                |
|----------|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| $x$      | $(B, C_{in}, T_{samples})$  | $(B, C_{res}, T_{samples})$                                                                                                                                                |
| $t$      | $(B,)$                      | $(B, 512)$                                                                                                                                                                 |
| $mel$    | $(B, N) $                   | $$\text{mel} = \begin{cases} \text{some value or expression}, & \text{if } \text{mel is None} \\ \text{self.spectrogram\_upsampler(mel)}, & \text{otherwise} \end{cases}$$ |
| $skips$  | $(B, C_{res}, T_{samples})$ | `(batch_size, C_hidden, H, W)` (aggregated skips)                                                                                                                          |
| $skip$   | $(B, C_{res}, T_{samples})$ | `(batch_size, C_block, H, W)`                                                                                                                                              |
| $out$    | $(B, C_{in}, T_{samples})$  | `(batch_size, C_out, H, W)`                                                                                                                                                |


---

### TimestepEmbedder Module

#### Full Timestep Embedder Code:
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

#### Class Attributes Notes:
- #### self.timestep_embeddings:
  $t_{\text{embedding}} = \left[
  \sin\left(10^{\frac{0 \times 4}{63}} t\right), \dots,
  \sin\left(10^{\frac{63 \times 4}{63}} t\right),
  \cos\left(10^{\frac{0 \times 4}{63}} t\right), \dots,
  \cos\left(10^{\frac{63 \times 4}{63}} t\right)
  \right]$

#### Forward Pass Notes:
1. $t$ is the $(B, )$ tensor that holds the corresponding timestep of each sample in the batch. The output
   $(B, 512)$ tensor consists of the corresponding final timestep embeddings for the batch:
   ```python
   def forward(self, t: torch.Tensor) -> torch.Tensor:
   ```
2. Each timestep in $t$ is first converted to their corresponding $(128, )$ initial timestep embeddings,
   transforming $t$ from $(B, )$ to $(B, 128)$:
   ```python
       t = self.timestep_embeddings[t]
   ```
3. The initial timestep embeddings are then passed through two fully-connected layers and a Swish with
   $\beta = 1$ (SiLU) to get the final embeddings, transforming $t$ from $(B, 128)$ to $(B, 512)$:
   ```python
       t = self.linear1(t)
       t = self.swish(t)
       t = self.linear2(t)
       t = self.swish(t)
       return t
   ```
   
---
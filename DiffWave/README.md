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
| Symbol    | Description                 |
|-----------|-----------------------------|
| $B$       | Batch size                  |
| $C_{in}$  | Number of input channels    |
| $C_{res}$ | Number of residual channels |
| $D$       | Dimensionality of the data  |

#### This implementation has 4 custom modules:
- **[DiffWaveDenoiser](#DiffWaveDenoiser-Module):** Encapsulates all denoiser components.
- **[TimestepEmbedder](#TimestepEmbedder-Module):** The timestep embedder of the denoiser.
- **ResidualLayer**: The individual layers of the denoiser.
- **DiffWave:** Encapsulates the entire model, including the noising and denoising processes.

*The entire model implementation will be explained below, and in the order that I believe is most intuitive...*

---

### DiffWaveDenoiser Module

#### Full Denoiser Code:
```python
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
```

#### Class Attributes Notes:
- **self.res_blocks:** The residual layers instantiated are divided evenly between the residual blocks. This is because
  all residual blocks have the same dilation cycle in the paper (e.g., if there are $n$ residual layers and $m$
  residual blocks, then there are **$\frac{n}{m}$** residual layers per block, each with a dilation cycle of
  $[2^0, 2^1, ..., 2^{\frac{n}{m}-1}]$). The paper also states that the receptive field of the output can be calculated
  as: $r = (k-1) \sum_i d_i + 1$, where $k$ is the kernel size of the dilated convolutions in the residual layers, and
  $d_i$ is the dilation rate at the $i$-th residual layer.

#### Forward Pass Notes:
1. $x$ is the $(B, C_{in}, D)$ waveform tensor at timestep $t$ of the denoising process.
   $t$ is the $(B, )$ tensor that holds the corresponding timestep of each sample in the batch. The output will be
   the predicted noise $\hat{\epsilon}$ in $x$:
    ```python
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    ```
2. The forward pass of the denoiser begins with a convolution and ReLU. The convolution increases the input
   channels $C_{in}$ to the desired number of residual channels to be used $C_{res}$:
    ```python
        x = self.in_conv(x)
        x = self.relu(x)
    ```
3. Each timestep in $t$ is then embedded as a $(512, )$ tensor via the
   [timestep embedder](#TimestepEmbedder-Module), transforming $t$ from $(B, )$ to $(B, 512)$:
   ```python
       t = self.timestep_embedder(t)
   ```

4. $x$ and $t$ are then passed through every residual layer of every residual block. $t$ stays the same for
   each layer. The skip connections $skip$ are additively accumulated into $skips$:
    ```python
        skips = None
        for block in self.res_blocks:
            for layer in block:
                x, skip = layer(x, t)
                skips = skip if skips is None else skips + skip
    ```

5. Once $x$ has passed through all residual layers, two convolutions and a ReLU transform $skips$
   from $(B, C_{res}, D)$ to the initial $x$ dimensions $(B, C_{in}, D)$:
    ```python
        out = self.out_conv_1(skips)
        out = self.relu(out)
        out = self.out_conv_2(out)
        return out
    ```

---

### TimestepEmbedder Module

#### Full Timestep Embedder Code:
```python
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
# [DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://openreview.net/pdf?id=a-xFK8Ymz5J)

### Model Description
A non-autoregressive denoising diffusion probabilistic model for waveform generation directly in the
time domain, supporting both conditional and unconditional audio synthesis.

## [Full Implementation](model.py)

#### Custom Classes
| Classes                                      | Description                                   |
|----------------------------------------------|-----------------------------------------------|
| [DiffWaveDenoiser](#DiffWaveDenoiser-Module) | Model denoiser                                |
| [TimestepEmbedder](#TimestepEmbedder-Module) | Timestep embedder of denoiser                 |
| [MelUpsampler](#MelUpsampler-Module)         | Upsampler for conditional Mel spectrogram     |
| [ResidualLayer](#ResidualLayer-Module)       | Individual layers of denoiser                 |
| [DiffWave](#DiffWave-Module)                 | Model                                         |

#### Dimension Symbols
| Symbol          | Description                                                 |
|-----------------|-------------------------------------------------------------|
| $B$             | Batch size                                                  |
| $C_{in}$        | Number of input channels                                    |
| $C_{res}$       | Number of residual channels                                 |
| $N$             | Number of mel frequency bins in conditional Mel spectrogram |
| $T_{mel}$       | Total time frames in conditional Mel spectrogram            |
| $T_{sample}$    | Total samples in waveform                                   |

*The entire model implementation is detailed below in what is believed to be the most intuitive order...*

---

### DiffWaveDenoiser Module

```python
class _DiffWaveDenoiser(nn.Module):
    """DiffWave model denoiser.

    Attributes:
        in_conv (nn.Conv1d): Input convolution layer.
        timestep_embedder (_TimestepEmbedder): Denoising step embedding module.
        mel_upsampler (_MelUpsampler): Mel upsampler module.
        res_blocks (nn.ModuleList): Blocks or residual layers.
        out_conv1 (nn.Conv1d): First output convolution layer.
        out_conv2 (nn.Conv1d): Second output convolution layer.
        relu (nn.ReLU): ReLU activation function.
    """
    def __init__(self,
        in_channels: int,
        residual_channels: int,
        residual_layers: int,
        residual_blocks: int,
        max_denoising_steps: int,
        is_conditional: bool,
        mel_bands: int | None,
    ):
        """Initializes the DiffWaveDenoiser module.

        Args:
            in_channels (int): Number of input channels.
            residual_channels (int): Number of residual channels.
            residual_layers (int): Number of residual layers.
            residual_blocks (int): Number of residual blocks.
            max_denoising_steps (int): Maximum number of denoising diffusion steps.
            is_conditional (bool): Whether the model is conditional or not.
            mel_bands (int): Number of mel bands.
        """
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, residual_channels, 1)
        self.timestep_embedder = _TimestepEmbedder(max_denoising_steps)
        self.mel_upsampler = _MelUpsampler()
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
        """Applies the denoiser.

        Args:
            x (torch.Tensor): Input waveform.
            t (torch.Tensor): Current timestep.
            mel (torch.Tensor): Mel spectrogram tensor to be upsampled.

        Returns:
            torch.Tensor: Predicted noise in input waveform.
        """
        x = self.in_conv(x)                                         # (B, C_res, T_sample)
        x = self.relu(x)                                            # (B, C_res, T_sample)
        t = self.timestep_embedder(t)                               # (B, 512)
        if mel is not None: mel = self.mel_upsampler(mel)           # (B, N, T_sample)
        skips = None
        for block in self.res_blocks:  # type: Iterable[nn.Module]
            for layer in block:
                x, skip = layer(x, t, mel)                          # (B, C_res, T_sample), (B, C_res, T_sample)
                skips = skip if skips is None else skips + skip     # (B, C_res, T_sample)
        out = self.out_conv1(skips)                                 # (B, C_res, T_sample)
        out = self.relu(out)                                        # (B, C_res, T_sample)
        out = self.out_conv2(out)                                   # (B, C_in, T_sample)
        return out                                                  # (B, C_in, T_sample)
```

---

### TimestepEmbedder Module

```python
class _TimestepEmbedder(nn.Module):
    """Embeds the timestep to inform the model of the current diffusion step.

    Attributes:
        timestep_embeddings (torch.Tensor): Initial timestep embedding tensor.
        linear1 (nn.Linear): Linear layer.
        linear2 (nn.Linear): Linear layer.
        swish (nn.SiLU): SiLU activation function.
    """
    def __init__(self, max_denoising_steps: int):
        """Initializes the TimestepEmbedder module.

        Args:
            max_denoising_steps (int): Maximum number of denoising diffusion steps.
        """
        super().__init__()
        self.register_buffer(
            "timestep_embeddings",
            self._get_timestep_embeddings(max_denoising_steps)
        )
        self.linear1 = nn.Linear(128, 512)
        self.linear2 = nn.Linear(512, 512)
        self.swish = nn.SiLU()

    @staticmethod
    def _get_timestep_embeddings(max_denoising_steps: int):
        """Creates the timestep embedding tensor.

        Args:
            max_denoising_steps (int): Maximum number of denoising diffusion steps.

        Returns:
            torch.Tensor: Initial timestep embedding tensor.
        """
        channel_wise = torch.arange(64).unsqueeze(0)
        time_wise = torch.arange(1, max_denoising_steps + 1).unsqueeze(1)
        sinusoid_inputs = 10**(channel_wise * 4 / 63) * time_wise
        return torch.cat((sinusoid_inputs.sin(), sinusoid_inputs.cos()), 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embeds the current diffusion timestep.

        Args:
            t (torch.Tensor): Current timestep.

        Returns:
            torch.Tensor: Final timestep embedding tensor.
        """
        t = self.timestep_embeddings[t]  # (B, 128)
        t = self.linear1(t)              # (B, 512)
        t = self.swish(t)                # (B, 512)
        t = self.linear2(t)              # (B, 512)
        t = self.swish(t)                # (B, 512)
        return t                         # (B, 512)
```

---

### MelUpsampler Module

```python
class _MelUpsampler(nn.Module):
    """Upsampler for mel spectrograms to align its time frames with the input waveform's sample count.

    Attributes:
        convt1 (nn.ConvTranspose2d): Transposed convolution to expand mel spectrogram time frame dimension.
        convt2 (nn.ConvTranspose2d): Transposed convolution to expand mel spectrogram time frame dimension.
        leaky_relu (nn.LeakyReLU): Leaky ReLU activation function.
    """
    def __init__(self):
        """Initializes the MelUpsampler module."""
        super().__init__()
        self.convt1 = nn.ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16),
                                          padding=(1, 12))
        self.convt2 = nn.ConvTranspose2d(1, 1, kernel_size=(3, 32), stride=(1, 16),
                                          padding=(1, 8))
        self.leaky_relu = nn.LeakyReLU(0.4)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Upsamples the input mel spectrogram.

        Args:
            mel (torch.Tensor): Mel spectrogram tensor to be upsampled.

        Returns:
            torch.Tensor: Upsampled mel spectrogram tensor.
        """
        mel = mel.unsqueeze(1)      # (B, 1, N, T_mel)
        mel = self.convt1(mel)      # (B, 1, N, 16 * T_mel - 8)
        mel = self.leaky_relu(mel)  # (B, 1, N, 16 * T_mel - 8)
        mel = self.convt2(mel)      # (B, 1, N, 256 * T_mel - 128)
        mel = self.leaky_relu(mel)  # (B, 1, N, 256 * T_mel - 128)
        mel = mel.squeeze(1)        # (B, N, 256 * T_mel - 128)
        return mel                  # (B, N, 256 * T_mel - 128)
```

---

### ResidualLayer Module

```python
class _ResidualLayer(nn.Module):
    """The residual layers of the model.

    Attributes:
        timestep_linear (nn.Linear): Timestep linear layer.
        bi_dilated_conv (nn.Conv1d): Bi-directional dilated convolution layer.
        mel_conv (nn.Conv1d | None): Mel spectrogram convolution layer.
        res_conv (nn.Conv1d): Residual convolution layer.
        skip_conv (nn.Conv1d): Skip convolution layer.
        sigmoid (nn.Sigmoid): Sigmoid activation function.
        tanh (nn.Tanh): Tanh activation function.
    """
    def __init__(self, residual_channels: int, dilation: int, mel_bands: int | None):
        """Initializes the ResidualLayer module.

        Args:
            residual_channels (int): Number of residual channels.
            dilation (int): Dilation rate of layer.
            mel_bands (int | None): Number of mel bands.
        """
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
        """Applies the residual layer.

        Args:
            x (torch.Tensor): Input waveform.
            t (torch.Tensor): Current timestep.
            mel (torch.Tensor): Mel spectrogram tensor to be upsampled.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - Input waveform after residual processing.
                - Skip connection tensor.
        """
        residual = x                                           # (B, C_res, T_sample)
        x = x + self.timestep_linear(t).unsqueeze(2)           # (B, C_res, T_sample)
        x = (self.bi_dilated_conv(x) if self.mel_conv is None
            else self.bi_dilated_conv(x) + self.mel_conv(mel)) # (B, 2 * C_res, T_sample)
        gates, filters = x.chunk(2, 1)                         # (B, C_res, T_sample), (B, C_res, T_sample)
        x = self.sigmoid(gates) * self.tanh(filters)           # (B, C_res, T_sample)
        return self.res_conv(x) + residual, self.skip_conv(x)  # (B, C_res, T_sample), (B, C_res, T_sample)
```

---

### DiffWave Module

```python
class DiffWave(nn.Module):
    """The DiffWave model.

    Attributes:
        is_conditional (bool): Whether the model is conditional or not.
        diffwave (DDPM): Diffwave model.
    """
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
        """Initialize the DiffWave module.

        This initializer sets up a denoising diffusion probabilistic model (DDPM)
        using a DiffWave-style denoiser. The model can optionally be conditioned
        on mel spectrogram features if `is_conditional` is True.

        Args:
            scheduler (str, optional): The scheduler type to use for the diffusion process.
            denoising_steps (int, optional): The number of denoising steps in the diffusion process.
            in_channels (int, optional): Number of input channels.
            residual_channels (int, optional): Number of channels used in the residual blocks.
            residual_layers (int, optional): Total number of residual layers in the model.
            residual_blocks (int, optional): Number of residual blocks in the model.
            is_conditional (bool, optional): Whether the model is conditioned on mel spectrogram features.
            mel_bands (int, optional): Number of mel frequency bands for conditioning.

        Raises:
            ValueError: If `residual_layers` is not evenly divisible by `residual_blocks`.
            TypeError: If `is_conditional` is True and `mel_bands` is not provided as an integer.
        """
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
        """Add noise to the input signal.

        Applies the noise scheduler of the DiffWave model to the input waveform,
        generating a noised signal along with the corresponding noise that was applied.

        Args:
            x (torch.Tensor): Input waveform tensor.
            t (torch.Tensor): Current diffusion timestep tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The noised signal tensor.
                - The noise tensor used to generate the noised signal.
        """
        return self.diffwave.noise(x, t)  # (B, C_in, T_sample), (B, C_in, T_sample)

    def denoise(self, x: torch.Tensor, t: torch.Tensor, mel: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoise the input signal using the predicted noise.

        Reconstructs the denoised signal from the noised input by utilizing the predicted
        noise. If the model is conditional, a mel spectrogram is used as an additional
        conditioning input.

        Args:
            x (torch.Tensor): Noised input waveform tensor.
            t (torch.Tensor): Current diffusion timestep tensor.
            mel (torch.Tensor, optional): Mel spectrogram tensor for conditioning.
                Required if `is_conditional` is True. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The denoised signal tensor.
                - The predicted noise tensor.

        Raises:
            TypeError: If the model is conditional and `mel` is not a torch.Tensor.
            ValueError: If the model is not conditional and `mel` is provided.
        """
        if self.is_conditional:
            if not isinstance(mel, torch.Tensor):
                raise TypeError("`mel` must be torch.Tensor when `is_conditional` is True.")
        else:
            if mel is not None:
                raise ValueError("`mel` must be None when `is_conditional` is False.")
        return self.diffwave.denoise(x, t, mel)  # (B, C_in, T_sample), (B, C_in, T_sample)
```

---

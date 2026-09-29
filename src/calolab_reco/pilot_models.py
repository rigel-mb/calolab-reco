"""Small models for timing and resume checks, before the reconstruction study.

Inputs are nonnegative log-transformed deposits in stored (iphi, ieta) order.
Fixed index features describe array locations, not physical crystal centers.
No incident-energy or position target enters any model's forward computation.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

GRID_SHAPE = (30, 85)
PATCH_SHAPE = (5, 5)
PATCH_GRID = (6, 17)
N_PATCHES = 102
PATCH_VALUES = 25
ENCODER_WIDTH = 64


def _check_images(x: Tensor) -> None:
    if x.ndim != 4 or tuple(x.shape[1:]) != (1, *GRID_SHAPE):
        raise ValueError("Expected deposits with shape (batch, 1, 30, 85)")


def patchify(x: Tensor) -> Tensor:
    """Return 102 row-major patches, each retaining 25 ordered cell values."""
    _check_images(x)
    return (
        x.reshape(x.shape[0], 1, 6, 5, 17, 5)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(x.shape[0], N_PATCHES, PATCH_VALUES)
    )


def unpatchify(patches: Tensor) -> Tensor:
    """Restore the stored grid without cropping, padding or reordering cells."""
    if patches.ndim != 3 or tuple(patches.shape[1:]) != (N_PATCHES, PATCH_VALUES):
        raise ValueError("Expected patches with shape (batch, 102, 25)")
    return (
        patches.reshape(patches.shape[0], 6, 17, 1, 5, 5)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(patches.shape[0], 1, *GRID_SHAPE)
    )


def _index_channels() -> Tensor:
    row, column = torch.meshgrid(
        torch.linspace(-1, 1, GRID_SHAPE[0]),
        torch.linspace(-1, 1, GRID_SHAPE[1]),
        indexing="ij",
    )
    return torch.stack((row, column), dim=0).unsqueeze(0)


def _sincos_positions(width: int) -> Tensor:
    """Encode patch row/column indices; positions are not physical coordinates."""
    if width % 4:
        raise ValueError("Position-encoding width must be divisible by four")
    row, column = torch.meshgrid(
        torch.arange(PATCH_GRID[0], dtype=torch.float32),
        torch.arange(PATCH_GRID[1], dtype=torch.float32),
        indexing="ij",
    )
    frequency = torch.exp(-math.log(10000.0) * torch.arange(width // 4) / (width // 4))
    angles_row = row.reshape(-1, 1) * frequency
    angles_column = column.reshape(-1, 1) * frequency
    return torch.cat(
        (angles_row.sin(), angles_row.cos(), angles_column.sin(), angles_column.cos()), dim=1
    ).unsqueeze(0)


def _transformer_stack(width: int, depth: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=4,
        dim_feedforward=width * 2,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    stack = nn.TransformerEncoder(
        layer, num_layers=depth, norm=nn.LayerNorm(width), enable_nested_tensor=False
    )
    # TransformerEncoder clones its prototype layer. Initialize each block separately.
    for block in stack.layers:
        nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
        nn.init.zeros_(block.self_attn.in_proj_bias)
        for module in block.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    return stack


class RegressionHeads(nn.Module):
    """Predict standardized energy, local iphi and local ieta, in this order."""

    def __init__(self) -> None:
        super().__init__()
        self.energy = nn.Linear(ENCODER_WIDTH, 1)
        self.position = nn.Linear(ENCODER_WIDTH, 2)

    def forward(self, features: Tensor) -> Tensor:
        return torch.cat((self.energy(features), self.position(features)), dim=1)


class CNN(nn.Module):
    """Three convolutions with deposits and two fixed grid-index channels."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("index_channels", _index_channels())
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GELU(),
            # Cover the full 8 x 22 grid without adaptive pooling's CUDA backward.
            nn.AvgPool2d(kernel_size=(4, 6), stride=(2, 4)),
            nn.Flatten(),
            nn.Linear(32 * 3 * 5, ENCODER_WIDTH),
            nn.GELU(),
        )
        self.heads = RegressionHeads()

    def forward(self, x: Tensor) -> Tensor:
        _check_images(x)
        coordinates = self.index_channels.to(dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        return self.heads(self.encoder(torch.cat((x, coordinates), dim=1)))


class PatchEncoder(nn.Module):
    """Common encoder for direct regression and masked pretraining.

    Ordered patch values and fixed patch positions together identify every cell
    on the same grid available to the CNN. No padding is needed for 30 x 85.
    """

    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Linear(PATCH_VALUES, ENCODER_WIDTH)
        self.register_buffer("positions", _sincos_positions(ENCODER_WIDTH))
        self.blocks = _transformer_stack(ENCODER_WIDTH, depth=3)

    def forward(self, x: Tensor, visible_indices: Tensor | None = None) -> Tensor:
        tokens = self.patch_embedding(patchify(x))
        tokens = tokens + self.positions.to(dtype=tokens.dtype)
        if visible_indices is not None:
            tokens = tokens.gather(1, visible_indices.unsqueeze(-1).expand(-1, -1, ENCODER_WIDTH))
        return self.blocks(tokens)


class Transformer(nn.Module):
    """Direct multitask regression using the shared patch encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = PatchEncoder()
        self.heads = RegressionHeads()

    def forward(self, x: Tensor) -> Tensor:
        return self.heads(self.encoder(x).mean(dim=1))


class MaskedAutoencoder(nn.Module):
    """Encode only visible patches; predict all patches and score hidden cells."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = PatchEncoder()
        self.decoder_projection = nn.Linear(ENCODER_WIDTH, 32)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 32))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("decoder_positions", _sincos_positions(32))
        self.decoder = _transformer_stack(32, depth=2)
        self.decoder_output = nn.Linear(32, PATCH_VALUES)

    def forward(
        self, x: Tensor, mask_ratio: float = 0.5, mask: Tensor | None = None
    ) -> dict[str, Tensor]:
        _check_images(x)
        batch = x.shape[0]
        if mask is None:
            if not 0 < mask_ratio < 1:
                raise ValueError("mask_ratio must be strictly between zero and one")
            n_masked = max(1, min(N_PATCHES - 1, round(N_PATCHES * mask_ratio)))
            order = torch.rand(batch, N_PATCHES, device=x.device).argsort(dim=1)
            mask = torch.zeros(batch, N_PATCHES, dtype=torch.bool, device=x.device)
            mask.scatter_(1, order[:, :n_masked], True)
            visible_indices = order[:, n_masked:]
        else:
            if mask.dtype != torch.bool or tuple(mask.shape) != (batch, N_PATCHES):
                raise ValueError("mask must be boolean with shape (batch, 102)")
            if mask.device != x.device:
                raise ValueError("mask and deposits must be on the same device")
            counts = mask.sum(dim=1)
            if not bool(((counts > 0) & (counts < N_PATCHES)).all()):
                raise ValueError("Every event needs at least one visible and one masked patch")
            if not bool((counts == counts[0]).all()):
                raise ValueError("Each event must have the same number of masked patches")
            n_visible = N_PATCHES - int(counts[0])
            indices = torch.arange(N_PATCHES, device=x.device).expand(batch, -1)
            visible_indices = indices[~mask].reshape(batch, n_visible)

        visible = self.decoder_projection(self.encoder(x, visible_indices))
        decoder_input = self.mask_token.to(dtype=visible.dtype).expand(batch, N_PATCHES, -1)
        decoder_input = decoder_input.scatter(
            1, visible_indices.unsqueeze(-1).expand(-1, -1, 32), visible
        )
        decoder_input = decoder_input + self.decoder_positions.to(dtype=decoder_input.dtype)
        predictions = self.decoder_output(self.decoder(decoder_input))
        return {"predictions": predictions, "targets": patchify(x).detach(), "mask": mask}


def supervised_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Weight energy and the pair of stored-coordinate targets equally."""
    if predictions.ndim != 2 or predictions.shape[1] != 3 or targets.shape != predictions.shape:
        raise ValueError("Predictions and targets must both have shape (batch, 3)")
    energy = F.mse_loss(predictions[:, 0], targets[:, 0])
    position = F.mse_loss(predictions[:, 1:], targets[:, 1:])
    return 0.5 * energy + 0.5 * position


def masked_loss(predictions: Tensor, targets: Tensor, masked_cells: Tensor) -> Tensor:
    """Balance positive and true-zero targets over hidden cells only.

    A (batch, tokens) patch mask is expanded over cells; an explicit cell mask is
    also accepted. Empty positive/zero groups contribute no weight. If no cells
    are selected, the result is a differentiable zero.
    """
    if predictions.ndim != 3 or predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must share (batch, tokens, cells) shape")
    if masked_cells.dtype != torch.bool:
        raise ValueError("masked_cells must be boolean")
    if masked_cells.shape == targets.shape[:2]:
        masked_cells = masked_cells.unsqueeze(-1).expand_as(targets)
    elif masked_cells.shape != targets.shape:
        raise ValueError("Mask must select patches or cells of the target tensor")
    error = (predictions - targets).square()
    positive = masked_cells & (targets > 0)
    zero = masked_cells & (targets == 0)
    n_positive = positive.sum()
    n_zero = zero.sum()
    positive_mean = torch.where(positive, error, 0).sum() / n_positive.clamp_min(1)
    zero_mean = torch.where(zero, error, 0).sum() / n_zero.clamp_min(1)
    active_groups = (n_positive > 0).to(error.dtype) + (n_zero > 0).to(error.dtype)
    return (positive_mean + zero_mean) / active_groups.clamp_min(1)

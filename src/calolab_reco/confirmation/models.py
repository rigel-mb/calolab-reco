"""Identical supervised local models and a temporary masked reconstruction decoder."""

from __future__ import annotations

import torch
from torch import nn

from calolab_reco.pilot_models import CNN, Transformer, _transformer_stack


class LocalCNN(nn.Module):
    """Local direct regression; deposit, edge mask and fixed relative indices.

    The observed anchor is supplied as known geometry, not a calibrated position.
    Output energy is predicted directly; there is no physical-sum correction.
    """

    def __init__(self):
        super().__init__()
        row, col = torch.meshgrid(torch.arange(-3, 4) / 3, torch.arange(-3, 4) / 3, indexing="ij")
        self.register_buffer("coordinates", torch.stack((row, col))[None])
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.GELU(),
        )
        self.energy = nn.Linear(66, 1)
        self.position = nn.Linear(66, 2)

    def forward(self, inputs, anchors):
        coordinates = self.coordinates.expand(len(inputs), -1, -1, -1)
        latent = self.encoder(torch.cat((inputs, coordinates), dim=1))
        features = torch.cat((latent, anchors / anchors.new_tensor([29, 84])), dim=1)
        return torch.cat((self.energy(features), self.position(features)), dim=1)


class LocalTransformer(nn.Module):
    """Cell-token attention over the same observed 7x7 window as LocalCNN.

    Tokens retain signed deposit, validity and fixed relative row/column indices.
    The observed anchor is appended only after token pooling, as for LocalCNN.
    """

    def __init__(self):
        super().__init__()
        row, col = torch.meshgrid(torch.arange(-3, 4) / 3, torch.arange(-3, 4) / 3, indexing="ij")
        self.register_buffer("coordinates", torch.stack((row, col))[None])
        self.embedding = nn.Linear(4, 64)
        self.encoder = _transformer_stack(64, depth=3)
        for module in self.encoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if isinstance(module, nn.MultiheadAttention):
                module.dropout = 0.0
        self.energy = nn.Linear(66, 1)
        self.position = nn.Linear(66, 2)

    def token_features(self, inputs):
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (2, 7, 7):
            raise ValueError("Expected local deposits and validity with shape (batch,2,7,7)")
        coordinates = self.coordinates.to(dtype=inputs.dtype).expand(len(inputs), -1, -1, -1)
        return torch.cat((inputs, coordinates), dim=1).flatten(2).transpose(1, 2)

    def forward(self, inputs, anchors):
        if anchors.ndim != 2 or tuple(anchors.shape) != (len(inputs), 2):
            raise ValueError("Expected observed anchors with shape (batch,2)")
        latent = self.encoder(self.embedding(self.token_features(inputs))).mean(dim=1)
        features = torch.cat((latent, anchors / anchors.new_tensor([29, 84])), dim=1)
        return torch.cat((self.energy(features), self.position(features)), dim=1)


class DirectRegressor(nn.Module):
    def __init__(self, kind, statistics, architecture="cnn"):
        super().__init__()
        networks = {
            ("full", "cnn"): CNN,
            ("full", "transformer"): Transformer,
            ("local", "cnn"): LocalCNN,
            ("local", "transformer"): LocalTransformer,
        }
        if (kind, architecture) not in networks:
            raise ValueError("Expected a full/local CNN or Transformer")
        self.network = networks[(kind, architecture)]()
        if architecture == "transformer" and kind == "full":
            # The earlier controlled Transformer experiment selected zero dropout.
            for module in self.network.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
                if isinstance(module, nn.MultiheadAttention):
                    module.dropout = 0.0
        self.local = kind == "local"
        self.register_buffer("mean", torch.tensor(statistics["target_mean"], dtype=torch.float32))
        self.register_buffer("std", torch.tensor(statistics["target_std"], dtype=torch.float32))

    def forward(self, inputs, anchors):
        standardized = self.network(inputs, anchors) if self.local else self.network(inputs)
        prediction = standardized * self.std + self.mean
        if self.local:
            prediction = torch.cat((prediction[:, :1], prediction[:, 1:] + anchors), dim=1)
        return prediction


class CyclingSampler:
    """Exactly sized updates across deterministic permutations, with resumable cursor."""

    def __init__(self, count, seed):
        self.count = count
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(count, generator=self.generator)
        self.cursor = 0
        self.cycles = 0

    def next(self, count):
        pieces = []
        while count:
            available = min(count, self.count - self.cursor)
            pieces.append(self.order[self.cursor : self.cursor + available])
            self.cursor += available
            count -= available
            if self.cursor == self.count:
                self.order = torch.randperm(self.count, generator=self.generator)
                self.cursor = 0
                self.cycles += 1
        return torch.cat(pieces)

    def state_dict(self):
        return {
            "count": self.count,
            "order": self.order.clone(),
            "cursor": self.cursor,
            "cycles": self.cycles,
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        if state["count"] != self.count:
            raise ValueError("Sampler count changed")
        self.order = state["order"].clone()
        self.cursor, self.cycles = state["cursor"], state["cycles"]
        self.generator.set_state(state["generator"])


def encoder_state(model):
    """The embedding, spatial constants and all attention blocks, without task heads."""
    network = model.network if isinstance(model, DirectRegressor) else model.network
    return {
        k: v.detach().cpu().clone()
        for k, v in network.state_dict().items()
        if k == "coordinates" or k.startswith(("embedding.", "encoder."))
    }


def transfer_encoder(model, state):
    expected = encoder_state(model)
    if set(state) != set(expected):
        raise ValueError("Encoder transfer has missing or extra parameters")
    heads = {
        k: v.detach().clone()
        for k, v in model.network.state_dict().items()
        if k.startswith(("energy.", "position."))
    }
    model.network.load_state_dict(state, strict=False)
    if any(not torch.equal(v, model.network.state_dict()[k]) for k, v in heads.items()):
        raise RuntimeError("Transfer changed a freshly initialized head")
    return True


class LocalMaskedModel(nn.Module):
    """Masked-token reconstruction of observed deposits, not clean-truth denoising.

    Masked values are replaced before embedding. An additional learned vector
    distinguishes a hidden value from a measured zero. Invalid edge cells are
    never reconstruction targets. The downstream embedding/encoder is identical.
    """

    def __init__(self):
        super().__init__()
        self.network = LocalTransformer()
        del self.network.energy
        del self.network.position
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 64))
        nn.init.normal_(self.mask_token, std=0.02)
        self.projection = nn.Linear(64, 32)
        self.decoder = _transformer_stack(32, depth=2)
        for module in self.decoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if isinstance(module, nn.MultiheadAttention):
                module.dropout = 0.0
        self.output = nn.Linear(32, 1)

    def forward(self, inputs, mask):
        valid = inputs[:, 1].flatten(1).bool()
        if mask.shape != valid.shape or mask.dtype != torch.bool or (mask & ~valid).any():
            raise ValueError("Mask must select valid measured cells only")
        features = self.network.token_features(inputs).clone()
        features[:, :, 0] = features[:, :, 0].masked_fill(mask, 0)
        embedded = self.network.embedding(features) + mask.unsqueeze(-1) * self.mask_token
        latent = self.network.encoder(embedded)
        return self.output(self.decoder(self.projection(latent))).squeeze(-1)


def sample_mask(inputs, ratio, generator):
    valid = inputs[:, 1].flatten(1).bool().cpu()
    if not 0 < ratio < 1 or (valid.sum(1) < 2).any():
        raise ValueError("Need at least two valid cells and a mask ratio in (0,1)")
    ranks = torch.rand(valid.shape, generator=generator).masked_fill(~valid, 2).argsort(1)
    counts = (valid.sum(1) * ratio).round().long().clamp(min=1)
    counts = torch.minimum(counts, valid.sum(1) - 1)
    mask = torch.zeros_like(valid)
    mask.scatter_(1, ranks, torch.arange(valid.shape[1])[None] < counts[:, None])
    return mask


def choose_mask(inputs, config):
    records = []
    for ratio in (config["mask_ratio"], config["mask_fallback_ratio"]):
        generator = torch.Generator().manual_seed(config["mask_seed"])
        mask = sample_mask(inputs, ratio, generator)
        positive = (inputs[:, 0].flatten(1) > 0) & inputs[:, 1].flatten(1).bool()
        hidden = positive.any(1) & ~(positive & ~mask).any(1)
        records.append(
            dict(
                ratio=ratio,
                all_positive_hidden_fraction=float(hidden.float().mean()),
                zero_positive_events=int((~positive.any(1)).sum()),
            )
        )
        if records[-1]["all_positive_hidden_fraction"] <= config["mask_all_positive_hidden_limit"]:
            break
    return dict(ratio=records[-1]["ratio"], checks=records, selection="train measured inputs only")


def reconstruction_loss(predictions, targets, mask, group_counts=None):
    """Equal positive/zero group weights across the entire effective batch."""
    groups = (mask & (targets > 0), mask & (targets == 0))
    if (targets < 0).any():
        raise ValueError("This pretraining protocol uses nonnegative noise-plus-cut inputs")
    counts = [int(g.sum()) for g in groups] if group_counts is None else group_counts
    active = sum(c > 0 for c in counts)
    result = predictions.sum() * 0
    for group, count in zip(groups, counts, strict=True):
        if count:
            result = result + ((predictions - targets).square() * group).sum() / (count * active)
    return result

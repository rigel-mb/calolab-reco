"""Model contracts and numerical checks on small synthetic inputs only."""

import pytest
import torch

from calolab_reco.pilot_models import (
    CNN,
    MaskedAutoencoder,
    Transformer,
    masked_loss,
    patchify,
    supervised_loss,
    unpatchify,
)


@pytest.fixture(autouse=True, scope="module")
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_patches_retain_every_cell_in_known_order():
    image = torch.arange(30 * 85, dtype=torch.float32).reshape(1, 1, 30, 85)
    patches = patchify(image)
    assert patches.shape == (1, 102, 25)
    torch.testing.assert_close(patches[0, 0], image[0, 0, :5, :5].reshape(-1))
    torch.testing.assert_close(patches[0, 1], image[0, 0, :5, 5:10].reshape(-1))
    torch.testing.assert_close(patches[0, 17], image[0, 0, 5:10, :5].reshape(-1))
    torch.testing.assert_close(unpatchify(patches), image, rtol=0, atol=0)


@pytest.mark.parametrize("model_type", [CNN, Transformer])
def test_regression_accepts_only_deposits_and_has_finite_gradients(model_type):
    model = model_type()
    deposits = torch.rand(2, 1, 30, 85)
    predictions = model(deposits)
    assert predictions.shape == (2, 3)
    loss = supervised_loss(predictions, torch.randn(2, 3))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match="deposits with shape"):
        model(torch.rand(2, 3, 30, 85))


def test_regression_loss_gives_each_task_half_weight():
    prediction = torch.tensor([[2.0, 2.0, 4.0]], requires_grad=True)
    loss = supervised_loss(prediction, torch.zeros_like(prediction))
    assert loss.item() == pytest.approx(7.0)  # 0.5 * 4 + 0.5 * mean(4, 16)
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[2.0, 1.0, 2.0]]))


def test_cnn_pooling_keeps_the_final_spatial_shape_and_all_border_cells():
    model = CNN()
    pools = [layer for layer in model.encoder if isinstance(layer, torch.nn.AvgPool2d)]
    grid = torch.ones(1, 1, 30, 85)
    assert pools[0](grid).shape[-2:] == (15, 43)
    assert pools[1](pools[0](grid)).shape[-2:] == (8, 22)
    final_input = torch.ones(1, 32, 8, 22, requires_grad=True)
    result = pools[2](final_input)
    assert result.shape == (1, 32, 3, 5)
    result.sum().backward()
    assert (final_input.grad > 0).all()


def test_hidden_values_cannot_reach_the_encoder_or_predictions():
    model = MaskedAutoencoder().eval()
    deposits = torch.rand(2, 1, 30, 85)
    mask = torch.zeros(2, 102, dtype=torch.bool)
    mask[:, ::2] = True
    altered = torch.where(mask.unsqueeze(-1), patchify(deposits) + 100, patchify(deposits))
    first = model(deposits, mask=mask)
    second = model(unpatchify(altered), mask=mask)
    torch.testing.assert_close(first["predictions"], second["predictions"], rtol=0, atol=0)
    assert not torch.equal(first["targets"][mask], second["targets"][mask])
    assert torch.equal(first["mask"], mask)


def test_masked_zero_is_explicit_and_all_model_gradients_are_finite():
    model = MaskedAutoencoder()
    result = model(torch.zeros(2, 1, 30, 85), mask_ratio=0.5)
    assert result["predictions"].shape == result["targets"].shape == (2, 102, 25)
    assert result["mask"].dtype == torch.bool
    assert result["mask"].sum(dim=1).tolist() == [51, 51]
    assert not result["targets"].any()
    loss = masked_loss(result["predictions"], result["targets"], result["mask"])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.mask_token.grad.abs().sum() > 0


def test_random_mask_repeats_after_restoring_torch_rng():
    model = MaskedAutoencoder().eval()
    deposits = torch.rand(1, 1, 30, 85)
    state = torch.get_rng_state()
    first = model(deposits)
    torch.set_rng_state(state)
    second = model(deposits)
    torch.testing.assert_close(first["mask"], second["mask"])
    torch.testing.assert_close(first["predictions"], second["predictions"], rtol=0, atol=0)


@pytest.mark.parametrize("fill", [False, True])
def test_mask_rejects_events_without_both_visible_and_hidden_patches(fill):
    with pytest.raises(ValueError, match="visible and one masked"):
        MaskedAutoencoder()(torch.zeros(1, 1, 30, 85), mask=torch.full((1, 102), fill))


def test_masked_loss_balances_groups_and_excludes_unmasked_cells():
    prediction = torch.tensor([[[0.0, 0.0], [2.0, 2.0], [99.0, 99.0]]], requires_grad=True)
    targets = torch.tensor([[[1.0, 3.0], [0.0, 0.0], [0.0, 10.0]]])
    mask = torch.tensor([[True, True, False]])
    loss = masked_loss(prediction, targets, mask)
    assert loss.item() == pytest.approx(4.5)  # Equal mean weights for errors 5 and 4.
    loss.backward()
    assert torch.equal(prediction.grad[:, 2], torch.zeros(1, 2))
    cell_mask = mask.unsqueeze(-1).expand_as(targets)
    assert masked_loss(prediction, targets, cell_mask).item() == pytest.approx(4.5)


@pytest.mark.parametrize("positive", [False, True])
def test_masked_loss_renormalizes_when_one_group_is_absent(positive):
    values = torch.tensor([[[1.0, 3.0]]])
    prediction, targets = (
        (torch.zeros_like(values), values)
        if positive
        else (
            values,
            torch.zeros_like(values),
        )
    )
    assert masked_loss(prediction, targets, torch.tensor([[True]])).item() == pytest.approx(5.0)


def test_pretrained_encoder_transfers_strictly_to_regression():
    pretraining = MaskedAutoencoder().eval()
    regression = Transformer().eval()
    regression.encoder.load_state_dict(pretraining.encoder.state_dict(), strict=True)
    deposits = torch.rand(2, 1, 30, 85)
    torch.testing.assert_close(
        pretraining.encoder(deposits), regression.encoder(deposits), rtol=0, atol=0
    )

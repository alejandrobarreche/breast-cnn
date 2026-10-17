from __future__ import annotations

from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import nn

from breast_cnn.models import ModelConfig, build_model, load_model_config, summarize_model

ROOT = Path(__file__).resolve().parents[1]


def small_config(**changes: object) -> dict[str, object]:
    config: dict[str, object] = {
        "in_channels": 3,
        "kernel_size": 3,
        "activation": "relu",
        "stages": [{"channels": 4, "convs": 1}, {"channels": 8, "convs": 2}],
        "dropout": 0.3,
    }
    return config | changes


def test_baseline_yaml_builds_single_logit_model() -> None:
    model = build_model(load_model_config(ROOT / "configs/model/baseline.yaml"))
    logits = model.eval()(torch.rand(2, 3, 256, 256))
    assert logits.shape == (2,)
    assert logits.dtype == torch.float32


def test_parameter_count_matches_hand_calculation() -> None:
    model = build_model(ModelConfig.model_validate(small_config()))
    convs = 9 * 3 * 4 + 9 * 4 * 8 + 9 * 8 * 8  # 3x3 kernels, no bias
    batch_norms = 2 * (4 + 8 + 8)
    linear = 8 + 1
    assert sum(p.numel() for p in model.parameters()) == convs + batch_norms + linear


def test_summary_tracks_shapes_and_parameters() -> None:
    model = build_model(ModelConfig.model_validate(small_config()))
    rows = summarize_model(model, (3, 32, 32))
    assert rows[0]["layer"] == "Conv2d"
    assert rows[0]["output_shape"] == (4, 32, 32)
    pools = [row["output_shape"] for row in rows if row["layer"] == "MaxPool2d"]
    assert pools == [(4, 16, 16), (8, 8, 8)]
    assert rows[-1]["output_shape"] == (1,)
    assert sum(row["parameters"] for row in rows) == sum(p.numel() for p in model.parameters())
    assert model.training


@pytest.mark.parametrize("activation", ["relu", "leaky_relu", "elu", "gelu"])
def test_activation_is_configurable(activation: str) -> None:
    model = build_model(ModelConfig.model_validate(small_config(activation=activation)))
    expected = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "elu": nn.ELU, "gelu": nn.GELU}
    assert any(isinstance(module, expected[activation]) for module in model.modules())


def test_initialization_is_he_for_convolutions_and_zero_bias_for_output() -> None:
    torch.manual_seed(0)
    config = small_config(stages=[{"channels": 256, "convs": 1}])
    model = build_model(ModelConfig.model_validate(config))
    conv = next(m for m in model.modules() if isinstance(m, nn.Conv2d))
    expected_std = (2 / (3 * 3 * 3)) ** 0.5  # He: sqrt(2 / fan_in)
    assert conv.weight.std().item() == pytest.approx(expected_std, rel=0.05)
    assert torch.all(model.head[-1].bias == 0)


@pytest.mark.parametrize(
    "changes",
    [
        {"activation": "selu"},
        {"stages": []},
        {"stages": [{"channels": 0, "convs": 1}]},
        {"kernel_size": 4},
        {"dropout": 1.0},
        {"unknown": 1},
    ],
)
def test_invalid_config_is_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ModelConfig.model_validate(small_config(**changes))

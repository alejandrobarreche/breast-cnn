"""Build a plain 2D CNN from a YAML description; no architecture is hard-coded here."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

Activation = Literal["relu", "leaky_relu", "elu", "gelu"]

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


class StageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    channels: int = Field(gt=0)
    convs: int = Field(gt=0)
    # "separable": depthwise k x k per channel followed by a pointwise 1 x 1 convolution.
    conv: Literal["standard", "separable"] = "standard"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    in_channels: int = Field(gt=0)
    kernel_size: int = Field(gt=0)
    activation: Activation
    stages: list[StageConfig] = Field(min_length=1)
    dropout: float = Field(ge=0, lt=1)

    def model_post_init(self, __context: object) -> None:
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so that 'same' padding is symmetric")


def load_model_config(path: Path | str) -> ModelConfig:
    """Read and validate an architecture YAML."""
    return ModelConfig.model_validate(OmegaConf.to_container(OmegaConf.load(path), resolve=True))


class CNN(nn.Module):
    """Convolutional stages, global average pooling and a single-logit linear head."""

    def __init__(self, features: nn.Sequential, head: nn.Sequential) -> None:
        super().__init__()
        self.features = features
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per input image, shape [B]."""
        return self.head(self.features(x)).squeeze(1)


def build_model(config: ModelConfig) -> CNN:
    """Construct the network and apply the initialization described in `_initialize`."""
    layers: list[nn.Module] = []
    in_channels = config.in_channels
    for stage in config.stages:
        for _ in range(stage.convs):
            # BatchNorm supplies the shift, so the convolution bias would be redundant.
            padding = config.kernel_size // 2
            if stage.conv == "separable":
                layers += [
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        config.kernel_size,
                        padding=padding,
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.Conv2d(in_channels, stage.channels, 1, bias=False),
                ]
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels, stage.channels, config.kernel_size, padding=padding, bias=False
                    )
                )
            layers += [
                nn.BatchNorm2d(stage.channels),
                _ACTIVATIONS[config.activation](),
            ]
            in_channels = stage.channels
        layers.append(nn.MaxPool2d(2))
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(config.dropout),
        nn.Linear(in_channels, 1),
    )
    model = CNN(nn.Sequential(*layers), head)
    _initialize(model, config.activation)
    return model


def _initialize(model: nn.Module, activation: Activation) -> None:
    """He normal for convolutions, Glorot uniform for the linear output, BN as identity.

    He initialization preserves activation variance for rectifiers; ELU and GELU have no
    closed-form gain and use the ReLU gain as an approximation. The output layer has no
    nonlinearity, so Glorot initialization is used there.
    """
    nonlinearity = "leaky_relu" if activation == "leaky_relu" else "relu"
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            # a=0.01 matches nn.LeakyReLU's default slope; it is ignored for "relu".
            nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity=nonlinearity)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)


def summarize_model(model: nn.Module, input_shape: tuple[int, ...]) -> list[dict[str, object]]:
    """List every leaf layer with its output shape (without batch) and parameter count."""
    rows: list[dict[str, object]] = []
    hooks = []
    for name, module in model.named_modules():
        if next(module.children(), None) is not None:
            continue
        hooks.append(
            module.register_forward_hook(
                lambda module, _inputs, output, name=name: rows.append(
                    {
                        "name": name,
                        "layer": type(module).__name__,
                        "output_shape": tuple(output.shape[1:]),
                        "parameters": sum(p.numel() for p in module.parameters()),
                    }
                )
            )
        )
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, *input_shape))
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    return rows

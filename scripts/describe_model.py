#!/usr/bin/env python3
"""Print the layer table (output volume and parameters) of an architecture YAML as Markdown."""

import argparse
from pathlib import Path

from breast_cnn.models import build_model, load_model_config, summarize_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, nargs="?", default=Path("configs/model/baseline.yaml"))
    args = parser.parse_args()
    config = load_model_config(args.config)
    rows = summarize_model(build_model(config), (config.in_channels, 256, 256))
    print("| Capa | Operación | Volumen de salida | Parámetros |")
    print("|---|---|---|---:|")
    print(f"| input | — | {config.in_channels} × 256 × 256 | — |")
    for row in rows:
        shape = " × ".join(str(size) for size in row["output_shape"])
        parameters = f"{row['parameters']:,}".replace(",", ".")
        print(f"| {row['name']} | {row['layer']} | {shape} | {parameters} |")
    total = sum(row["parameters"] for row in rows)
    print(f"| **Total** | | | **{total:,}** |".replace(",", "."))


if __name__ == "__main__":
    main()

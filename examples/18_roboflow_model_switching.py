"""Example: Roboflow integration and model switching.

Demonstrates:
1. Loading a Roboflow dataset
2. Switching between different encoders
3. Few-shot evaluation
"""

from __future__ import annotations

import argparse

from lumen import ModelSwitcher, preset_configs
from lumen.data import ROBOFLOW_AVAILABLE, build_roboflow_dataset, RoboflowDatasetConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Roboflow + Model Switching Example")
    parser.add_argument("--encoder", choices=["eupe", "dinov3"], default="eupe")
    parser.add_argument("--roboflow-api-key", type=str, help="Roboflow API key")
    parser.add_argument("--roboflow-workspace", type=str, help="Roboflow workspace")
    parser.add_argument("--roboflow-project", type=str, help="Roboflow project")
    parser.add_argument("--roboflow-version", type=str, default="latest")
    args = parser.parse_args()

    # Check Roboflow availability
    if not ROBOFLOW_AVAILABLE:
        print("Roboflow not available. Install with: pip install roboflow")
        return

    # Build Roboflow dataset if credentials provided
    if args.roboflow_api_key and args.roboflow_workspace and args.roboflow_project:
        config = RoboflowDatasetConfig(
            api_key=args.roboflow_api_key,
            workspace=args.roboflow_workspace,
            project=args.roboflow_project,
            version=args.roboflow_version,
            task_type="classification",
        )
        dataset = build_roboflow_dataset(config)
        print(f"Loaded Roboflow dataset: {len(dataset)} samples")

    # Demonstrate model switching
    print("\n=== Model Switching Demo ===")
    print(f"Available presets: {list(preset_configs().keys())}")

    # Load EUPE preset
    switcher = ModelSwitcher(preset_configs()["multihead_dinov3"])
    print(f"\n1. Initial config: {switcher.config.encoder_name} encoder")

    # Switch to EUPE
    switcher.switch_encoder("eupe-pretrained", variant="vit_t")
    print(f"2. Switched to: {switcher.config.encoder_name} encoder")

    # Build model
    model = switcher.get_model()
    print(f"\n3. Built model:")
    print(f"   - Encoder: {type(model.encoder).__name__}")
    print(f"   - Embed dim: {model.encoder.embed_dim}")
    print(f"   - Patch size: {model.encoder.patch_size}")

    # Switch head configuration
    switcher.switch_heads(
        classification_num_classes=5,
        segmentation_num_classes=3,
        use_mae=True,
    )
    print(f"\n4. Switched heads:")
    print(f"   - Classification: {switcher.config.classification_num_classes} classes")
    print(f"   - Segmentation: {switcher.config.segmentation_num_classes} classes")
    print(f"   - MAE: {switcher.config.use_mae}")

    # Save configuration
    switcher.save_config("model_switch_config.json")
    print("\n5. Configuration saved to model_switch_config.json")


if __name__ == "__main__":
    main()

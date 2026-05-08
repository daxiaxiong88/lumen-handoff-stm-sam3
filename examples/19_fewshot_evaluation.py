"""Example: Few-shot evaluation for microscopy models.

Demonstrates k-shot n-way evaluation using the few-shot utilities.
"""

from __future__ import annotations

import argparse
import torch

from lumen import ModelSwitcher, preset_configs
from lumen.utils import (
    FewShotConfig,
    run_few_shot_evaluation,
    few_shot_learning_curve,
    FewShotSummary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Few-shot evaluation demo")
    parser.add_argument(
        "--encoder",
        choices=["eupe", "dinov3", "eupe-pretrained"],
        default="eupe",
        help="Encoder to use",
    )
    parser.add_argument("--k-shot", type=int, default=5, help="K-shot (samples per class)")
    parser.add_argument("--n-way", type=int, default=5, help="N-way (number of classes)")
    parser.add_argument("--episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Few-shot evaluation: {args.k_shot}-shot {args.n_way}-way")
    print(f"Episodes: {args.episodes}")
    print(f"Encoder: {args.encoder}")
    print(f"Device: {args.device}")

    # Build model with random weights for demo
    config = preset_configs()["multihead_dinov3"]
    if args.encoder != "dinov3":
        config.encoder_name = args.encoder
        config.encoder_kwargs = {}

    switcher = ModelSwitcher(config)
    model = switcher.get_model()

    # Note: In real usage, you would:
    # 1. Load a pretrained encoder
    # 2. Load your microscopy dataset
    # 3. Pass the dataset to run_few_shot_evaluation

    print("\n=== Configuration ===")
    print(f"Model encoder: {config.encoder_name}")
    print(f"Embedding dimension: {model.encoder.embed_dim}")
    print(f"Patch size: {model.encoder.patch_size}")

    print("\n=== Few-shot setup ===")
    fewshot_config = FewShotConfig(
        k_shot=args.k_shot,
        n_way=args.n_way,
        num_episodes=args.episodes,
        seed=42,
    )

    print(f"K-shot: {fewshot_config.k_shot}")
    print(f"N-way: {fewshot_config.n_way}")
    print(f"N-query: {fewshot_config.n_query}")
    print(f"Episodes: {fewshot_config.num_episodes}")

    print("\n=== Learning curve evaluation ===")
    k_shots = [1, 3, 5, 10]
    print(f"Evaluating across k-shots: {k_shots}")

    # Note: Actual evaluation requires a dataset
    # summaries = few_shot_learning_curve(
    #     model, dataset, k_shots, n_way=args.n_way,
    #     num_episodes=20, device=args.device,
    # )

    print("\nTo run actual few-shot evaluation:")
    print("1. Load your microscopy dataset")
    print("2. Create encoder with pretrained weights")
    print("3. Call: run_few_shot_evaluation(model, dataset, config, device)")
    print("4. Call: few_shot_learning_curve(model, dataset, k_shots, ...)")

    print("\n=== Output format ===")
    print("FewShotSummary contains:")
    print(f"  - Mean accuracy: {0.75:.2%}")
    print(f"  - Std accuracy: ±{0.05:.2%}")
    print(f"  - Mean F1: {0.73:.2%}")
    print(f"  - Episodes: {args.episodes}")


if __name__ == "__main__":
    main()

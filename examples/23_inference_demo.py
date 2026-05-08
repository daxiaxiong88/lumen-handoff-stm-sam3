"""Example: Inference API demonstration.

Shows how to use the inference API for:
1. Single image inference
2. Batch inference
3. Model switching
4. Saving predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lumen.inference import (
    InferenceConfig,
    InferenceServer,
    MicroscopyInference,
)
from lumen.utils import CheckpointManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference API demonstration"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        help="ID of checkpoint to load (default: best)",
    )
    parser.add_argument(
        "--encoder",
        choices=["eupe", "eupe-pretrained", "dinov3"],
        default="eupe-pretrained",
        help="Encoder architecture",
    )
    parser.add_argument(
        "--head",
        choices=["segmentation", "upernet"],
        default="upernet",
        help="Segmentation head type",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "segmentation", "detection"],
        default="segmentation",
        help="Task type",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="Single image to run inference on",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Directory of images to run batch inference",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("inference_results"),
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for batch inference",
    )
    return parser.parse_args()


def single_image_inference(args: argparse.Namespace) -> None:
    """Run inference on a single image."""
    print("\n" + "=" * 60)
    print("SINGLE IMAGE INFERENCE")
    print("=" * 60)

    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)

    # Create inference config
    config = InferenceConfig(
        checkpoint_path=args.checkpoint_dir / f"{args.checkpoint_id}.pt" if args.checkpoint_id else None,
        encoder_name=args.encoder,
        head_name=args.head,
        task_type=args.task,
        device=args.device,
    )

    # Create inference instance
    inference = MicroscopyInference(config, checkpoint_manager)

    # Load model
    inference.load_model()

    # Show model info
    model_info = inference.get_model_info()
    print(f"\nModel Info:")
    print(f"  Encoder: {model_info['encoder']}")
    print(f"  Head: {model_info['head']}")
    print(f"  Parameters: {model_info['parameters']:.2f}M")
    print(f"  Device: {model_info['device']}")
    print(f"  Status: {model_info['status']}")

    # Load image
    print(f"\nLoading image: {args.image}")
    from lumen.data.dataset import load_image_array
    image, _ = load_image_array(args.image)

    # Run inference
    print("Running inference...")
    result = inference.infer(image)

    # Show results
    print(f"\nResults:")
    print(f"  Latency: {result.latency_ms:.2f}ms")
    print(f"  Prediction shape: {result.predictions.shape}")

    # Save prediction
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"prediction_{args.image.stem}.png"

    from PIL import Image
    if result.predictions.ndim == 2:
        img = Image.fromarray(result.predictions.astype(np.uint8), mode="L")
    else:
        img = Image.fromarray(result.predictions.astype(np.uint8))

    img.save(output_path)
    print(f"  Saved to: {output_path}")


def batch_inference(args: argparse.Namespace) -> None:
    """Run batch inference on multiple images."""
    print("\n" + "=" * 60)
    print("BATCH INFERENCE")
    print("=" * 60)

    # Setup checkpoint manager
    checkpoint_manager = CheckpointManager(args.checkpoint_dir)

    # Create inference config
    config = InferenceConfig(
        checkpoint_path=args.checkpoint_dir / f"{args.checkpoint_id}.pt" if args.checkpoint_id else None,
        encoder_name=args.encoder,
        head_name=args.head,
        task_type=args.task,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Create inference instance
    inference = MicroscopyInference(config, checkpoint_manager)
    inference.load_model()

    # Get all images
    image_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    image_paths = [
        p for ext in image_extensions
        for p in args.image_dir.glob(f"*{ext}")
        for p in args.image_dir.glob(f"*{ext.upper()}")
    ]

    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return

    print(f"\nFound {len(image_paths)} images")

    # Run batch inference
    results = inference.infer_batch(image_paths, batch_size=args.batch_size)

    # Calculate stats
    latencies = [r.latency_ms for r in results]
    print(f"\nInference Statistics:")
    print(f"  Total images: {len(results)}")
    print(f"  Avg latency: {np.mean(latencies):.2f}ms")
    print(f"  Min latency: {min(latencies):.2f}ms")
    print(f"  Max latency: {max(latencies):.2f}ms")

    # Save predictions
    inference.save_predictions(results, args.output_dir)


def server_inference_demo(args: argparse.Namespace) -> None:
    """Demonstrate InferenceServer usage."""
    print("\n" + "=" * 60)
    print("INFERENCE SERVER DEMO")
    print("=" * 60)

    # Create server
    server = InferenceServer(args.checkpoint_dir, device=args.device)

    # Load models
    print("\nLoading models...")

    server.load_model("eupe_upernet", checkpoint_id="best", task_type="segmentation")
    server.load_model("dinov3_upernet", checkpoint_id="latest", task_type="segmentation")

    # List loaded models
    print("\nLoaded models:")
    for model_info in server.list_models():
        print(f"  - {model_info['id']}: {model_info['info']['encoder']}")

    # Demonstrate model switching
    print("\nSwitching between models...")

    # Get sample image
    from lumen.data.dataset import load_image_array
    sample_paths = list((args.image_dir if args.image_dir else args.checkpoint_dir).glob("*.png"))[:1]
    if not sample_paths:
        print("No sample images found")
        return

    sample_image, _ = load_image_array(sample_paths[0])

    for model_id in ["eupe_upernet", "dinov3_upernet"]:
        print(f"\nUsing model: {model_id}")
        server.switch_model(model_id)
        result = server.infer(sample_image, model_id=model_id)
        print(f"  Latency: {result.latency_ms:.2f}ms")


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("INFERENCE API DEMONSTRATION")
    print("=" * 60)
    print(f"\nCheckpoint directory: {args.checkpoint_dir}")
    print(f"Task: {args.task}")
    print(f"Device: {args.device}")

    # Determine mode
    if args.image:
        single_image_inference(args)
    elif args.image_dir:
        batch_inference(args)
    else:
        server_inference_demo(args)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

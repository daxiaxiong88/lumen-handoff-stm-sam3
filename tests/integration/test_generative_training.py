"""Real-model smoke test for the Vision Banana LoRA trainer.

Loads FLUX.2-klein-4B, runs a couple of rectified-flow training steps on a
synthetic segmentation target, saves the LoRA adapter, and confirms the loss is
finite. Gated on: the checkpoint being cached, ``diffusers``+``peft`` importing
cleanly, and **enough free GPU memory** (~18 GB) — the model needs ~16 GB to
load, so a busy GPU (e.g. another kernel) causes a skip rather than an error.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
CLASS_NAMES = ["background", "object_a", "object_b"]
CLASS_COLORS = {"background": BLACK, "object_a": GREEN, "object_b": RED}


def _flux_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return (
            try_to_load_from_cache(
                "black-forest-labs/FLUX.2-klein-4B", "model_index.json"
            )
            is not None
        )
    except Exception:
        return False


def _backend_ready() -> bool:
    try:
        import peft  # noqa: F401
        from diffusers import Flux2KleinPipeline  # noqa: F401

        return True
    except Exception:
        return False


def _gpu_free_mib() -> int:
    if not torch.cuda.is_available():
        return 0
    try:
        free, _total = torch.cuda.mem_get_info()
        return int(free // (1024 * 1024))
    except Exception:
        return 0


# FLUX.2-klein-4B needs ~16 GB to load (transformer ~8 GB + Qwen3 text encoder
# ~7.6 GB); leave headroom for activations during training.
_MIN_FREE_MIB = 18_000
_TRAIN_READY = _flux_cached() and _backend_ready() and _gpu_free_mib() >= _MIN_FREE_MIB


def _make_synthetic_dataset() -> object:
    from lumen.training.generative import InMemorySegDataset

    rng = np.random.default_rng(0)
    images, label_maps = [], []
    for _ in range(2):
        img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        lm = np.zeros((256, 256), dtype=int)
        lm[:128] = 1
        lm[128:] = 2
        images.append(img)
        label_maps.append(lm)
    return InMemorySegDataset(images, label_maps, CLASS_NAMES, CLASS_COLORS)


@pytest.mark.skipif(not _TRAIN_READY, reason="FLUX.2-klein unavailable or GPU memory < 18 GB free")
def test_lora_training_smoke(tmp_path: Path) -> None:
    from lumen.models import build_segmenter
    from lumen.training.generative import Flux2KleinLoRATrainer, LoRAConfig

    segmenter = build_segmenter("vision_banana")
    trainer = Flux2KleinLoRATrainer(
        segmenter.pipe, LoRAConfig(rank=8, alpha=8, lr=1e-4)
    )
    dataset = _make_synthetic_dataset()

    losses = trainer.fit(dataset, steps=2, height=256, width=256)

    assert len(losses) == 2
    assert all(np.isfinite(losses)), f"non-finite losses: {losses}"

    adapter_dir = tmp_path / "vb_lora"
    trainer.save_lora(adapter_dir)
    assert (adapter_dir / "adapter_config.json").exists()
    assert any(adapter_dir.glob("adapter_model.*"))

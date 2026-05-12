# HyperData Integration

Lumen integrates with [HyperData](https://github.com/hyper-instrument/hyper-data) to
load scientific image data and push trained model weights back—all with built-in
version control via IceChunk/Zarr.

## Installation

```bash
uv pip install -e ".[hyperdata]"
```

This installs the `hyperdata` SDK as an optional dependency.

## Quick Start

```python
from lumen.data.hyperdata import (
    HyperDataImageDataset,
    HyperDataSegmentationDataset,
    WeightManager,
)
```

### Load images for self-supervised training

```python
dataset = HyperDataImageDataset(
    "./my_data",          # local path or @user/project virtual path
    array_name="images",  # Zarr array name
    channels=3,           # expand grayscale → 3-ch if needed
    image_size=224,        # resize on the fly
)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)
```

Each sample is a dict `{"image": Tensor(C, H, W)}` matching Lumen's batch
convention, so it plugs directly into `train_self_supervised_epoch`.

### Load paired images + masks for segmentation

```python
seg_dataset = HyperDataSegmentationDataset(
    "./my_data",
    image_array="images",
    mask_array="masks",
    image_size=256,
    channels=1,
)
```

Returns `{"image": (C, H, W), "mask": (H, W)}` compatible with
`SegmentationTrainer`.

### Push model weights

```python
wm = WeightManager("./weights_store")
wm.push_weights(
    model,
    message="epoch-50 pretrain",
    tag="v1.0",
    metrics={"loss": 0.12},
)
```

Weights are stored as a byte blob in a versioned Zarr array. A companion
metadata array records architecture info, parameter count, timestamp, and
optional metrics.

### Pull weights into a fresh model

```python
wm = WeightManager("./weights_store")
meta = wm.pull_weights(model, tag="v1.0")
print(meta)  # {'model_class': ..., 'num_parameters': ..., 'metrics': ...}
```

### Full checkpoint (model + optimizer + epoch)

```python
# Push
wm.push_checkpoint(model, optimizer, epoch=50, message="mid-training")

# Pull
ckpt = wm.pull_checkpoint(model, optimizer)
start_epoch = ckpt["epoch"]
```

## API Reference

### `open_hyperdata(path, *, branch="main")`

Open a HyperData dataset. Returns a `HyperData` instance.

### `HyperDataImageDataset(dataset, array_name, **kw)`

| Parameter    | Type             | Default    | Description                          |
|-------------|------------------|------------|--------------------------------------|
| `dataset`    | `str \| HyperData` | required | Path or HyperData instance          |
| `array_name` | `str`            | `"images"` | Zarr array name                      |
| `transform`  | callable         | `None`     | `(C,H,W) → (C,H,W)` transform      |
| `normalize`  | `bool`           | `True`     | Scale to `[0,1]`                     |
| `image_size` | `int \| (H,W)`   | `None`     | Resize target                        |
| `channels`   | `int`            | `None`     | Target channel count (1 or 3)        |
| `branch`     | `str`            | `"main"`   | HyperData branch                     |

### `HyperDataSegmentationDataset(dataset, image_array, mask_array, **kw)`

Same keyword arguments as above, plus:

| Parameter     | Type  | Default    |
|--------------|-------|------------|
| `image_array` | `str` | `"images"` |
| `mask_array`  | `str` | `"masks"`  |

### `WeightManager(dataset, *, branch="main")`

| Method             | Description                              |
|-------------------|------------------------------------------|
| `push_weights()`   | Serialize & store `state_dict` bytes     |
| `pull_weights()`   | Load `state_dict` into a model           |
| `push_checkpoint()` | Store model + optimizer + epoch          |
| `pull_checkpoint()` | Restore model + optimizer + epoch        |
| `list_tags()`      | List all version tags                    |
| `list_branches()`  | List all branches                        |

## Array Shape Conventions

| Input shape       | Interpretation       | Output tensor |
|-------------------|---------------------|---------------|
| `(N, H, W)`       | Grayscale stack     | `(1, H, W)`  |
| `(N, C, H, W)`    | Channel-first (C≤4) | `(C, H, W)`  |
| `(N, H, W, C)`    | Channel-last (C≤4)  | `(C, H, W)`  |

## Version Control

HyperData uses IceChunk for Git-like version control on Zarr arrays.
Every `push_weights` / `push_checkpoint` call creates a commit.
Optional `tag` parameter creates a named tag for easy retrieval.

```python
wm = WeightManager("./store")
wm.push_weights(model, tag="best")
# later ...
wm.pull_weights(model, tag="best")
```

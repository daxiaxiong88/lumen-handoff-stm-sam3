# Lumen Model Weight Manifest

Large checkpoints are intentionally not committed to git.

Local model-zoo assets are organized by family:

| Family | Path |
| --- | --- |
| EUPE ViT checkpoints | `model/eupe/*.pt` |
| EUPE ConvNeXt checkpoints | `model/convnext/*.pt` |
| DINO local Transformers checkpoints | `model/dino/<checkpoint-dir>/` |
| SAM3 local Transformers checkpoints | `model/sam3/` |
| Local benchmark/fine-tune checkpoints | `model/<dataset>/*.pt` |

Expected public microscopy benchmark artifact:

| Dataset | Task | Path | Required evidence |
| --- | --- | --- | --- |
| LiveCELL or TissueNet | Multi-head microscopy pretraining + fine-tuning | `model/<dataset>/lumen_multihead.pt` | Matching `.benchmarks/*.json` validated by `lumen.utils.validate_benchmark_report` |

Produced local benchmark artifact:

| Dataset | Task | Path | Evidence |
| --- | --- | --- | --- |
| LiveCELL | Segmentation, pretrained EUPE + multi-head SSL/supervised fine-tuning on official 2% split | `model/livecell/lumen_multihead.pt` | `.benchmarks/livecell_multihead.json` |

Latest LiveCELL result:

- Baseline supervised mIoU: `0.31893997073294417`
- Multi-head mIoU: `0.3764882796452597`
- Few-shot relative improvement: `0.18043617668887943`
- Joint/sequential compute ratio: `0.5327049203441075`

Validation gates:

- Few-shot improvement over pure supervised fine-tuning: `>= 0.05`
- Joint pretext compute divided by sequential supervised + SSL compute: `<= 0.70`
- `checkpoint_path` must point to the produced pretrained weights.

# Lumen Weight Manifest

Large checkpoints are intentionally not committed to git.

Expected public microscopy benchmark artifact:

| Dataset | Task | Path | Required evidence |
| --- | --- | --- | --- |
| LiveCELL or TissueNet | Multi-head microscopy pretraining + fine-tuning | `weights/<dataset>/lumen_multihead.pt` | Matching `.benchmarks/*.json` validated by `lumen.utils.validate_benchmark_report` |

Validation gates:

- Few-shot improvement over pure supervised fine-tuning: `>= 0.05`
- Joint pretext compute divided by sequential supervised + SSL compute: `<= 0.70`
- `checkpoint_path` must point to the produced pretrained weights.


# YOLO model artifacts

Each subfolder contains one trained Ultralytics YOLO run. Use `weights/best.pt` for inference; `last.pt` is the final training checkpoint. The accompanying `args.yaml`, `results.csv`, plots, and batch images document the original training run.

| Folder | Detector view |
| --- | --- |
| `sagittal_anatomy` | Midline sagittal anatomy |
| `anterior_coronal` | Anterior coronal view |
| `posterior_axial` | Posterior axial view |
| `straight_sinus` | Straight-sinus sagittal view |

The original training run names and dataset paths in `args.yaml` are provenance metadata; they are not needed for inference and deliberately retain their original values after these folders were renamed.

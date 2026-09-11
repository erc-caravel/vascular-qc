# YOLO model artifacts

Each subfolder contains one trained Ultralytics YOLO run. Use `weights/best.pt` for inference; `last.pt` is the final training checkpoint. The accompanying `args.yaml`, `results.csv`, plots, and batch images document the original training run.

| Folder | Detector view |
| --- | --- |
| `h100_run_MASTER_sagittal_no_color` | Midline sagittal anatomy |
| `h100_run_MASTER_coronal_anterior_NO_COLOR_70_pat` | Anterior coronal view |
| `h100_run_MASTER_axial_posterior_NO_COLOR_70_pat` | Posterior axial view |
| `h100_run_MASTER_straight_sinus_no_color` | Straight-sinus sagittal view |

The original training dataset paths in `args.yaml` are HPC paths kept for provenance; they are not needed for inference.

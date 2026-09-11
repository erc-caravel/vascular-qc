# Inference

`infer_yolo.py` applies four bundled YOLO models to a standardized MIP directory:

- sagittal — anterior/posterior/center detection on `MIPs_Mid_32`.
- coronal — anterior cross-view detection.
- axial — posterior cross-view detection.
- straight sinus — detection on `MIPs_Mid_14`.

The output CSV contains each model's predictions, plus metadata-projected best 1D IoU and cross-view result for anterior and posterior. The agreement threshold is `0.25`.

Run IXI inference:

```powershell
.\.venv\Scripts\python.exe inference\infer_yolo.py --input-dir results\mips\ixi\manual --weights-sag inference\yolo_models\sagittal_anatomy\weights\best.pt --weights-cor inference\yolo_models\anterior_coronal\weights\best.pt --weights-ax inference\yolo_models\posterior_axial\weights\best.pt --weights-ss inference\yolo_models\straight_sinus\weights\best.pt --output results\inference\ixi.csv
```

Add `--manual_inspection true` to save only views involved in spatial disagreement. Add `--debug true` to save all annotated views; it takes precedence over manual inspection.

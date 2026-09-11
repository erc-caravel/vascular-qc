# Utilities

`compute_metrics_crossview.py` is a legacy standalone evaluator for pre-existing JSON inference output. It projects coronal and axial detections back onto the sagittal canvas using the MIP metadata, computes best 1D IoU, and classifies agreement at IoU ≥ `0.25`.

The same projection and IoU logic is now used directly by `inference/infer_yolo.py`. Use this utility only when auditing an older JSON result set; edit `JSON_PATH` and `SAG_DIR` at the top before running it.

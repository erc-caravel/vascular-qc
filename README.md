# Vascular QC

Vascular QC is an MRA quality-control pipeline for IXI and TubeTK cases. It registers MRA and vessel-segmentation volumes to MNI space, generates standardized multi-view MIPs, optionally creates TubeTK augmentations, and runs four YOLO detectors with spatial cross-view QA.

## Pipeline

```text
data/ → registration → MIP generation → optional augmentation → YOLO inference → CSV / review images
```

## Setup

On Windows PowerShell, create and populate the environment once:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick start

Run either registration script; their defaults use the repository data layout:

```powershell
.\.venv\Scripts\python.exe processing\registration\ANT_register_mra_t1_mni_batch_args.py
.\.venv\Scripts\python.exe processing\registration\ANT_register_TubeTK_batch_args.py
```

Generate MIPs with a label that identifies the segmentation source:

```powershell
.\.venv\Scripts\python.exe processing\generate_mips\full_pipeline_mid_sags_and_axial_IXI_metadata.py --input_dir data --output_dir results\mips\ixi --model_name manual
.\.venv\Scripts\python.exe processing\generate_mips\full_pipeline_mid_sags_and_axial.py --input_dir data --output_dir results\mips\tubetk --model_name manual
```

See the README in each folder for the remaining commands, expected inputs, and outputs.

## Repository layout

- `data/` — source MNI, MRA, T1, and vessel-segmentation files.
- `processing/` — registration, MIP-generation, and augmentation scripts.
- `inference/` — YOLO weights and multi-model inference script.
- `utils/` — standalone legacy QA utilities.
- `results/` — generated artifacts; ignored by Git except its README.

## Conventions

Standardized MIP files are named `<dataset>_<model_name>_<id>.png`, for example `IXI_manual_012.png` or `TubeTK_manual_105.png`. MIP metadata is stored beside the `MIPs_Mid_32/Plain` image as `<stem>_meta.json` and is required for cross-view IoU.

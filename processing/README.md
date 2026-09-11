# Processing

This folder holds the three preprocessing stages:

- `registration/` performs rigid MRA → T1 → padded-MNI registration.
- `generate_mips/` creates the four standardized detector views and spatial metadata.
- `augmentation/` creates additional TubeTK sagittal views for training/QC.

Run the scripts from the repository root with `.venv\Scripts\python.exe` so their repository-relative defaults resolve correctly.

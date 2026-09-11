# TubeTK augmentation

Both scripts use registered TubeTK volumes from `results/registration/tubetk/` and discover cases from standardized MIPs in `results/mips/tubetk/`.

```powershell
.\.venv\Scripts\python.exe processing\augmentation\grid_config_centerline_sagittal_TubeTK.py
.\.venv\Scripts\python.exe processing\augmentation\grid_config_centerline_straight_sinus_TubeTK.py
```

Outputs are written to `results/augmentation/tubetk/`.

- `sagittal/` creates five central-slab configurations at three left/right tilts (15 variants per case).
- `straight_sinus/` creates three un-tilted slab configurations per case.

Each plain sagittal image has a metadata JSON file and a paired axial QC image.

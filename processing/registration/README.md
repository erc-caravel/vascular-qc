# Registration

The two scripts rigidly align each MRA to its brain-only T1, then align the T1 to a padded MNI template. MRA data uses linear interpolation; vessel segmentations are thresholded to binary and resampled with nearest-neighbour interpolation.

Run:

```powershell
.\.venv\Scripts\python.exe processing\registration\ANT_register_mra_t1_mni_batch_args.py
.\.venv\Scripts\python.exe processing\registration\ANT_register_TubeTK_batch_args.py
```

Outputs are paired NIfTI files in `results/registration/ixi/` and `results/registration/tubetk/`. Every input/output path has an argument override; use `--help` to inspect them.

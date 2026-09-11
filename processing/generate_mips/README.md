# MIP generation

These self-contained pipelines create a shared registration cache plus standard detector inputs:

- `2_Sagittal_Crops/MIPs_Mid_32/Plain/` — sagittal model input and `<stem>_meta.json`.
- `2_Sagittal_Crops/MIPs_Mid_14/Plain/` — straight-sinus model input.
- `3_Posterior/` — posterior axial model input.
- `4_Anterior/Coronal_Isolated/` — anterior coronal model input.

Run IXI:

```powershell
.\.venv\Scripts\python.exe processing\generate_mips\full_pipeline_mid_sags_and_axial_IXI_metadata.py --input_dir data --output_dir results\mips\ixi --model_name manual
```

Run TubeTK:

```powershell
.\.venv\Scripts\python.exe processing\generate_mips\full_pipeline_mid_sags_and_axial.py --input_dir data --output_dir results\mips\tubetk --model_name manual
```

`model_name` becomes part of each standardized filename and identifies the segmentation source.

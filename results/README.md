# Results

Generated output is intentionally excluded from Git. The standard locations are:

```text
results/
├── registration/{ixi,tubetk}/
├── mips/{ixi,tubetk}/<model_name>/
├── augmentation/tubetk/{sagittal,straight_sinus}/
└── inference/
```

`mips/*/<model_name>/` is the input root for `inference/infer_yolo.py`. The adjacent `Shared_Registrations/` folder is an internal cache created by the MIP-generation scripts.

# Data

This folder contains input images only.

```text
data/
├── MNI_Template/mni152.nii.gz
├── MRA/IXI-MRA/IXI<ID>*-MRA.nii.gz
├── MRA/TubeTK-MRA/Normal<ID>-MRA.mha
├── Segmentation/IXI-Manual/IXI<ID>*-MRA.nii.gz
├── Segmentation/TubeTK-Manual/labels-<ID>.nii.gz
├── T1-BrainOnly/IXI-T1-BrainOnly/my_brain_<ID>*.nii.gz
└── T1-BrainOnly/TubeTK-T1-BrainOnly/output_Brain_Normal-<ID>_MRA*.nii.gz
```

The MRA and segmentation of a case must have matching voxel dimensions, because registration copies MRA spatial metadata to the segmentation before resampling.

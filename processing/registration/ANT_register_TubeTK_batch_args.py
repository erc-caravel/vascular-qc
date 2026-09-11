"""Register TubeTK MRA and vessel-segmentation volumes into padded MNI space.

The script supports the repository's flat ``Normal<ID>-MRA.mha`` layout and
writes paired MRA/segmentation NIfTI volumes to ``results/registration/tubetk``.
Run it with the project virtual environment; input and output paths are
available as command-line overrides.
"""

import os
import glob
import re
import argparse
import sys
from pathlib import Path

# Keep Matplotlib's cache inside the project instead of the user's protected
# profile directory.  ANTs imports Matplotlib as part of its dependency stack.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))

import SimpleITK as sitk
import ants

# Windows PowerShell sessions may still use a legacy code page.  The scripts
# print status symbols, so explicitly select UTF-8 when the stream supports it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------
# 1. DIRECTORIES / COMMAND-LINE ARGUMENTS
# ---------------------------------------------------------
base_path = Path(__file__).resolve().parents[2]

# Keep the current paths as defaults, but allow every path to be overridden
# from the command line.
parser = argparse.ArgumentParser(
    description="Batch rigid registration of TubeTK MRA data to MNI space via T1 brain images."
)
parser.add_argument(
    "--mni-path",
    default=os.path.join(base_path, 'data', 'MNI_Template', 'mni152.nii.gz'),
    help="Path to the MNI template NIfTI file."
)
parser.add_argument(
    "--t1-dir",
    default=os.path.join(base_path, 'data', 'T1-BrainOnly', 'TubeTK-T1-BrainOnly'),
    help="Directory containing TubeTK brain-only T1 images."
)
parser.add_argument(
    "--mra-dir",
    default=os.path.join(base_path, 'data', 'MRA', 'TubeTK-MRA'),
    help="Root directory containing the TubeTK MRA data."
)
parser.add_argument(
    "--seg-dir",
    default=os.path.join(base_path, 'data', 'Segmentation', 'TubeTK-Manual'),
    help="Directory containing TubeTK segmentation files."
)
parser.add_argument(
    "--out-dir",
    default=os.path.join(base_path, 'results', 'registration', 'tubetk'),
    help="Directory where registered outputs will be written."
)
args = parser.parse_args()

mni_path = args.mni_path
t1_dir = args.t1_dir
mra_dir = args.mra_dir
seg_dir = args.seg_dir
out_dir = args.out_dir

os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# 2. PREPARE THE MNI TEMPLATE (Done ONCE for the whole batch)
# ---------------------------------------------------------
print("\n--- Prepping Master Canvas ---")
if not os.path.exists(mni_path):
    print(f"⚠️ Missing MNI Template file at:\n   {mni_path}\nAborting.")
    exit()

temp_mni_path = os.path.join(out_dir, "temp_master_padded_mni.nii.gz")

# Pad the MNI Template so we don't cut off the neck
fixed_mni = sitk.ReadImage(mni_path, sitk.sitkFloat32)
padder = sitk.ConstantPadImageFilter()
padder.SetPadLowerBound([40, 40, 40])
padder.SetPadUpperBound([40, 40, 40])
expanded_mni_grid = padder.Execute(fixed_mni)
sitk.WriteImage(expanded_mni_grid, temp_mni_path)

# Load the padded template into ANTs once
fixed_ants_canvas = ants.image_read(temp_mni_path)
print("✅ Master MNI canvas padded and loaded into ANTs.")

# ---------------------------------------------------------
# 3. BATCH PROCESSING LOOP
# ---------------------------------------------------------
# Find all TubeTK T1 brain files
t1_brain_files = glob.glob(os.path.join(t1_dir, "output_Brain_Normal-*_MRA*.nii*"))
print(f"\n🚀 Found {len(t1_brain_files)} TubeTK T1 Brain files. Starting batch registration...\n")

for t1_path in t1_brain_files:
    filename = os.path.basename(t1_path)

    # Extract ID (e.g., '001' from 'output_Brain_Normal-001_MRA.nii.gz')
    match = re.search(r'output_Brain_Normal-(\d+)_MRA', filename)
    if not match:
        continue

    scan_id = match.group(1)

    # Locate corresponding full files.  Support both the original nested
    # TubeTK layout and the flat layout used by the supplied sample data.
    # Nested: TubeTK/Normal-001/MRA/*.mha
    # Flat:   TubeTK-MRA/Normal001-MRA.mha
    mra_match = glob.glob(os.path.join(mra_dir, f"Normal-{scan_id}", "MRA", "*.mha"))
    if not mra_match:
        mra_match = glob.glob(os.path.join(mra_dir, f"Normal{scan_id}-MRA*.mha"))

    # Segmentation: TubeTK_Segs/labels-001.nii.gz
    seg_match = glob.glob(os.path.join(seg_dir, f"labels-{scan_id}.nii*"))

    if not mra_match or not seg_match:
        print(f"⚠️ Missing Original MRA (.mha) or Seg match for ID Normal-{scan_id}. Skipping.")
        continue

    mra_path = mra_match[0]
    seg_path = seg_match[0]

    # Output Paths for this patient
    out_mra = os.path.join(out_dir, f"ANTs_MNI_Full_MRA_Normal-{scan_id}.nii.gz")
    out_seg = os.path.join(out_dir, f"ANTs_MNI_Seg_Normal-{scan_id}.nii.gz")
    temp_seg_path = os.path.join(out_dir, f"temp_fixed_seg_Normal-{scan_id}.nii.gz")

    # Skip if already fully processed
    if os.path.exists(out_mra) and os.path.exists(out_seg):
        print(f"⏭️ ID Normal-{scan_id} already exists. Skipping.")
        continue

    print(f"--- Processing ID: Normal-{scan_id} ---")

    try:
        # ---------------------------------------------------------
        # A. Read Sizes & Repair Segmentation Metadata (SimpleITK)
        # ---------------------------------------------------------
        moving_seg_raw = sitk.ReadImage(seg_path, sitk.sitkFloat32)
        moving_mra_full = sitk.ReadImage(mra_path, sitk.sitkFloat32)

        # Get and print dimensions (X, Y, Z) - Z is typically the number of slices
        mra_size = moving_mra_full.GetSize()
        seg_size = moving_seg_raw.GetSize()
        print(f"    > Input MRA Volume Size: {mra_size} (Slices: {mra_size[2]})")
        print(f"    > Input Seg Volume Size: {seg_size} (Slices: {seg_size[2]})")

        # Copy spatial info from MRA to Seg to ensure perfect alignment
        moving_seg_raw.CopyInformation(moving_mra_full)
        moving_seg = sitk.BinaryThreshold(
            moving_seg_raw, lowerThreshold=0.1, upperThreshold=9999.0, insideValue=1, outsideValue=0
        )
        moving_seg = sitk.Cast(moving_seg, sitk.sitkUInt8)
        sitk.WriteImage(moving_seg, temp_seg_path)

        # ---------------------------------------------------------
        # B. Calculate Two-Step Transforms
        # ---------------------------------------------------------
        moving_t1_brain = ants.image_read(t1_path)
        moving_full_ants = ants.image_read(mra_path)
        moving_seg_ants = ants.image_read(temp_seg_path)

        print("    > 1. Calculating MRA -> T1 Brain Rigid Transform...")
        tx_mra_to_t1 = ants.registration(
            fixed=moving_t1_brain,
            moving=moving_full_ants,
            type_of_transform='Rigid'
        )

        print("    > 2. Calculating T1 Brain -> MNI Rigid Transform...")
        tx_t1_to_mni = ants.registration(
            fixed=fixed_ants_canvas,
            moving=moving_t1_brain,
            type_of_transform='Rigid'
        )

        # ---------------------------------------------------------
        # C. Combine & Apply Transforms to Full MRA & Seg
        # ---------------------------------------------------------
        # ANTs processes lists from left to right (Outermost transform first)
        combined_transforms = tx_t1_to_mni['fwdtransforms'] + tx_mra_to_t1['fwdtransforms']

        print("    > 3. Resampling Full MRA to MNI (Single Pass)...")
        registered_mra = ants.apply_transforms(
            fixed=fixed_ants_canvas,
            moving=moving_full_ants,
            transformlist=combined_transforms,
            interpolator='linear'
        )
        ants.image_write(registered_mra, out_mra)

        print("    > 4. Resampling Segmentation to MNI (Single Pass)...")
        registered_seg = ants.apply_transforms(
            fixed=fixed_ants_canvas,
            moving=moving_seg_ants,
            transformlist=combined_transforms,
            interpolator='nearestNeighbor'
        )
        ants.image_write(registered_seg, out_seg)

        print(f"✅ Finished ID Normal-{scan_id}\n")

    except Exception as e:
        print(f"❌ Failed ID Normal-{scan_id}: {e}\n")

    finally:
        # Clean up the patient-specific temporary segmentation file
        if os.path.exists(temp_seg_path):
            os.remove(temp_seg_path)

# ---------------------------------------------------------
# 4. FINAL CLEANUP
# ---------------------------------------------------------
# Remove the master padded MNI template now that the loop is totally finished
if os.path.exists(temp_mni_path):
    os.remove(temp_mni_path)

print("🎯 All TubeTK scans have been successfully aligned to MNI space using ANTs!")

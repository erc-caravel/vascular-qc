"""Create multi-slab TubeTK straight-sinus augmentation images.

Standardized TubeTK MIPs select the cases to augment; registered volumes in
``results/registration/tubetk`` provide the source data.  Three 448-pixel
sagittal variants, metadata JSON files, and paired axial QC images are written.
Run it without arguments after registration and MIP generation have completed.
"""

import os
import glob
import re
import json
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(base_path / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


# ==========================================
# 1. AUTO-ROTATION FUNCTION (For Sagittal)
# ==========================================
def get_auto_rotation_angle(image_2d):
    """Calculates the exact tilt of the scanner box using PCA."""
    mask = image_2d > 1e-3
    if not np.any(mask):
        return 0.0

    y, x = np.nonzero(mask)
    cov_matrix = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]
    angle_deg = np.degrees(np.arctan2(major_axis[1], major_axis[0]))

    correction_angle = angle_deg
    correction_angle = (correction_angle + 45) % 90 - 45

    return float(correction_angle)


# ==========================================
# 2. SETUP DIRECTORIES & BATCH PARAMETERS
# ==========================================
input_dir = base_path / 'results' / 'registration' / 'tubetk'
reference_pattern = base_path / 'results' / 'mips' / 'tubetk' / '*' / '2_Sagittal_Crops' / 'MIPs_Mid_14' / 'Plain' / 'TubeTK_*.png'

# Output directories updated to denote plain/grayscale
axial_out_dir = base_path / 'results' / 'augmentation' / 'tubetk' / 'straight_sinus' / 'axial_qc'
sagittal_out_dir = base_path / 'results' / 'augmentation' / 'tubetk' / 'straight_sinus' / 'plain'

os.makedirs(axial_out_dir, exist_ok=True)
os.makedirs(sagittal_out_dir, exist_ok=True)

lr_axis = 0
si_axis = 2
internvl_canvas_size = 448
fallback_threshold_slices = 8

# ⚙️ MULTI-CONFIG SETTINGS (No Tilts)
# Format: ("Name", offset_left, offset_right) -> Relative to the calculated center
slice_configs = [
    ("Mid_14", -7, 7),  # 14 total
    ("Mid_10", -5, 5),  # 10 total
    ("Mid_8", -4, 4),   # 8 total
]

# ==========================================
# 3. LOCATE TARGET IDs FROM REFERENCE FOLDER
# ==========================================
reference_files = glob.glob(str(reference_pattern))
target_cases = {}

for ref_path in reference_files:
    match = re.search(r'^TubeTK_.+_(\d+)\.png$', os.path.basename(ref_path))
    if match:
        target_cases[match.group(1)] = os.path.splitext(os.path.basename(ref_path))[0]

target_ids = sorted(target_cases)

if not target_ids:
    print(f"⚠️ No matching TubeTK IDs found in: {reference_pattern}")
    exit()

print(f"🚀 Found {len(target_ids)} target IDs to process.")
print(f"   Generating {len(slice_configs)} plain configurations per scan (Total: {len(target_ids) * len(slice_configs)} images).")
print(f"   Order of Operations: SLICE ONLY (No Tilting applied).")
print(f"   Outputting to: {sagittal_out_dir} and {axial_out_dir}\n")

# ==========================================
# 4. BATCH PROCESSING LOOP
# ==========================================
for scan_id in target_ids:
    case_name = target_cases[scan_id]
    mra_path = input_dir / f"ANTs_MNI_Full_MRA_Normal-{scan_id}.nii.gz"
    seg_path = input_dir / f"ANTs_MNI_Seg_Normal-{scan_id}.nii.gz"

    if not os.path.exists(mra_path) or not os.path.exists(seg_path):
        print(f"    ❌ Error: Missing MRA or Segmentation file for ID {scan_id}. Skipping.")
        continue

    print(f"--- Processing ID: {scan_id} ---")

    try:
        # Load Base Data ONCE per ID
        img = nib.load(mra_path)
        seg = nib.load(seg_path)
        img_data_orig = img.get_fdata()
        seg_data_orig = seg.get_fdata()
        aspect_ratio = img.header.get_zooms()[1] / img.header.get_zooms()[0]

        total_width_lr = img_data_orig.shape[lr_axis]
        mni_center_lr_orig = total_width_lr // 2

        # ─── STEP 1: FIND CENTER ON THE ORIGINAL (STRAIGHT) HEAD ───
        mask_binary_orig = (seg_data_orig > 0)
        if not np.any(mask_binary_orig):
            com_center_lr = mni_center_lr_orig
        else:
            com_coords = ndimage.center_of_mass(mask_binary_orig)
            com_center_lr = int(np.round(com_coords[lr_axis]))

        drift_diff = abs(mni_center_lr_orig - com_center_lr)

        if drift_diff <= fallback_threshold_slices:
            center_lr = mni_center_lr_orig
            center_choice_str = "MNI_Center"
        else:
            center_lr = com_center_lr
            center_choice_str = "CenterOfMass_Fallback"
            
        # Get the PCA auto-angle for the sagittal canvas uprighting
        mip_background_sag = img_data_orig.max(axis=lr_axis)
        auto_angle = get_auto_rotation_angle(mip_background_sag.T)

        # ─── STEP 2: RENDER CONFIGURATIONS ───
        for config_name, offset_left, offset_right in slice_configs:
            
            # Keep the standardized TubeTK_<model>_<id> source stem.
            config_lower = config_name.lower()
            sagittal_png = os.path.join(sagittal_out_dir, f"{case_name}_sagittal_mip_{config_lower}_plain.png")
            axial_png = os.path.join(axial_out_dir, f"{case_name}_axial_qc_{config_lower}_plain.png")
            output_json = os.path.join(sagittal_out_dir, f"{case_name}_sagittal_mip_{config_lower}_plain_meta.json")

            if os.path.exists(sagittal_png) and os.path.exists(axial_png):
                continue

            start_lr = max(0, center_lr + offset_left)
            end_lr = min(total_width_lr, center_lr + offset_right)
            crop_width_lr = end_lr - start_lr

            # Isolate the targeted slice slab
            mask_center = np.zeros_like(seg_data_orig)
            mask_center[start_lr:end_lr, :, :] = (seg_data_orig[start_lr:end_lr, :, :] > 0)
            center_vessels_orig = img_data_orig * mask_center

            # ==========================================
            # PIPELINE A: SAGITTAL PLAIN
            # ==========================================
            mip_center_sag = center_vessels_orig.max(axis=lr_axis)

            vessels_rotated = ndimage.rotate(mip_center_sag.T, auto_angle, reshape=False, order=1)
            vessels_rotated = np.rot90(vessels_rotated, k=2)
            vessels_rotated = np.fliplr(vessels_rotated)

            h, w = vessels_rotated.shape
            canvas_vessels = np.zeros((internvl_canvas_size, internvl_canvas_size), dtype=vessels_rotated.dtype)

            y_off = (internvl_canvas_size - h) // 2
            x_off = (internvl_canvas_size - w) // 2
            canvas_vessels[y_off:y_off + h, x_off:x_off + w] = vessels_rotated

            # Save pure grayscale canvas
            plt.imsave(sagittal_png, canvas_vessels, cmap='gray')

            # Metadata
            metadata = {
                "scan_id": scan_id,
                "tilt_angle_applied": 0.0,
                "slice_configuration": config_name,
                "crop_width_lr": int(crop_width_lr),
                "center_slice_lr": int(center_lr),
                "center_choice_method": center_choice_str,
                "center_drift_slices": int(drift_diff),
                "applied_pca_angle_degrees": auto_angle,
                "applied_180_rotation": True,
                "applied_fliplr": True,
                "canvas_size_yx": [internvl_canvas_size, internvl_canvas_size],
                "padding_offsets_yx": [int(y_off), int(x_off)],
                "original_image_yx": [h, w]
            }
            with open(output_json, 'w') as f:
                json.dump(metadata, f, indent=4)

            # ==========================================
            # PIPELINE B: AXIAL QC PLAIN
            # ==========================================
            mip_center_vessels_ax = center_vessels_orig.max(axis=si_axis)
            mip_center_vessels_ax_flipped = np.fliplr(mip_center_vessels_ax.T)

            fig_ax = plt.figure(figsize=(8, 8), facecolor='black')

            # Plot only the vessels in standard grayscale
            plt.imshow(mip_center_vessels_ax_flipped, cmap='gray', origin='lower', aspect=aspect_ratio)

            # FLIPPED QC LINES
            flipped_start_lr = total_width_lr - end_lr
            flipped_end_lr = total_width_lr - start_lr

            plt.axvline(x=flipped_start_lr, color='yellow', linestyle='--', alpha=0.5, linewidth=1)
            plt.axvline(x=flipped_end_lr, color='yellow', linestyle='--', alpha=0.5, linewidth=1)

            plt.title(f"ID: {scan_id} | Slices: {config_name} (PLAIN/FLIPPED)", color='white', pad=15)
            plt.axis('off')
            plt.tight_layout()

            fig_ax.savefig(axial_png, facecolor=fig_ax.get_facecolor(), edgecolor='none', dpi=150)
            plt.close(fig_ax)

        print(f"    ✅ Finished {len(slice_configs)} configs for ID {scan_id}")

    except Exception as e:
        print(f"    ❌ Failed processing ID {scan_id}: {e}\n")

print("\n🎯 Batch Processing Complete!")

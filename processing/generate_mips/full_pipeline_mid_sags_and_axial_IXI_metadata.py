"""Generate standardized IXI MIPs and cross-view metadata.

This self-contained pipeline registers each IXI case, creates sagittal,
posterior axial, and anterior coronal MIPs, and stores their geometry metadata.
Usage: ``python processing/generate_mips/full_pipeline_mid_sags_and_axial_IXI_metadata.py
--input_dir data --output_dir results/mips/ixi --model_name <name>``.
"""

import os
import glob
import re
import argparse
import shutil
import sys
from pathlib import Path

# Keep dependency caches in the project on managed Windows machines.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))
os.environ.setdefault("MPLBACKEND", "Agg")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import ants
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import ndimage
import json


# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def get_auto_rotation_angle(image_2d):
    """
    Calculates the tilt of the scanner/head bounding structure
    using PCA.

    Returns:
        float: Rotation angle in degrees.
    """
    mask = image_2d > 1e-3

    if not np.any(mask):
        return 0.0

    y, x = np.nonzero(mask)

    cov_matrix = np.cov(x, y)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    major_axis = eigenvectors[:, np.argmax(eigenvalues)]

    angle_deg = np.degrees(
        np.arctan2(major_axis[1], major_axis[0])
    )

    correction_angle = (angle_deg + 45) % 90 - 45

    return float(correction_angle)


def save_plain_image(image, output_path, aspect=1.0):
    """
    Save an isolated grayscale image on a black background.
    """
    fig = plt.figure(
        figsize=(10.24, 10.24),
        dpi=100,
        facecolor="black"
    )

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.imshow(
        image,
        cmap="gray",
        origin="lower",
        aspect=aspect
    )

    fig.savefig(
        output_path,
        facecolor="black",
        edgecolor="none"
    )

    plt.close(fig)


def save_color_overlay(background, vessels, output_path, aspect=1.0):
    """
    Save grayscale background with vessel overlay in black-to-red.
    """
    black_to_red = mcolors.LinearSegmentedColormap.from_list(
        "black_red",
        ["black", "red"]
    )

    fig = plt.figure(
        figsize=(10.24, 10.24),
        dpi=100,
        facecolor="black"
    )

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    masked_background = np.ma.masked_where(
        background < 1e-3,
        background
    )

    masked_vessels = np.ma.masked_where(
        vessels < 1e-3,
        vessels
    )

    ax.imshow(
        masked_background,
        cmap="gray",
        origin="lower",
        aspect=aspect
    )

    ax.imshow(
        masked_vessels,
        cmap=black_to_red,
        origin="lower",
        aspect=aspect
    )

    fig.savefig(
        output_path,
        facecolor="black",
        edgecolor="none"
    )

    plt.close(fig)


def center_image_on_canvas(image, canvas_size=448):
    """
    Center a 2D image on a square canvas.
    """
    h, w = image.shape

    canvas = np.zeros(
        (canvas_size, canvas_size),
        dtype=image.dtype
    )

    y_offset = (canvas_size - h) // 2
    x_offset = (canvas_size - w) // 2

    # Handle images larger than the canvas defensively.
    y_start_src = max(0, -y_offset)
    x_start_src = max(0, -x_offset)

    y_start_dst = max(0, y_offset)
    x_start_dst = max(0, x_offset)

    copy_h = min(
        h - y_start_src,
        canvas_size - y_start_dst
    )

    copy_w = min(
        w - x_start_src,
        canvas_size - x_start_dst
    )

    if copy_h > 0 and copy_w > 0:
        canvas[
            y_start_dst:y_start_dst + copy_h,
            x_start_dst:x_start_dst + copy_w
        ] = image[
            y_start_src:y_start_src + copy_h,
            x_start_src:x_start_src + copy_w
        ]

    return canvas


# ============================================================
# 2. POSTERIOR EXTRACTION
# ============================================================

def generate_posterior(
    img_data,
    seg_data,
    zooms,
    output_path,
    distance_from_back=40,
    angle_degrees=60
):
    LR_AXIS = 0
    AP_AXIS = 1
    SI_AXIS = 2

    aspect_axial = zooms[AP_AXIS] / zooms[LR_AXIS]

    # 1: AUTOMATED ROTATIONAL CORRECTION
    mip_background_sag = img_data.max(axis=LR_AXIS)
    auto_angle = get_auto_rotation_angle(mip_background_sag.T)
    correction_angle = -auto_angle

    img_data_corrected = ndimage.rotate(
        img_data, angle=correction_angle, axes=(AP_AXIS, SI_AXIS), reshape=True, order=1
    )

    seg_data_corrected = ndimage.rotate(
        seg_data, angle=correction_angle, axes=(AP_AXIS, SI_AXIS), reshape=True, order=0
    ) > 0.5
    seg_data_corrected = seg_data_corrected.astype(bool)

    # 2: FIND POSTERIOR ANCHOR USING OTSU
    mip_sag_raw = img_data_corrected.max(axis=LR_AXIS)
    mip_norm = cv2.normalize(
        mip_sag_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    otsu_thresh_val, head_mask_2d = cv2.threshold(
        mip_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        head_mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError("No head contour found during posterior extraction.")

    largest_contour = max(contours, key=cv2.contourArea)
    head_mask_clean = np.zeros_like(head_mask_2d)
    cv2.drawContours(head_mask_clean, [largest_contour], -1, 255, thickness=cv2.FILLED)

    ap_coords, si_coords = np.where(head_mask_clean > 0)

    if len(ap_coords) == 0:
        raise RuntimeError("Head mask is empty during posterior extraction.")

    si_min = si_coords.min()
    si_max = si_coords.max()
    si_mid = (si_min + si_max) / 2.0

    idx = np.argmin(ap_coords)
    ap_min_posterior = ap_coords[idx]

    anchor_ap = ap_min_posterior + distance_from_back
    anchor_si = si_mid

    # 3: CREATE 3D POSTERIOR CUTTING PLANE
    slope_m = np.tan(np.radians(angle_degrees))
    _, grid_ap, grid_si = np.indices(img_data_corrected.shape)

    above_line_mask = (grid_si > (slope_m * (grid_ap - anchor_ap) + anchor_si))
    roi_mask = (seg_data_corrected & above_line_mask)
    roi_vessels = (img_data_corrected * roi_mask)

    # 4: AXIAL MIP OF ISOLATED POSTERIOR VESSELS
    roi_mip_ax = roi_vessels.max(axis=SI_AXIS)
    roi_mip_ax_disp = np.fliplr(roi_mip_ax.T)

    save_plain_image(roi_mip_ax_disp, output_path, aspect=aspect_axial)

    # RETURN METADATA FOR YOLO QC
    return {
        "canvas": 1024.0,
        "h": int(roi_mip_ax_disp.shape[0]),
        "w": int(roi_mip_ax_disp.shape[1]),
        "aspect": float(aspect_axial)
    }


# ============================================================
# 3. ANTERIOR EXTRACTION
# ============================================================

def generate_anterior(
    img_data,
    seg_data,
    zooms,
    output_paths,
    distance_from_right=20,
    angle_degrees=55,
    forehead_search_ratio=0.45
):
    LR_AXIS = 0
    AP_AXIS = 1
    SI_AXIS = 2

    aspect_sagittal = zooms[SI_AXIS] / zooms[AP_AXIS]
    aspect_axial = zooms[AP_AXIS] / zooms[LR_AXIS]
    aspect_coronal = zooms[SI_AXIS] / zooms[LR_AXIS]

    black_to_red = mcolors.LinearSegmentedColormap.from_list("black_red", ["black", "red"])

    # 1: AUTOMATED ROTATIONAL CORRECTION
    mip_background_sag = img_data.max(axis=LR_AXIS)
    auto_angle = get_auto_rotation_angle(mip_background_sag.T)
    correction_angle = -auto_angle

    img_data_corrected = ndimage.rotate(
        img_data, angle=correction_angle, axes=(AP_AXIS, SI_AXIS), reshape=True, order=1
    )
    seg_data_corrected = ndimage.rotate(
        seg_data, angle=correction_angle, axes=(AP_AXIS, SI_AXIS), reshape=True, order=0
    ) > 0.5
    seg_data_corrected = seg_data_corrected.astype(bool)

    # 2: OTSU HEAD MASK
    mip_sag_raw = img_data_corrected.max(axis=LR_AXIS)
    mip_norm = cv2.normalize(mip_sag_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    otsu_thresh_val, head_mask_2d = cv2.threshold(mip_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(head_mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No head contour found during anterior extraction.")

    largest_contour = max(contours, key=cv2.contourArea)
    head_mask_clean = np.zeros_like(head_mask_2d)
    cv2.drawContours(head_mask_clean, [largest_contour], -1, 255, thickness=cv2.FILLED)

    ap_coords, si_coords = np.where(head_mask_clean > 0)
    if len(ap_coords) == 0:
        raise RuntimeError("Head mask is empty during anterior extraction.")

    si_min = si_coords.min()
    si_max = si_coords.max()
    si_mid = (si_min + si_max) / 2.0

    # 3: FIND FOREHEAD
    forehead_si_limit = si_max - (si_max - si_min) * forehead_search_ratio
    upper_mask = (si_coords > forehead_si_limit)

    if np.any(upper_mask):
        idx = np.argmax(ap_coords[upper_mask])
        ap_max_forehead = ap_coords[upper_mask][idx]
        actual_forehead_si = si_coords[upper_mask][idx]
    else:
        idx = np.argmax(ap_coords)
        ap_max_forehead = ap_coords[idx]
        actual_forehead_si = si_coords[idx]

    anchor_ap = ap_max_forehead - distance_from_right
    anchor_si = si_mid
    forehead_intensity = mip_sag_raw[ap_max_forehead, actual_forehead_si]

    # 4: CREATE ANTERIOR 3D CUTTING PLANE
    slope_m = -np.tan(np.radians(angle_degrees))
    _, grid_ap, grid_si = np.indices(img_data_corrected.shape)

    above_line_mask = (grid_si > (slope_m * (grid_ap - anchor_ap) + anchor_si))
    roi_mask = (seg_data_corrected & above_line_mask)
    roi_vessels = (img_data_corrected * roi_mask)

    # 5: SAGITTAL DEBUG
    roi_mip_sag = roi_vessels.max(axis=LR_AXIS)
    bg_mip_sag_disp = np.rot90(mip_sag_raw)
    roi_mip_sag_disp = np.rot90(roi_mip_sag)
    head_mask_sag_disp = np.rot90(head_mask_clean)
    roi_sag_transparent = np.ma.masked_where(roi_mip_sag_disp < 1e-3, roi_mip_sag_disp)

    N_ap, N_si = mip_sag_raw.shape
    plot_anchor_x = anchor_ap
    plot_forehead_x = ap_max_forehead
    plot_anchor_y = N_si - 1 - anchor_si
    plot_forehead_y = N_si - 1 - actual_forehead_si
    line_x = np.array([0, N_ap])
    line_si = slope_m * (line_x - anchor_ap) + anchor_si
    line_y = N_si - 1 - line_si

    fig_sag = plt.figure(figsize=(10.24, 10.24), dpi=100, facecolor="black")
    ax_sag = fig_sag.add_axes([0, 0, 1, 1])
    ax_sag.axis("off")
    ax_sag.imshow(bg_mip_sag_disp, cmap="gray", aspect=aspect_sagittal)
    ax_sag.imshow(roi_sag_transparent, cmap=black_to_red, aspect=aspect_sagittal)
    ax_sag.contour(head_mask_sag_disp, levels=[0.5], colors="magenta", linewidths=0.8, linestyles="dashed")
    ax_sag.plot(line_x, line_y, color="yellow", linewidth=1)
    ax_sag.plot([plot_forehead_x, plot_anchor_x], [plot_anchor_y, plot_anchor_y], color="white", linewidth=1)
    ax_sag.plot(plot_forehead_x, plot_forehead_y, marker="o", color="blue", markersize=4)
    ax_sag.plot(plot_forehead_x, plot_anchor_y, marker="o", color="red", markersize=4)
    ax_sag.plot(plot_anchor_x, plot_anchor_y, marker="o", color="lime", markersize=4)
    text_x = plot_forehead_x + 5
    text_y = plot_forehead_y - 15

    ax_sag.text(
        text_x, text_y, (f"Intensity: {forehead_intensity:.1f}\nOtsu Thresh: {otsu_thresh_val}"),
        color="cyan", fontsize=9, fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2)
    )
    fig_sag.savefig(output_paths["sagittal_debug"], facecolor="black", edgecolor="none")
    plt.close(fig_sag)

    # 6: AXIAL ISOLATED
    roi_mip_ax = roi_vessels.max(axis=SI_AXIS)
    roi_mip_ax_disp = np.fliplr(roi_mip_ax.T)
    save_plain_image(roi_mip_ax_disp, output_paths["axial"], aspect=aspect_axial)

    # 7: CORONAL ISOLATED
    roi_mip_cor = roi_vessels.max(axis=AP_AXIS)
    
    # Transpose fixes the up/down flip caused by origin="lower" 
    roi_mip_cor_disp = roi_mip_cor.T 
    
    save_plain_image(roi_mip_cor_disp, output_paths["coronal"], aspect=aspect_coronal)

    # RETURN METADATA FOR YOLO QC
    return {
        "canvas": 1024.0,
        "h": int(roi_mip_cor_disp.shape[0]),
        "w": int(roi_mip_cor_disp.shape[1]),
        "aspect": float(aspect_coronal)
    }


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="IXI MRA registration, sagittal crop, posterior and anterior extraction pipeline."
    )
    parser.add_argument("--input_dir", required=True, help="Base directory containing MNI, T1, MRA and segmentation data.")
    parser.add_argument("--output_dir", required=True, help="Base directory for all pipeline outputs.")
    parser.add_argument("--model_name", required=True, help="Segmentation model name. Used in the standardized output filename.")
    args = parser.parse_args()

    # ========================================================
    # INPUT PATHS (Configured for IXI dataset)
    # ========================================================
    mni_path = os.path.join(args.input_dir, "MNI_Template", "mni152.nii.gz")
    t1_dir = os.path.join(args.input_dir, "T1-BrainOnly", "IXI-T1-BrainOnly")
    mra_dir = os.path.join(args.input_dir, "MRA", "IXI-MRA")
    seg_dir = os.path.join(args.input_dir, "Segmentation", "IXI-Manual")

    # ========================================================
    # OUTPUT PATHS
    # ========================================================
    shared_dir = os.path.join(args.output_dir, "Shared_Registrations")
    trans_dir = os.path.join(shared_dir, "Transforms")
    mra_out_dir = os.path.join(shared_dir, "Registered_MRAs")

    model_dir = os.path.join(args.output_dir, args.model_name)
    seg_out_dir = os.path.join(model_dir, "1_Registered_Segs")
    sagittal_out_dir = os.path.join(model_dir, "2_Sagittal_Crops")
    posterior_out_dir = os.path.join(model_dir, "3_Posterior")
    anterior_out_dir = os.path.join(model_dir, "4_Anterior")

    # ========================================================
    # CREATE OUTPUT DIRECTORIES
    # ========================================================
    sagittal_configs = ["MIPs_Mid_14", "MIPs_Mid_32"]
    for config_name in sagittal_configs:
        os.makedirs(os.path.join(sagittal_out_dir, config_name, "Color"), exist_ok=True)
        os.makedirs(os.path.join(sagittal_out_dir, config_name, "Plain"), exist_ok=True)

    os.makedirs(posterior_out_dir, exist_ok=True)

    anterior_sag_dir = os.path.join(anterior_out_dir, "Sagittal_Debug")
    anterior_ax_dir = os.path.join(anterior_out_dir, "Axial_Isolated")
    anterior_cor_dir = os.path.join(anterior_out_dir, "Coronal_Isolated")

    for d in [trans_dir, mra_out_dir, seg_out_dir, posterior_out_dir, anterior_sag_dir, anterior_ax_dir, anterior_cor_dir]:
        os.makedirs(d, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"INITIALIZING IXI PIPELINE FOR MODEL: {args.model_name}")
    print("=" * 70)

    if not os.path.exists(mni_path):
        print(f"⚠️ Missing MNI Template: {mni_path}")
        return

    # ========================================================
    # PREPARE PADDED MNI TEMPLATE
    # ========================================================
    temp_mni_path = os.path.join(shared_dir, "temp_master_padded_mni.nii.gz")
    fixed_mni = sitk.ReadImage(mni_path, sitk.sitkFloat32)
    padder = sitk.ConstantPadImageFilter()
    padder.SetPadLowerBound([40, 40, 40])
    padder.SetPadUpperBound([40, 40, 40])
    expanded_mni_grid = padder.Execute(fixed_mni)
    sitk.WriteImage(expanded_mni_grid, temp_mni_path)
    
    fixed_ants_canvas = ants.image_read(temp_mni_path)
    print("✅ Master MNI canvas padded and loaded into ANTs.\n")

    # ========================================================
    # FIND INPUT SCANS (IXI PATTERNS)
    # ========================================================
    t1_brain_files = glob.glob(os.path.join(t1_dir, "my_brain_*.nii*"))
    print(f"🚀 Found {len(t1_brain_files)} IXI scans.\n")

    # ========================================================
    # BATCH PROCESSING
    # ========================================================
    for t1_path in t1_brain_files:
        match = re.search(r"my_brain_(\d+)", os.path.basename(t1_path))
        if not match:
            continue
        scan_id = match.group(1)

        # Added a * after MRA to catch _0000.nii.gz variations
        mra_match = glob.glob(os.path.join(mra_dir, f"IXI{scan_id}*-MRA*.nii*"))
        seg_match = glob.glob(os.path.join(seg_dir, f"IXI{scan_id}*-MRA*.nii*"))

        if not mra_match or not seg_match:
            print(f"⚠️ Missing Original MRA or Segmentation for IXI ID: {scan_id}. Skipping.")
            continue

        mra_path = mra_match[0]
        seg_path = seg_match[0]

        # IXI Standardized Filename
        standardized_name = f"IXI_{args.model_name}_{scan_id}.png"
        meta_filename = standardized_name.replace(".png", "_meta.json")
        meta_path = os.path.join(sagittal_out_dir, "MIPs_Mid_32", "Plain", meta_filename)

        print("\n" + "=" * 70)
        print(f"PROCESSING: IXI ID {scan_id}")
        print(f"OUTPUT NAME: {standardized_name}")
        print("=" * 70)

        # =================================================
        # RESUME LOGIC CHECK
        # =================================================
        if os.path.exists(meta_path):
            print("    ⏭️ Final metadata file exists. Skipping this scan to pick up where left off.")
            continue

        try:
            # =================================================
            # STEP 1: REGISTER MRA
            # =================================================
            t1_to_mni_mat = os.path.join(trans_dir, f"T1_to_MNI_{scan_id}.mat")
            mra_to_t1_mat = os.path.join(trans_dir, f"MRA_to_T1_{scan_id}.mat")
            reg_mra_path = os.path.join(mra_out_dir, f"ANTs_MNI_MRA_{scan_id}.nii.gz")

            moving_full_ants = ants.image_read(mra_path)

            if not (os.path.exists(t1_to_mni_mat) and os.path.exists(mra_to_t1_mat) and os.path.exists(reg_mra_path)):
                print("    > Calculating ANTs rigid transforms...")
                moving_t1_brain = ants.image_read(t1_path)
                tx_mra_to_t1 = ants.registration(fixed=moving_t1_brain, moving=moving_full_ants, type_of_transform="Rigid")
                tx_t1_to_mni = ants.registration(fixed=fixed_ants_canvas, moving=moving_t1_brain, type_of_transform="Rigid")

                shutil.copy(tx_t1_to_mni["fwdtransforms"][0], t1_to_mni_mat)
                shutil.copy(tx_mra_to_t1["fwdtransforms"][0], mra_to_t1_mat)

                combined_transforms = [t1_to_mni_mat, mra_to_t1_mat]
                registered_mra = ants.apply_transforms(fixed=fixed_ants_canvas, moving=moving_full_ants, transformlist=combined_transforms, interpolator="linear")
                ants.image_write(registered_mra, reg_mra_path)
                print("    ✅ MRA registered.")
            else:
                print("    ⏭️ Existing MRA registration found.")
                combined_transforms = [t1_to_mni_mat, mra_to_t1_mat]

            # =================================================
            # STEP 2: REGISTER SEGMENTATION
            # =================================================
            reg_seg_path = os.path.join(seg_out_dir, f"ANTs_MNI_Seg_{scan_id}.nii.gz")
            if not os.path.exists(reg_seg_path):
                print("    > Registering segmentation...")
                moving_seg_raw = sitk.ReadImage(seg_path, sitk.sitkFloat32)
                moving_mra_full = sitk.ReadImage(mra_path, sitk.sitkFloat32)
                moving_seg_raw.CopyInformation(moving_mra_full)

                moving_seg = sitk.BinaryThreshold(moving_seg_raw, lowerThreshold=0.1, upperThreshold=9999.0, insideValue=1, outsideValue=0)
                moving_seg = sitk.Cast(moving_seg, sitk.sitkUInt8)
                
                temp_seg_path = os.path.join(seg_out_dir, f"temp_seg_{scan_id}.nii.gz")
                sitk.WriteImage(moving_seg, temp_seg_path)

                moving_seg_ants = ants.image_read(temp_seg_path)
                registered_seg = ants.apply_transforms(fixed=fixed_ants_canvas, moving=moving_seg_ants, transformlist=combined_transforms, interpolator="nearestNeighbor")
                ants.image_write(registered_seg, reg_seg_path)

                if os.path.exists(temp_seg_path):
                    os.remove(temp_seg_path)
                print("    ✅ Segmentation registered.")
            else:
                print("    ⏭️ Existing segmentation registration found.")

            # =================================================
            # STEP 3: LOAD REGISTERED VOLUMES
            # =================================================
            print("    > Loading registered 3D volumes...")
            img = nib.load(reg_mra_path)
            seg = nib.load(reg_seg_path)
            
            img_data = img.get_fdata()
            seg_data = seg.get_fdata()
            zooms = img.header.get_zooms()

            LR_AXIS, AP_AXIS, SI_AXIS = 0, 1, 2

            # =================================================
            # STEP 4: SAGITTAL CROPS
            # =================================================
            print("    > Generating sagittal crops...")
            total_width_lr = img_data.shape[LR_AXIS]
            mni_center_lr = total_width_lr // 2

            mip_background_sag = img_data.max(axis=LR_AXIS)
            auto_angle = get_auto_rotation_angle(mip_background_sag.T)
            mask_binary = (seg_data > 0)

            if np.any(mask_binary):
                com_center_lr = int(np.round(ndimage.center_of_mass(mask_binary)[LR_AXIS]))
            else:
                com_center_lr = mni_center_lr

            if abs(mni_center_lr - com_center_lr) <= 10:
                center_lr = mni_center_lr
            else:
                center_lr = com_center_lr

            slice_configs = [("MIPs_Mid_14", -7, 7), ("MIPs_Mid_32", -16, 16)]
            
            # Removed np.fliplr() here to horizontally flip the background
            bg_rotated_sag = np.rot90(ndimage.rotate(mip_background_sag.T, auto_angle, reshape=False, order=1), k=2)

            sag_meta_data = {}

            for config_name, offset_left, offset_right in slice_configs:
                col_dir = os.path.join(sagittal_out_dir, config_name, "Color")
                pln_dir = os.path.join(sagittal_out_dir, config_name, "Plain")

                start_lr = max(0, center_lr + offset_left)
                end_lr = min(total_width_lr, center_lr + offset_right)

                mask_center = np.zeros_like(seg_data)
                mask_center[start_lr:end_lr, :, :] = (seg_data[start_lr:end_lr, :, :] > 0)
                center_vessels = img_data * mask_center

                mip_center_sag = center_vessels.max(axis=LR_AXIS)
                
                # Removed np.fliplr() here to horizontally flip the vessels
                vessels_rot = np.rot90(ndimage.rotate(mip_center_sag.T, auto_angle, reshape=False, order=1), k=2)

                cv_bg = center_image_on_canvas(bg_rotated_sag, 448)
                cv_ves = center_image_on_canvas(vessels_rot, 448)

                # Capture metadata for the main sagittal crop (MIPs_Mid_32)
                if config_name == "MIPs_Mid_32":
                    h_rot, w_rot = vessels_rot.shape
                    sag_meta_data = {
                        "canvas": 448.0,
                        "h": int(h_rot),
                        "w": int(w_rot),
                        "offset_y": int(max(0, (448 - h_rot) // 2)),
                        "offset_x": int(max(0, (448 - w_rot) // 2))
                    }

                plain_path = os.path.join(pln_dir, standardized_name)
                color_path = os.path.join(col_dir, standardized_name)

                plt.imsave(plain_path, cv_ves, cmap="gray")
                save_color_overlay(cv_bg, cv_ves, color_path, aspect=1.0)

            print("    ✅ Sagittal crops generated.")

            # =================================================
            # STEP 5: POSTERIOR
            # =================================================
            print("    > Generating posterior extraction...")
            posterior_path = os.path.join(posterior_out_dir, standardized_name)
            ax_meta_data = generate_posterior(img_data=img_data, seg_data=seg_data, zooms=zooms, output_path=posterior_path, distance_from_back=40, angle_degrees=60)
            print("    ✅ Posterior generated.")

            # =================================================
            # STEP 6: ANTERIOR
            # =================================================
            print("    > Generating anterior extraction...")
            anterior_paths = {
                "sagittal_debug": os.path.join(anterior_sag_dir, standardized_name),
                "axial": os.path.join(anterior_ax_dir, standardized_name),
                "coronal": os.path.join(anterior_cor_dir, standardized_name)
            }
            cor_meta_data = generate_anterior(img_data=img_data, seg_data=seg_data, zooms=zooms, output_paths=anterior_paths, distance_from_right=20, angle_degrees=55, forehead_search_ratio=0.45)
            print("    ✅ Anterior generated.")

            # =================================================
            # STEP 7: SAVE METADATA FOR YOLO QC
            # =================================================
            metadata = {
                "id": scan_id,
                "sagittal": sag_meta_data,
                "coronal": cor_meta_data,
                "axial": ax_meta_data
            }
            
            with open(meta_path, "w", encoding="utf-8") as meta_f:
                json.dump(metadata, meta_f, indent=4)
            print("    ✅ Pipeline metadata saved for YOLO.")

            print(f"\n    ✅ COMPLETE: IXI ID {scan_id}")
            print(f"       Filename: {standardized_name}")

        except Exception as e:
            print(f"\n    ❌ FAILED: IXI ID {scan_id}")
            print(f"       Error: {e}")

    # ========================================================
    # CLEANUP
    # ========================================================
    if os.path.exists(temp_mni_path):
        os.remove(temp_mni_path)

    print("\n" + "=" * 70)
    print("🎯 PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

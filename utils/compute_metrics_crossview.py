#!/usr/bin/env python3
"""Legacy standalone evaluator for cross-view IoU metrics from inference JSON.

Its metadata projection and 1D IoU helpers are mirrored in
``inference/infer_yolo.py`` for live CSV output.  The constants at the top of
this script must be edited before using it against an older JSON result set.
"""

import json
import os
import argparse
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
# JSON_PATH = r"D:\Academic Stuff\Internship\QualityControl\jsons_results\TubeTK_Spockmip\batch_inference_output_TubeTK_Spockmip_ALL.json"

# SAG_DIR = r"D:\Academic Stuff\Internship\QualityControl\IXI_Pipeline_Outputs\IXI_Spockmip\2_Sagittal_Crops\MIPs_Mid_32\Plain"


JSON_PATH = r"D:\Academic Stuff\Internship\QualityControl\jsons_results\IXI_Spockmip\batch_inference_output_IXI_Spockmip_ALL.json"


SAG_DIR = r"D:\Academic Stuff\Internship\QualityControl\IXI_Pipeline_Outputs\IXI_Spockmip\2_Sagittal_Crops\MIPs_Mid_32\Plain"


# JSON_PATH = r"D:\Academic Stuff\Internship\QualityControl\jsons_results\IXI_nnUNet\batch_inference_output_IXI_nnUNet_ALL.json"



# SAG_DIR = r"D:\Academic Stuff\Internship\QualityControl\IXI_Pipeline_Outputs\IXI_nnUNet_Full\IXI_nnUNet_Mips\2_Sagittal_Crops\MIPs_Mid_32\Plain"

# Toggles for flipping coordinate mappings if orientation is visually mirrored
INVERT_AXIAL_MAPPING = False   # Sagittal X-axis 
INVERT_CORONAL_MAPPING = False # Sagittal Y-axis

# Exact threshold for a successful Cross-View match
IOU_THRESHOLD = 0.25

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_1d_iou(min1, max1, min2, max2):
    """Calculates the 1D Intersection over Union (IoU) for two line segments."""
    intersection_min = max(min1, min2)
    intersection_max = min(max1, max2)
    
    intersection = max(0, intersection_max - intersection_min)
    union = (max1 - min1) + (max2 - min2) - intersection
    
    if union <= 0:
        return 0.0
        
    return intersection / union

def is_anterior(pred):
    name = str(pred.get("class_name", "")).lower()
    return "anterior" in name or pred.get("class_id") == 0

def is_posterior(pred):
    name = str(pred.get("class_name", "")).lower()
    return "posterior" in name or pred.get("class_id") == 1

def is_center(pred):
    name = str(pred.get("class_name", "")).lower()
    return "center" in name or pred.get("class_id") == 2

def is_straight_sinus(pred):
    name = str(pred.get("class_name", "")).lower()
    return "straight" in name or "sinus" in name

def map_cv_to_sagittal(cv_min, cv_max, meta_cv, meta_sag, is_axial=False, invert=False):
    """
    Reverse-engineers Matplotlib letterboxing and 3D grid expansion to map 
    a YOLO bounding box from a Cross-View back to the Sagittal plane perfectly.
    """
    canvas_cv = meta_cv["canvas"]
    eff_w = meta_cv["w"]
    eff_h = meta_cv["h"] * meta_cv["aspect"]
    
    scale = min(canvas_cv / eff_w, canvas_cv / eff_h)
    rendered_h = eff_h * scale
    pad_y = (canvas_cv - rendered_h) / 2.0
    
    prop1 = (cv_min - pad_y) / rendered_h
    prop2 = (cv_max - pad_y) / rendered_h
    
    array_pos1 = prop1 * meta_cv["h"]
    array_pos2 = prop2 * meta_cv["h"]
    
    center_cv = meta_cv["h"] / 2.0
    d1 = array_pos1 - center_cv
    d2 = array_pos2 - center_cv
    
    if invert:
        d1, d2 = -d1, -d2
        
    if is_axial:
        center_sag = meta_sag["w"] / 2.0
        sag_pos1 = center_sag + d1 + meta_sag["offset_x"]
        sag_pos2 = center_sag + d2 + meta_sag["offset_x"]
    else:
        center_sag = meta_sag["h"] / 2.0
        sag_pos1 = center_sag + d1 + meta_sag["offset_y"]
        sag_pos2 = center_sag + d2 + meta_sag["offset_y"]
        
    return min(sag_pos1, sag_pos2), max(sag_pos1, sag_pos2)


# ==========================================
# 3. MAIN LOGIC
# ==========================================
def main():
    if not os.path.exists(JSON_PATH):
        print(f"❌ Could not find JSON at {JSON_PATH}")
        return

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_patients = len(data)
    print(f"Loaded JSON with {total_patients} entries.")
    print(f"Evaluating IoU mathematically using threshold {IOU_THRESHOLD}...\n")

    stats = {
        "anterior": {"sag_found": 0, "cv_agreed": 0, "cv_low_iou": 0, "cv_0_over": 0, "cv_missed": 0, "cv_orphan": 0, "multiple_boxes": 0},
        "posterior": {"sag_found": 0, "cv_agreed": 0, "cv_low_iou": 0, "cv_0_over": 0, "cv_missed": 0, "cv_orphan": 0, "multiple_boxes": 0},
        "center": {"sag_found": 0, "cv_agreed": "N/A", "cv_low_iou": "N/A", "cv_0_over": "N/A", "cv_missed": "N/A", "cv_orphan": "N/A", "multiple_boxes": 0},
        "straight_sinus": {"sag_found": 0, "cv_agreed": "N/A", "cv_low_iou": "N/A", "cv_0_over": "N/A", "cv_missed": "N/A", "cv_orphan": "N/A", "multiple_boxes": 0}
    }

    missing_meta = 0

    # ------------------------------------------
    # PROCESS EVERY PATIENT
    # ------------------------------------------
    for patient_key, patient_data in data.items():
        filename = patient_data.get("filename")
        if not filename:
            # Fallback if filename isn't properly stored
            patient_id = patient_key.split('_')[-1] 
            filename = f"{patient_key.split('_')[0]}_{patient_key.split('_')[1]}_{patient_id}.png"

        # Load specific Metadata for this scan
        meta_filename = filename.replace(".png", "_meta.json")
        meta_path = os.path.join(SAG_DIR, meta_filename)
        
        if not os.path.exists(meta_path):
            missing_meta += 1
            continue

        with open(meta_path, 'r', encoding='utf-8') as mf:
            meta = json.load(mf)

        # Extract Predictions
        sag_preds = patient_data.get("sagittal_predictions") or []
        ax_preds = patient_data.get("axial_predictions") or []
        cor_preds = patient_data.get("anterior_predictions") or patient_data.get("coronal_predictions") or []
        ss_preds = patient_data.get("straight_sinus_predictions") or []

        sag_ant = [p for p in sag_preds if is_anterior(p)]
        sag_post = [p for p in sag_preds if is_posterior(p)]
        sag_cen = [p for p in sag_preds if is_center(p)]

        # --- Center ---
        if len(sag_cen) > 0:
            stats["center"]["sag_found"] += 1
        if len(sag_cen) > 1:
            stats["center"]["multiple_boxes"] += 1

        # --- Straight Sinus ---
        if len(ss_preds) > 0:
            stats["straight_sinus"]["sag_found"] += 1
        if len(ss_preds) > 1:
            stats["straight_sinus"]["multiple_boxes"] += 1

        # --- Anterior (Sag vs Coronal) ---
        has_sag_ant = len(sag_ant) > 0
        has_cv_ant = len(cor_preds) > 0

        if len(sag_ant) > 1 or len(cor_preds) > 1:
            stats["anterior"]["multiple_boxes"] += 1

        if has_sag_ant:
            stats["anterior"]["sag_found"] += 1
            if has_cv_ant:
                best_iou = 0.0
                for p_cor in cor_preds:
                    _, c_y1, _, c_y2 = p_cor["box_xyxy"]
                    sag_y1_proj, sag_y2_proj = map_cv_to_sagittal(
                        c_y1, c_y2, meta["coronal"], meta["sagittal"], 
                        is_axial=False, invert=INVERT_CORONAL_MAPPING
                    )
                    for p_sag in sag_ant:
                        _, s_y1, _, s_y2 = p_sag["box_xyxy"]
                        iou = calculate_1d_iou(sag_y1_proj, sag_y2_proj, s_y1, s_y2)
                        best_iou = max(best_iou, iou)

                # Categorize the match
                if best_iou >= IOU_THRESHOLD:
                    stats["anterior"]["cv_agreed"] += 1
                elif best_iou > 0:
                    stats["anterior"]["cv_low_iou"] += 1
                else:
                    stats["anterior"]["cv_0_over"] += 1
            else:
                stats["anterior"]["cv_missed"] += 1
        elif has_cv_ant:
            stats["anterior"]["cv_orphan"] += 1


        # --- Posterior (Sag vs Axial) ---
        has_sag_post = len(sag_post) > 0
        has_cv_post = len(ax_preds) > 0

        if len(sag_post) > 1 or len(ax_preds) > 1:
            stats["posterior"]["multiple_boxes"] += 1

        if has_sag_post:
            stats["posterior"]["sag_found"] += 1
            if has_cv_post:
                best_iou = 0.0
                for p_ax in ax_preds:
                    _, a_y1, _, a_y2 = p_ax["box_xyxy"]
                    sag_x1_proj, sag_x2_proj = map_cv_to_sagittal(
                        a_y1, a_y2, meta["axial"], meta["sagittal"], 
                        is_axial=True, invert=INVERT_AXIAL_MAPPING
                    )
                    for p_sag in sag_post:
                        s_x1, _, s_x2, _ = p_sag["box_xyxy"]
                        iou = calculate_1d_iou(sag_x1_proj, sag_x2_proj, s_x1, s_x2)
                        best_iou = max(best_iou, iou)

                # Categorize the match
                if best_iou >= IOU_THRESHOLD:
                    stats["posterior"]["cv_agreed"] += 1
                elif best_iou > 0:
                    stats["posterior"]["cv_low_iou"] += 1
                else:
                    stats["posterior"]["cv_0_over"] += 1
            else:
                stats["posterior"]["cv_missed"] += 1
        elif has_cv_post:
            stats["posterior"]["cv_orphan"] += 1

    # ------------------------------------------
    # FORMAT OUTPUT TABLE
    # ------------------------------------------
    stats_text = []
    stats_text.append("==========================================================================================")
    stats_text.append("                          SPATIAL IoU INFERENCE STATISTICS")
    stats_text.append("==========================================================================================")
    stats_text.append(f"Total Patients JSON:    {total_patients}")
    stats_text.append(f"Missing Metadata Files: {missing_meta} (These were skipped from spatial evaluation)")
    stats_text.append("")
    stats_text.append("--- Cross-View Spatial Agreement (PATIENT-LEVEL) ---")
    stats_text.append(f"  (Evaluated with dynamic Metadata projection. Threshold = {IOU_THRESHOLD})")
    stats_text.append("")
    stats_text.append(f"| {'Anatomy':<14} | {'Sag Found':<9} | {'CV Agreed':<9} | {'CV LowIoU':<9} | {'CV 0 Over':<9} | {'CV Missed':<9} | {'CV Orphan':<9} | {'>1 Box/Vw':<9} |")
    stats_text.append("|----------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|")

    for cls_name in ["anterior", "posterior", "center", "straight_sinus"]:
        d = stats[cls_name]

        sag_f = str(d["sag_found"])
        cv_a  = str(d["cv_agreed"])
        cv_l  = str(d["cv_low_iou"])
        cv_no = str(d["cv_0_over"])
        cv_m  = str(d["cv_missed"])
        cv_o  = str(d["cv_orphan"])
        mult  = str(d["multiple_boxes"])

        display_name = cls_name.replace("_", " ").title()

        stats_text.append(f"| {display_name:<14} | {sag_f:<9} | {cv_a:<9} | {cv_l:<9} | {cv_no:<9} | {cv_m:<9} | {cv_o:<9} | {mult:<9} |")

    stats_text.append("==========================================================================================")

    print("\n".join(stats_text))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run four YOLO detectors on standardized IXI or TubeTK MIP outputs.

The script writes one compact CSV row per case with model predictions and
metadata-projected anterior/posterior cross-view IoU results.  ``--debug``
saves every annotated view; ``--manual_inspection`` saves only spatially
disagreeing anterior/posterior view pairs.  See ``inference/README.md`` for a
complete invocation.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Ultralytics otherwise writes settings into the user's roaming profile, which
# can be unavailable in managed Windows environments.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# HELPERS
# ============================================================

def parse_bool(value):
    """Accept ``--flag`` as well as ``--flag true``/``false``."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")

def is_anterior(pred):
    """Check if a sagittal prediction is classified as anterior."""
    name = str(pred.get("class_name", "")).lower()
    return "anterior" in name or pred.get("class_id") == 0

def is_posterior(pred):
    """Check if a sagittal prediction is classified as posterior."""
    name = str(pred.get("class_name", "")).lower()
    return "posterior" in name or pred.get("class_id") == 1

def is_straight_sinus(pred):
    """Check if a sagittal prediction is classified as straight sinus."""
    name = str(pred.get("class_name", "")).lower()
    return "straight" in name or "sinus" in name or pred.get("class_id") not in [0, 1]

def merge_overlapping_boxes(boxes_data, img_shape):
    """
    Merge overlapping boxes within each class.

    Boxes are first rasterized into a binary mask. Connected
    overlapping regions are converted back into one bounding box.
    Confidence statistics are calculated from all source boxes
    contributing to the merged region.
    """
    merged_results = []
    classes = set(b["cls"] for b in boxes_data)

    for cls in classes:
        cls_boxes = [b for b in boxes_data if b["cls"] == cls]

        mask = np.zeros(img_shape[:2], dtype=np.uint8)

        for b in cls_boxes:
            x1, y1, x2, y2 = map(int, b["box"])
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            merged_x1 = x
            merged_y1 = y
            merged_x2 = x + w
            merged_y2 = y + h

            confs = []

            for b in cls_boxes:
                bx1, by1, bx2, by2 = b["box"]

                if not (
                    bx2 < merged_x1
                    or bx1 > merged_x2
                    or by2 < merged_y1
                    or by1 > merged_y2
                ):
                    confs.append(b["conf"])

            if confs:
                merged_results.append(
                    {
                        "box": (
                            merged_x1,
                            merged_y1,
                            merged_x2,
                            merged_y2,
                        ),
                        "cls": cls,
                        "max_conf": float(max(confs)),
                        "avg_conf": float(sum(confs) / len(confs)),
                        "num_boxes_merged": len(confs),
                    }
                )

    return merged_results


def extract_predictions(
    results,
    model,
    apply_spatial_correction=False,
):
    """
    Extract YOLO detections, optionally apply the existing
    sagittal left/right spatial correction, merge overlapping
    boxes, and return JSON-ready predictions.
    """
    if results.boxes is None or len(results.boxes) == 0:
        return []

    img_shape = results.orig_shape
    img_width = img_shape[1]
    img_midpoint_x = img_width / 2.0

    raw_boxes = []

    for box in results.boxes:
        coords = box.xyxy[0].tolist()
        x1, y1, x2, y2 = coords

        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        if apply_spatial_correction:
            box_center_x = (x1 + x2) / 2.0

            # Existing sagittal correction:
            # class 1 = Posterior
            # class 0 = Anterior
            if cls_id == 1 and box_center_x < img_midpoint_x:
                cls_id = 0

            elif cls_id == 0 and box_center_x >= img_midpoint_x:
                cls_id = 1

        raw_boxes.append(
            {
                "box": coords,
                "cls": cls_id,
                "conf": conf,
            }
        )

    merged_detections = merge_overlapping_boxes(
        raw_boxes,
        img_shape,
    )

    final_predictions = []

    for det in merged_detections:
        cls_id = det["cls"]

        if hasattr(model, "names"):
            if isinstance(model.names, dict):
                cls_name = model.names.get(
                    cls_id,
                    f"class_{cls_id}",
                )
            else:
                cls_name = (
                    model.names[cls_id]
                    if cls_id < len(model.names)
                    else f"class_{cls_id}"
                )
        else:
            cls_name = f"class_{cls_id}"

        final_predictions.append(
            {
                "box_xyxy": [
                    round(c, 2)
                    for c in det["box"]
                ],
                "confidence_max": round(
                    det["max_conf"],
                    4,
                ),
                "confidence_avg": round(
                    det["avg_conf"],
                    4,
                ),
                "class_id": cls_id,
                "class_name": cls_name,
                "num_boxes_merged": det[
                    "num_boxes_merged"
                ],
            }
        )

    return final_predictions


def discover_patients(input_root):
    """
    Discover patients from a standardized IXI or TubeTK filename.

    Expected filename:
        <dataset>_<model_name>_<ID>.png
        (e.g., IXI_manual_012.png or TubeTK_manual_105.png)

    The same filename is expected in every view folder.
    """
    input_root = Path(input_root)

    sagittal_dir = (
        input_root
        / "2_Sagittal_Crops"
        / "MIPs_Mid_32"
        / "Plain"
    )

    if not sagittal_dir.exists():
        raise FileNotFoundError(
            f"Missing sagittal input directory:\n{sagittal_dir}"
        )

    patients = []

    for path in sorted(sagittal_dir.glob("*.png")):
        if not path.is_file():
            continue

        stem = path.stem

        parts = stem.split("_")
        if len(parts) < 3 or parts[0] not in {"IXI", "TubeTK"}:
            print(
                f"  ⚠️ Ignoring non-standard filename: "
                f"{path.name}"
            )
            continue

        patient_id = parts[-1]

        patients.append(
            {
                "dataset": parts[0],
                "id": patient_id,
                "filename": path.name,
            }
        )

    return patients


def build_view_paths(input_root, filename):
    """
    Map one standardized filename to all four model inputs.
    """
    root = Path(input_root)

    return {
        "sagittal": (
            root
            / "2_Sagittal_Crops"
            / "MIPs_Mid_32"
            / "Plain"
            / filename
        ),
        "straight_sinus": (
            root
            / "2_Sagittal_Crops"
            / "MIPs_Mid_14"
            / "Plain"
            / filename
        ),
        "axial": (
            root
            / "3_Posterior"
            / filename
        ),
        "anterior": (
            root
            / "4_Anterior"
            / "Coronal_Isolated"
            / filename
        ),
    }


def render_predictions(image, predictions):
    """Render the exported (post-correction, merged) predictions on an image."""
    rendered = image.copy()
    for prediction in predictions:
        x1, y1, x2, y2 = map(int, prediction["box_xyxy"])
        label = f"{prediction['class_name']} {prediction['confidence_max']:.2f}"
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(rendered, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return rendered


def run_model(
    image_path,
    model,
    confidence,
    apply_spatial_correction=False,
):
    """
    Run one YOLO model on one image.
    """
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image:\n{image_path}"
        )

    result = model.predict(
        source=image,
        conf=confidence,
        verbose=False,
    )[0]

    predictions = extract_predictions(
        result,
        model,
        apply_spatial_correction=apply_spatial_correction,
    )

    return predictions, render_predictions(image, predictions)


IOU_THRESHOLD = 0.25
INVERT_AXIAL_MAPPING = False
INVERT_CORONAL_MAPPING = False


def calculate_1d_iou(min1, max1, min2, max2):
    """Calculate 1D IoU between two line segments."""
    intersection = max(0, min(max1, max2) - max(min1, min2))
    union = (max1 - min1) + (max2 - min2) - intersection
    return 0.0 if union <= 0 else intersection / union


def map_cv_to_sagittal(cv_min, cv_max, meta_cv, meta_sag, is_axial=False, invert=False):
    """Map a cross-view rendered-image interval back to the sagittal canvas."""
    canvas_cv = meta_cv["canvas"]
    effective_width = meta_cv["w"]
    effective_height = meta_cv["h"] * meta_cv["aspect"]
    scale = min(canvas_cv / effective_width, canvas_cv / effective_height)
    rendered_height = effective_height * scale
    pad_y = (canvas_cv - rendered_height) / 2.0

    array_pos1 = ((cv_min - pad_y) / rendered_height) * meta_cv["h"]
    array_pos2 = ((cv_max - pad_y) / rendered_height) * meta_cv["h"]
    center_cv = meta_cv["h"] / 2.0
    delta1, delta2 = array_pos1 - center_cv, array_pos2 - center_cv
    if invert:
        delta1, delta2 = -delta1, -delta2

    if is_axial:
        center_sag = meta_sag["w"] / 2.0
        sag_pos1 = center_sag + delta1 + meta_sag["offset_x"]
        sag_pos2 = center_sag + delta2 + meta_sag["offset_x"]
    else:
        center_sag = meta_sag["h"] / 2.0
        sag_pos1 = center_sag + delta1 + meta_sag["offset_y"]
        sag_pos2 = center_sag + delta2 + meta_sag["offset_y"]
    return min(sag_pos1, sag_pos2), max(sag_pos1, sag_pos2)


def evaluate_crossview_pair(sagittal_predictions, crossview_predictions, meta, anatomy):
    """Compute the best metadata-projected cross-view IoU for one anatomy."""
    if anatomy == "anterior":
        sagittal_matches = [p for p in sagittal_predictions if is_anterior(p)]
        cross_meta_key = "coronal"
        is_axial = False
        invert = INVERT_CORONAL_MAPPING
    else:
        sagittal_matches = [p for p in sagittal_predictions if is_posterior(p)]
        cross_meta_key = "axial"
        is_axial = True
        invert = INVERT_AXIAL_MAPPING

    if not sagittal_matches and not crossview_predictions:
        return {"best_iou": None, "status": "BOTH_NEGATIVE", "agree": True}
    if not sagittal_matches:
        return {"best_iou": None, "status": "SAGITTAL_MISSED", "agree": False}
    if not crossview_predictions:
        return {"best_iou": None, "status": "CROSS_VIEW_MISSED", "agree": False}
    if meta is None:
        return {"best_iou": None, "status": "METADATA_MISSING", "agree": None}

    try:
        best_iou = 0.0
        for crossview_prediction in crossview_predictions:
            _, cv_y1, _, cv_y2 = crossview_prediction["box_xyxy"]
            projected_min, projected_max = map_cv_to_sagittal(
                cv_y1,
                cv_y2,
                meta[cross_meta_key],
                meta["sagittal"],
                is_axial=is_axial,
                invert=invert,
            )
            for sagittal_prediction in sagittal_matches:
                sx1, sy1, sx2, sy2 = sagittal_prediction["box_xyxy"]
                sag_min, sag_max = (sx1, sx2) if is_axial else (sy1, sy2)
                best_iou = max(best_iou, calculate_1d_iou(projected_min, projected_max, sag_min, sag_max))
    except (KeyError, TypeError, ZeroDivisionError):
        return {"best_iou": None, "status": "METADATA_ERROR", "agree": None}

    if best_iou >= IOU_THRESHOLD:
        status, agree = "AGREED", True
    elif best_iou > 0:
        status, agree = "LOW_IOU", False
    else:
        status, agree = "NO_OVERLAP", False
    return {"best_iou": round(best_iou, 4), "status": status, "agree": agree}


def evaluate_crossviews(patient_results, metadata_path):
    """Evaluate anterior/coronal and posterior/axial spatial agreement."""
    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = None

    sagittal = patient_results.get("sagittal_predictions") or []
    return {
        "anterior": evaluate_crossview_pair(sagittal, patient_results.get("anterior_predictions") or [], metadata, "anterior"),
        "posterior": evaluate_crossview_pair(sagittal, patient_results.get("axial_predictions") or [], metadata, "posterior"),
    }


def save_visualizations(annotations, output_path, dataset, patient_id, debug, disagreements):
    """Save all predictions in debug mode, or only disagreeing cross-view pairs."""
    if not debug and not disagreements:
        return

    output_kind = "debug" if debug else "manual_inspection"
    case_dir = output_path.parent / output_kind / f"{dataset}_{patient_id}"
    case_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        selected_views = set(annotations)
    else:
        selected_views = set()
        if "anterior" in disagreements:
            selected_views.update({"sagittal", "anterior"})
        if "posterior" in disagreements:
            selected_views.update({"sagittal", "axial"})

    for view in sorted(selected_views):
        image = annotations.get(view)
        if image is not None:
            cv2.imwrite(str(case_dir / f"{view}.png"), image)


def write_csv(output_data, output_path):
    """Write one row per case, including predictions and spatial cross-view QA."""
    views = ["sagittal", "anterior", "axial", "straight_sinus"]
    fieldnames = [
        "dataset",
        "id",
        "anterior_best_iou",
        "anterior_crossview_result",
        "posterior_best_iou",
        "posterior_crossview_result",
    ]
    fieldnames.extend(f"{view}_predictions" for view in views)

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for data in output_data.values():
            row = {
                "dataset": data["dataset"],
                "id": data["id"],
                "anterior_best_iou": data["crossview"]["anterior"]["best_iou"],
                "anterior_crossview_result": data["crossview"]["anterior"]["status"],
                "posterior_best_iou": data["crossview"]["posterior"]["best_iou"],
                "posterior_crossview_result": data["crossview"]["posterior"]["status"],
            }
            for view in views:
                row[f"{view}_predictions"] = json.dumps(data.get(f"{view}_predictions"), ensure_ascii=False)
            writer.writerow(row)


def generate_statistics(output_data, output_path):
    """
    Analyzes the output data dictionary and generates a formatted
    statistics report, saving it to a log file and printing to console.
    """
    total_patients = len(output_data)
    if total_patients == 0:
        return

    missing_views = 0
    total_errors = 0
    patients_with_detections = 0

    # Patient-Level metrics tracking
    stats = {
        "anterior": {"sag_found": 0, "cv_agreed": 0, "cv_missed": 0, "cv_orphan": 0, "multiple_boxes": 0},
        "posterior": {"sag_found": 0, "cv_agreed": 0, "cv_missed": 0, "cv_orphan": 0, "multiple_boxes": 0},
        "straight_sinus": {"sag_found": 0, "cv_agreed": "N/A", "cv_missed": "N/A", "cv_orphan": "N/A", "multiple_boxes": 0}
    }

    for pat_key, data in output_data.items():
        has_any = False

        # Basic Check
        for view in ["sagittal", "anterior", "axial", "straight_sinus"]:
            if data.get(f"{view}_image_file") is None:
                missing_views += 1
            if f"{view}_error" in data:
                total_errors += 1
            if data.get(f"{view}_predictions"):
                has_any = True

        if has_any:
            patients_with_detections += 1

        # --- Extract predictions ---
        sag_preds = data.get("sagittal_predictions") or []
        ax_preds = data.get("axial_predictions") or []
        cor_preds = data.get("anterior_predictions") or []
        ss_preds = data.get("straight_sinus_predictions") or []

        sag_ant = [p for p in sag_preds if is_anterior(p)]
        sag_post = [p for p in sag_preds if is_posterior(p)]

        # --- Straight Sinus Analysis (Patient-Level) ---
        if len(ss_preds) > 0:
            stats["straight_sinus"]["sag_found"] += 1
        if len(ss_preds) > 1:
            stats["straight_sinus"]["multiple_boxes"] += 1

        # --- 1. Anterior Analysis (Patient-Level) ---
        has_sag_ant = len(sag_ant) > 0
        has_cv_ant = len(cor_preds) > 0

        if len(sag_ant) > 1 or len(cor_preds) > 1:
            stats["anterior"]["multiple_boxes"] += 1

        if has_sag_ant:
            stats["anterior"]["sag_found"] += 1
            if has_cv_ant:
                stats["anterior"]["cv_agreed"] += 1
            else:
                stats["anterior"]["cv_missed"] += 1
        elif has_cv_ant:
            stats["anterior"]["cv_orphan"] += 1

        # --- 2. Posterior Analysis (Patient-Level) ---
        has_sag_post = len(sag_post) > 0
        has_cv_post = len(ax_preds) > 0

        if len(sag_post) > 1 or len(ax_preds) > 1:
            stats["posterior"]["multiple_boxes"] += 1

        if has_sag_post:
            stats["posterior"]["sag_found"] += 1
            if has_cv_post:
                stats["posterior"]["cv_agreed"] += 1
            else:
                stats["posterior"]["cv_missed"] += 1
        elif has_cv_post:
            stats["posterior"]["cv_orphan"] += 1

    # ============================================================
    # FORMAT OUTPUT TEXT
    # ============================================================
    stats_text = []
    stats_text.append("================================================================================")
    stats_text.append("                              INFERENCE STATISTICS")
    stats_text.append("================================================================================")

    pct_detections = (patients_with_detections / total_patients) * 100 if total_patients > 0 else 0
    stats_text.append(f"Total Patients Processed:    {total_patients}")
    stats_text.append(f"Patients w/ ANY Detection:   {patients_with_detections} ({pct_detections:.1f}%)")
    stats_text.append(f"Missing Images Encountered:  {missing_views}")
    stats_text.append(f"Inference Errors:            {total_errors}")
    stats_text.append("")

    stats_text.append("--- Cross-View Co-Detection (PATIENT-LEVEL) ---")
    stats_text.append("  (Checks if BOTH views detected the target anatomy, ignoring spatial overlap)")
    stats_text.append("")
    stats_text.append(f"| {'Anatomy':<14} | {'Sag Found':<9} | {'CV Agreed':<9} | {'CV Missed':<9} | {'CV Orphan':<9} | {'>1 Box/Vw':<9} |")
    stats_text.append("|----------------|-----------|-----------|-----------|-----------|-----------|")

    for cls_name in ["anterior", "posterior", "straight_sinus"]:
        d = stats[cls_name]

        sag_f = str(d["sag_found"])
        cv_a  = str(d["cv_agreed"])
        cv_m  = str(d["cv_missed"])
        cv_o  = str(d["cv_orphan"])
        mult  = str(d["multiple_boxes"])

        display_name = cls_name.replace("_", " ").title()

        stats_text.append(f"| {display_name:<14} | {sag_f:<9} | {cv_a:<9} | {cv_m:<9} | {cv_o:<9} | {mult:<9} |")

    stats_text.append("================================================================================")

    stats_str = "\n".join(stats_text)

    # Print to console
    print("\n" + stats_str)

    # Save to file
    stats_path = output_path.with_name(output_path.stem + "_stats.txt")
    try:
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(stats_str + "\n")
        print(f"\nStatistics saved to: {stats_path}")
    except Exception as e:
        print(f"\n⚠️ Could not save statistics file: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "IXI/TubeTK multi-model YOLO inference pipeline. "
            "Uses the standardized outputs of the current "
            "preprocessing pipeline."
        )
    )

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Root folder containing 2_Sagittal_Crops, "
            "3_Posterior and 4_Anterior."
        ),
    )

    # --------------------------------------------------------
    # YOLO weights
    # --------------------------------------------------------

    parser.add_argument(
        "--weights-sag",
        required=True,
        help="Sagittal YOLO weights.",
    )

    parser.add_argument(
        "--weights-cor",
        required=True,
        help=(
            "Anterior YOLO weights. "
            "Runs on 4_Anterior/Coronal_Isolated."
        ),
    )

    parser.add_argument(
        "--weights-ax",
        required=True,
        help=(
            "Posterior/Axial YOLO weights. "
            "Runs on 3_Posterior."
        ),
    )

    parser.add_argument(
        "--weights-ss",
        required=True,
        help=(
            "Straight Sinus YOLO weights. "
            "Runs on MIPs_Mid_14/Plain."
        ),
    )

    # --------------------------------------------------------
    # Confidence thresholds
    # --------------------------------------------------------

    parser.add_argument(
        "--conf-sag",
        type=float,
        default=0.25,
        help="Sagittal confidence threshold.",
    )

    parser.add_argument(
        "--conf-cor",
        type=float,
        default=0.25,
        help="Anterior confidence threshold.",
    )

    parser.add_argument(
        "--conf-ax",
        type=float,
        default=0.15,
        help="Posterior/Axial confidence threshold.",
    )

    parser.add_argument(
        "--conf-ss",
        type=float,
        default=0.25,
        help="Straight Sinus confidence threshold.",
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        type=str,
        default="batch_inference_output.csv",
        help="Output CSV file.",
    )

    parser.add_argument(
        "--debug",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Save annotated predictions for every successfully inferred image.",
    )

    parser.add_argument(
        "--manual_inspection",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help=(
            "Save annotated sagittal/cross-view images only when anterior or "
            "posterior cross-view detections disagree. Ignored when --debug is true."
        ),
    )

    args = parser.parse_args()

    input_root = Path(args.input_dir)
    output_path = Path(args.output)
    if output_path.suffix.lower() != ".csv":
        output_path = output_path.with_suffix(".csv")

    if not input_root.exists():
        raise FileNotFoundError(
            f"Input directory does not exist:\n{input_root}"
        )

    print("=" * 70)
    print("MULTI-DATASET YOLO BATCH INFERENCE")
    print("=" * 70)
    print(f"Input root: {input_root}")
    print(f"Output CSV: {output_path}")
    print()

    # --------------------------------------------------------
    # Discover patients
    # --------------------------------------------------------

    patients = discover_patients(input_root)

    if not patients:
        print(
            "❌ No standardized IXI or TubeTK PNG files found."
        )
        return

    print(
        f"Found {len(patients)} standardized case(s)."
    )

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print()
    print("Loading YOLO models...")

    model_sag = YOLO(args.weights_sag)
    model_cor = YOLO(args.weights_cor)
    model_ax = YOLO(args.weights_ax)
    model_ss = YOLO(args.weights_ss)

    print("✅ All models loaded.")

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    output_data = {}

    for index, patient in enumerate(
        patients,
        start=1,
    ):
        patient_id = patient["id"]
        filename = patient["filename"]
        dataset = patient["dataset"]

        print()
        print("-" * 70)
        print(
            f"[{index}/{len(patients)}] "
            f"{dataset} ID: {patient_id}"
        )
        print(
            f"Filename: {filename}"
        )
        print("-" * 70)

        paths = build_view_paths(
            input_root,
            filename,
        )

        patient_results = {
            "dataset": dataset,
            "id": patient_id,
            "filename": filename,
        }
        annotations = {}

        # ----------------------------------------------------
        # View/model configuration
        # ----------------------------------------------------

        configurations = {
            "sagittal": {
                "model": model_sag,
                "confidence": args.conf_sag,
                "spatial_correction": True,
            },
            "anterior": {
                "model": model_cor,
                "confidence": args.conf_cor,
                "spatial_correction": False,
            },
            "axial": {
                "model": model_ax,
                "confidence": args.conf_ax,
                "spatial_correction": False,
            },
            "straight_sinus": {
                "model": model_ss,
                "confidence": args.conf_ss,
                "spatial_correction": False,
            },
        }

        for view, config in configurations.items():

            path = paths[view]

            if not path.exists():
                print(
                    f"  ⚠️ {view}: MISSING"
                )
                print(
                    f"     {path}"
                )

                patient_results[
                    f"{view}_image_file"
                ] = None

                patient_results[
                    f"{view}_applied_conf"
                ] = config["confidence"]

                patient_results[
                    f"{view}_predictions"
                ] = None

                continue

            print(
                f"  ✔ {view}: {path}"
            )

            try:
                predictions, annotation = run_model(
                    image_path=path,
                    model=config["model"],
                    confidence=config["confidence"],
                    apply_spatial_correction=(
                        config["spatial_correction"]
                    ),
                )

                patient_results[
                    f"{view}_image_file"
                ] = filename

                patient_results[
                    f"{view}_applied_conf"
                ] = config["confidence"]

                patient_results[
                    f"{view}_predictions"
                ] = predictions
                annotations[view] = annotation

                print(
                    f"    → {len(predictions)} "
                    f"merged detection(s)"
                )

            except Exception as exc:
                print(
                    f"    ❌ {view} inference failed: "
                    f"{exc}"
                )

                patient_results[
                    f"{view}_image_file"
                ] = filename

                patient_results[
                    f"{view}_applied_conf"
                ] = config["confidence"]

                patient_results[
                    f"{view}_predictions"
                ] = None

                patient_results[
                    f"{view}_error"
                ] = str(exc)

        metadata_path = paths["sagittal"].with_name(
            f"{paths['sagittal'].stem}_meta.json"
        )
        patient_results["crossview"] = evaluate_crossviews(
            patient_results,
            metadata_path,
        )
        disagreements = [
            anatomy
            for anatomy, assessment in patient_results["crossview"].items()
            if assessment["agree"] is not True
        ]
        patient_results["manual_inspection_reasons"] = disagreements
        if args.debug or args.manual_inspection:
            save_visualizations(
                annotations=annotations,
                output_path=output_path,
                dataset=dataset,
                patient_id=patient_id,
                debug=args.debug,
                disagreements=disagreements,
            )

        # Dataset-qualified key avoids collisions between IXI and TubeTK IDs.
        output_data[
            f"{dataset}_{patient_id}"
        ] = patient_results

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(output_data, output_path)

    print()
    print("=" * 70)
    print("✅ BATCH INFERENCE COMPLETE")
    print("=" * 70)
    print(
        f"Processed: {len(patients)} patients"
    )
    print(
        f"Results:   {output_path}"
    )


if __name__ == "__main__":
    main()

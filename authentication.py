"""
Biometric Authentication & Identification Pipeline.
Integrates FingerprintCNN distortion classifier, DDRNet_DIR geometric rectification,
and SourceAFIS matching against MySQL gallery templates.
Bozorth3 and MINDTCT have been completely removed from this active pipeline.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import zoom

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from model import FingerprintCNN
from mysql_auth import get_person_templates, get_all_gallery_templates
from matcher import MATCHER, SourceAFISMatcher
from database import MATCH_THRESHOLD

from DDRNet.models.DDRNet_DIR import DDRNet_DIR
from DDRNet.tools.fp_segmentation import segmentation_coherence


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cpu")

CLASSIFIER_CHECKPOINT = CURRENT_DIR / "experiments" / "classifier_hard_negative" / "best_model.pth"
CLASSIFIER_THRESHOLD = 0.60

DDRNET_CHECKPOINT = CURRENT_DIR / "experiments" / "ddrnet_baseline" / "best_model.pth"

AUTH_OUTPUT_DIR = CURRENT_DIR / "experiments" / "authentication_outputs"
AUTH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFIER = None
DDRNET = None


# ============================================================
# CLASSIFIER
# ============================================================

def load_classifier() -> FingerprintCNN:
    global CLASSIFIER

    if CLASSIFIER is None:
        if not CLASSIFIER_CHECKPOINT.exists():
            raise FileNotFoundError(f"Classifier checkpoint not found: {CLASSIFIER_CHECKPOINT}")

        CLASSIFIER = FingerprintCNN()
        checkpoint = torch.load(CLASSIFIER_CHECKPOINT, map_location=DEVICE, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        CLASSIFIER.load_state_dict(state_dict)
        CLASSIFIER.to(DEVICE)
        CLASSIFIER.eval()

    return CLASSIFIER


def classify_distortion(image: np.ndarray, threshold: float = CLASSIFIER_THRESHOLD) -> Dict[str, Any]:
    """
    Classify fingerprint as clean or distorted using trained FingerprintCNN.
    Expects raw single-channel grayscale image (not altered by histogram equalizers).
    Returns dictionary with full statistical breakdown:
      - prob_distorted: float in [0, 1]
      - prob_clean: float in [0, 1] (prob_clean + prob_distorted == 1.0)
      - is_distorted: bool
      - predicted_class: 'DISTORTED' or 'CLEAN'
      - threshold: float
    """
    if image is None:
        raise ValueError("Fingerprint image is None.")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_224 = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image_float = image_224.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_float).unsqueeze(0).unsqueeze(0)

    model = load_classifier()
    with torch.no_grad():
        logits = model(tensor.to(DEVICE))
        prob_dist = float(torch.sigmoid(logits).item())

    prob_clean = float(1.0 - prob_dist)
    is_distorted = bool(prob_dist >= threshold)
    pred_class = "DISTORTED" if is_distorted else "CLEAN"

    return {
        "prob_distorted": round(prob_dist, 4),
        "prob_clean": round(prob_clean, 4),
        "is_distorted": is_distorted,
        "predicted_class": pred_class,
        "threshold": threshold,
    }


# ============================================================
# DDRNET RECTIFICATION
# ============================================================

def load_ddrnet() -> DDRNet_DIR:
    global DDRNET

    if DDRNET is None:
        if not DDRNET_CHECKPOINT.exists():
            raise FileNotFoundError(f"DDRNet checkpoint not found: {DDRNET_CHECKPOINT}")

        DDRNET = DDRNet_DIR(dis_const=16)
        checkpoint = torch.load(DDRNET_CHECKPOINT, map_location=DEVICE, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        DDRNET.load_state_dict(state_dict)
        DDRNET.to(DEVICE)
        DDRNET.eval()

    return DDRNET


def rectify_with_ddrnet(image: np.ndarray) -> np.ndarray:
    """
    Rectify distorted fingerprint using frozen DDRNet_DIR model.
    Maintains full native biometric resolution:
    Downsamples to 224x224 only for CNN displacement field inference,
    then scales the displacement field and remaps the native-resolution image
    directly to preserve 100% of fine ridge frequencies and minutiae.
    """
    if image is None:
        raise ValueError("Fingerprint image is None.")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orig_h, orig_w = image.shape
    gray = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    mask = segmentation_coherence(gray, win_size=16, stride=8)
    mask16 = zoom(mask, 1 / 16, order=0)

    normalized = np.float32((255.0 - gray) / 255.0)
    x = normalized[None, None, :, :]
    mask_tensor = np.float32(mask16)[None, None, :, :]

    model = load_ddrnet()
    with torch.no_grad():
        field, _ = model(
            torch.from_numpy(x).to(DEVICE),
            torch.from_numpy(mask_tensor).to(DEVICE),
        )

    field = field[0].cpu().numpy()

    # Scale displacement field directly to original image dimensions
    scale_x = orig_w / 224.0
    scale_y = orig_h / 224.0
    dx_orig = cv2.resize(field[0], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) * scale_x
    dy_orig = cv2.resize(field[1], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) * scale_y

    grid_x, grid_y = np.meshgrid(
        np.arange(orig_w, dtype=np.float32),
        np.arange(orig_h, dtype=np.float32),
    )

    map_x = grid_x - dx_orig
    map_y = grid_y - dy_orig

    rectified = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    return rectified


# ============================================================
# IMAGE SAVING
# ============================================================

def save_authentication_images(
    original_image: np.ndarray,
    rectified_image: Optional[np.ndarray],
    prefix: str = "query",
) -> Tuple[str, Optional[str]]:
    """Save query and rectified images for frontend preview."""
    safe_prefix = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in prefix)
    timestamp = int(time.time() * 1000)

    original_filename = f"{safe_prefix}_{timestamp}_orig.png"
    original_path = AUTH_OUTPUT_DIR / original_filename
    cv2.imwrite(str(original_path), original_image)

    rectified_url = None
    if rectified_image is not None:
        rectified_filename = f"{safe_prefix}_{timestamp}_rect.png"
        rectified_path = AUTH_OUTPUT_DIR / rectified_filename
        cv2.imwrite(str(rectified_path), rectified_image)
        rectified_url = str(rectified_path)

    return str(original_path), rectified_url


# ============================================================
# 1:N IDENTIFICATION WORKFLOW (PRIMARY USER FLOW)
# ============================================================

def identify_fingerprint(
    fingerprint_image: np.ndarray,
    top_n: int = 5,
    threshold: float = MATCH_THRESHOLD,
    force_rectification: bool = False,
) -> Dict[str, Any]:
    """
    Automatic 1:N Biometric Identification Workflow:
    1. Preprocessing
    2. Distortion classification (FingerprintCNN)
    3. DDRNet rectification (if distorted or forced)
    4. SourceAFIS template generation
    5. Search against stored MySQL SourceAFIS gallery templates
    6. Similarity scoring & Top-N ranking
    7. Threshold decision: MATCH FOUND or NO RELIABLE MATCH FOUND
    """
    total_t0 = time.time()

    if fingerprint_image is None or fingerprint_image.size == 0:
        raise ValueError("Fingerprint image is empty or invalid.")

    if fingerprint_image.ndim == 3:
        fingerprint_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)

    original_image = fingerprint_image.copy()

    # Timing metrics
    stage_times: Dict[str, float] = {}

    # Stage 1: Distortion classification directly on raw input image (as trained)
    t_cls = time.time()
    cls_res = classify_distortion(original_image)
    prob_distorted = cls_res["prob_distorted"]
    prob_clean = cls_res["prob_clean"]
    is_distorted = cls_res["is_distorted"]
    pred_class = cls_res["predicted_class"]
    decision_threshold = cls_res["threshold"]
    stage_times["classification_ms"] = round((time.time() - t_cls) * 1000.0, 2)

    final_decision = "RECTIFY" if (is_distorted or force_rectification) else "BYPASS"
    print(f"[DISTORTION DETECTION] class={pred_class} | clean_prob={prob_clean:.4f} | dist_prob={prob_distorted:.4f} | thresh={decision_threshold} | decision={final_decision} (forced={force_rectification})")

    # Stage 2: DDRNet rectification
    rectified_img = None
    rectification_applied = False
    t_rect = time.time()

    if is_distorted or force_rectification:
        rectified_img = rectify_with_ddrnet(original_image)
        rectification_applied = True
        fingerprint_for_template = rectified_img
    else:
        # If bypassed, output fingerprint is the original unmodified image
        rectified_img = original_image.copy()
        rectification_applied = False
        fingerprint_for_template = original_image

    stage_times["rectification_ms"] = round((time.time() - t_rect) * 1000.0, 2)

    # Stage 3: Save images for frontend display (always saves both panels)
    saved_orig, saved_rect = save_authentication_images(
        original_image=original_image,
        rectified_image=rectified_img,
        prefix="search",
    )

    # Stage 4: SourceAFIS template generation
    t_tpl = time.time()
    probe_template = MATCHER.extract_template(fingerprint_for_template)
    stage_times["template_generation_ms"] = round((time.time() - t_tpl) * 1000.0, 2)

    # Stage 5: Database search against enrolled templates
    t_search = time.time()
    gallery = get_all_gallery_templates()
    if not gallery:
        gallery = get_all_gallery_templates(force_refresh=True)

    search_result = MATCHER.identify_1toN(
        probe_template=probe_template,
        gallery=gallery,
        top_n=top_n,
        threshold=threshold,
    )
    stage_times["database_search_ms"] = round((time.time() - t_search) * 1000.0, 2)

    total_runtime_ms = round((time.time() - total_t0) * 1000.0, 2)

    best_match = search_result.get("best_match")
    match_found = bool(best_match and best_match["raw_score"] >= threshold)

    orig_url = f"/outputs/{os.path.basename(saved_orig)}" if saved_orig else None
    rect_url = f"/outputs/{os.path.basename(saved_rect)}" if saved_rect else orig_url

    return {
        "status": "success",
        "matcher": "SourceAFIS",
        "match_found": match_found,
        "match_status": "MATCH FOUND" if match_found else "NO RELIABLE MATCH FOUND",
        "predicted_class": pred_class,
        "clean_probability": prob_clean,
        "distorted_probability": prob_distorted,
        "distortion_probability": prob_distorted,
        "decision_threshold": decision_threshold,
        "is_distorted": is_distorted,
        "force_rectification": force_rectification,
        "rectification_applied": rectification_applied,
        "rectification_status": "APPLIED" if rectification_applied else "BYPASSED",
        "original_image_url": orig_url,
        "rectified_image_url": rect_url,
        "best_match": best_match,
        "candidates": search_result.get("candidates", []),
        "threshold": threshold,
        "processing_time_ms": total_runtime_ms,
        "stage_times": stage_times,
        "total_gallery_count": len(gallery),
    }


# ============================================================
# 1:1 AUTHENTICATION (BACKWARD COMPATIBILITY)
# ============================================================

def authenticate(
    identity_code: str,
    fingerprint_image: np.ndarray,
    mysql_password: Optional[str] = None,
) -> Dict[str, Any]:
    """
    1:1 Biometric Authentication using SourceAFIS.
    Replaces legacy Bozorth3 matching.
    """
    if not identity_code:
        raise ValueError("Identity code is required.")
    if fingerprint_image is None:
        raise ValueError("Fingerprint image is required.")

    if fingerprint_image.ndim == 3:
        fingerprint_image = cv2.cvtColor(fingerprint_image, cv2.COLOR_BGR2GRAY)

    original_image = fingerprint_image.copy()

    # 1. Distortion classification on raw image
    cls_res = classify_distortion(original_image)
    prob_distorted = cls_res["prob_distorted"]
    is_distorted = cls_res["is_distorted"]

    # 2. Rectification
    rectification_applied = False
    if is_distorted:
        rectified_image = rectify_with_ddrnet(original_image)
        fingerprint_for_matching = rectified_image
        rectification_applied = True
    else:
        rectified_image = original_image.copy()
        fingerprint_for_matching = original_image

    # 3. Save images
    saved_orig, saved_rect = save_authentication_images(
        original_image=original_image,
        rectified_image=rectified_image,
        prefix=identity_code,
    )

    # 4. SourceAFIS template generation
    probe_template = MATCHER.extract_template(fingerprint_for_matching)

    # 5. Retrieve enrolled templates from MySQL
    enrollment_data = get_person_templates(subject_id=identity_code)
    enrolled_templates = enrollment_data["templates"]

    # 6. SourceAFIS matching against enrolled impressions
    scores = []
    for tpl_record in enrolled_templates:
        res = MATCHER.match_1to1(probe_template, tpl_record["template"])
        scores.append(
            {
                "impression_id": tpl_record["impression_id"],
                "score": res.score,
                "raw_score": res.raw_score,
                "decision": res.match,
            }
        )

    best_match = max(scores, key=lambda x: x["raw_score"])
    authenticated = best_match["raw_score"] >= MATCH_THRESHOLD

    return {
        "authenticated": authenticated,
        "identity_code": identity_code,
        "distortion_probability": prob_distorted,
        "is_distorted": is_distorted,
        "rectification_applied": rectification_applied,
        "best_score": best_match["score"],
        "best_raw_score": best_match["raw_score"],
        "best_impression_id": best_match["impression_id"],
        "scores": scores,
        "saved_original_path": saved_orig,
        "saved_rectified_path": saved_rect,
    }


# ============================================================
# DIAGNOSTIC MINUTIAE & INSPECTION PIPELINE
# ============================================================

import afis.algorithms.segmentation as afis_seg
import afis.algorithms.orientation as afis_ori
import afis.algorithms.frequency as afis_freq
import afis.algorithms.enhancement as afis_enh
import afis.algorithms.skeleton as afis_skel
import afis.algorithms.minutiae as afis_minu
import math

def generate_diagnostic_overlay(image: np.ndarray) -> Dict[str, Any]:
    """
    Generate comprehensive visual diagnostic components:
    1. Original image
    2. Preprocessed & Gabor enhanced image
    3. DDRNet rectified image
    4. Extracted minutiae overlay (ridge endings and bifurcations)
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image for diagnostic overlay.")

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orig_h, orig_w = image.shape
    bg_ratio = float(np.mean(image > 200) * 100.0)
    polarity = "Dark ridges on light background" if bg_ratio > 50 else "Light ridges on dark background"

    # Standardize to biometric resolution
    proc_img = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)

    normed = (proc_img.astype(np.float64) - float(np.mean(proc_img))) / (float(np.std(proc_img)) + 1e-6)
    mask, _ = afis_seg.segment(normed, block_size=16)
    fg_pct = float(np.mean(mask)) * 100.0

    orientation = afis_ori.estimate_orientation(normed, block_size=16)
    frequency = afis_freq.estimate_frequency(normed, orientation, block_size=16)
    frequency = afis_freq.fill_invalid(frequency, mask)
    enhanced = afis_enh.enhance(normed, orientation, frequency, mask=mask, block_size=16, angle_steps=8)

    binary = afis_skel.binarize(enhanced, mask=mask, block_size=16)
    skeleton = afis_skel.skeletonize(binary)
    skeleton = afis_skel.clean_skeleton(skeleton)

    raw_minutiae = afis_minu.extract_minutiae(skeleton, orientation, mask=mask, border_margin=12)
    pruned = afis_minu.prune_false_minutiae(raw_minutiae, skeleton, mask=mask, min_distance=8.0, max_minutiae=80)
    minutiae_arr = afis_minu.minutiae_to_array(pruned)

    # Render color overlay
    overlay = cv2.cvtColor(proc_img, cv2.COLOR_GRAY2BGR)
    for m in minutiae_arr:
        x, y, theta = int(m[0]), int(m[1]), float(m[2])
        kind = int(m[3]) if m.shape[0] > 3 else 0
        color = (0, 0, 255) if kind == 0 else (0, 255, 0)
        cv2.circle(overlay, (x, y), 3, color, 1)
        tx = int(x + 10.0 * math.cos(theta))
        ty = int(y - 10.0 * math.sin(theta))
        cv2.line(overlay, (x, y), (tx, ty), color, 1)

    # Save diagnostic images
    timestamp = int(time.time() * 1000)
    orig_path = AUTH_OUTPUT_DIR / f"diag_{timestamp}_orig.png"
    enh_path = AUTH_OUTPUT_DIR / f"diag_{timestamp}_enh.png"
    over_path = AUTH_OUTPUT_DIR / f"diag_{timestamp}_overlay.png"

    def to_u8(arr):
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    cv2.imwrite(str(orig_path), image)
    cv2.imwrite(str(enh_path), to_u8(enhanced))
    cv2.imwrite(str(over_path), overlay)

    # DDRNet rectified image
    rect = rectify_with_ddrnet(image)
    rect_path = AUTH_OUTPUT_DIR / f"diag_{timestamp}_rect.png"
    cv2.imwrite(str(rect_path), rect)

    diagnosis = "Usable quality"
    if fg_pct < 10.0:
        diagnosis = "Excessive background margin / severe crop (genuine lack of usable fingerprint area)"
    elif len(minutiae_arr) < 5:
        diagnosis = "Low minutiae count (< 5 minutiae)"
    elif len(minutiae_arr) >= 20:
        diagnosis = "High quality biometric sample with sufficient minutiae support"

    return {
        "original_width": int(orig_w),
        "original_height": int(orig_h),
        "minutiae_count": int(len(minutiae_arr)),
        "foreground_area_pct": round(fg_pct, 2),
        "polarity": polarity,
        "diagnosis": diagnosis,
        "original_image_url": f"/outputs/{orig_path.name}",
        "preprocessed_image_url": f"/outputs/{enh_path.name}",
        "minutiae_overlay_url": f"/outputs/{over_path.name}",
        "rectified_image_url": f"/outputs/{rect_path.name}",
    }
"""
Evaluation Module: DDRNet Rectification & SourceAFIS Biometric Recognition.
Computes real ROC curves, ROC-AUC, EER, FAR, FRR, score distributions, and improvement outcomes.
No fabricated metrics; all results originate from actual execution.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support, accuracy_score

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from database import get_db_connection, MATCH_THRESHOLD
from matcher import MATCHER

METRICS_CACHE_FILE = CURRENT_DIR / "experiments" / "final_results" / "evaluation_metrics.json"
TEST_CLEAN_DIR = CURRENT_DIR / "data" / "split" / "test"
TEST_DISTORTED_DIR = CURRENT_DIR / "data" / "rectification_test" / "distorted"
TEST_RECTIFIED_DIR = CURRENT_DIR / "data" / "rectified_ddrnet_test"


def compute_eer(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (point where FPR == FNR).
    Returns (eer, eer_threshold).
    """
    fnr = 1.0 - tpr
    diffs = np.abs(fpr - fnr)
    idx = int(np.nanargmin(diffs))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    raw_thresh = float(thresholds[idx]) if idx < len(thresholds) else float(thresholds[-1])
    eer_thresh = raw_thresh if np.isfinite(raw_thresh) else 1.0
    return eer, eer_thresh


def run_evaluation_protocol(sample_limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Runs actual biometric evaluation comparing:
    PATH A: Distorted -> SourceAFIS
    PATH B: Distorted -> DDRNet Rectified -> SourceAFIS
    against Clean reference templates.
    """
    t_start = time.time()

    if not TEST_CLEAN_DIR.exists() or not TEST_DISTORTED_DIR.exists():
        raise FileNotFoundError(
            f"Evaluation dataset paths missing: {TEST_CLEAN_DIR} or {TEST_DISTORTED_DIR}"
        )

    # 1. Discover clean reference images
    clean_images: Dict[Tuple[str, str], List[Path]] = {}
    for fam_dir in sorted(TEST_CLEAN_DIR.glob("FAMILY-*")):
        for mem_dir in sorted(fam_dir.iterdir()):
            if mem_dir.is_dir():
                imgs = sorted(mem_dir.glob("*.png"))
                if imgs:
                    clean_images[(fam_dir.name, mem_dir.name)] = imgs

    # Extract or load clean reference templates
    clean_templates: Dict[str, Dict[str, Any]] = {}
    print(f"Extracting templates for {sum(len(v) for v in clean_images.values())} clean references...")
    for (fam, mem), paths in clean_images.items():
        # First impression as reference
        ref_path = paths[0]
        img = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            clean_templates[f"{fam}/{mem}"] = MATCHER.extract_template(img)

    # 2. Discover distorted images
    distorted_files = sorted(TEST_DISTORTED_DIR.glob("*.png"))
    if sample_limit and sample_limit > 0:
        distorted_files = distorted_files[:sample_limit]

    print(f"Running evaluation on {len(distorted_files)} test samples...")

    evaluation_records = []
    improved_count = 0
    degraded_count = 0
    unchanged_count = 0

    durations = []
    
    # Store scores and labels for Path A and Path B
    labels = []
    orig_scores = []
    rect_scores = []

    db_conn = get_db_connection()
    cursor = db_conn.cursor()

    insert_eval_sql = """
        INSERT INTO evaluation_results (
            original_score, rectified_score, original_match, rectified_match,
            score_diff, outcome, processing_time_ms, label, distortion_type
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
    """

    families = sorted(list(clean_images.keys()))

    for idx, dist_path in enumerate(distorted_files, 1):
        # Format: FAMILY-XX__MEMBER__FMXXX_XX__distortion.png
        parts = dist_path.stem.split("__")
        if len(parts) >= 4:
            fam_id = parts[0]
            mem_type = parts[1]
            dist_type = parts[3]
        else:
            fam_id = "FAMILY-13"
            mem_type = "CHILD"
            dist_type = "unknown"

        subject_key = f"{fam_id}/{mem_type}"
        clean_ref_template = clean_templates.get(subject_key)
        if not clean_ref_template:
            continue

        # Load distorted image
        dist_img = cv2.imread(str(dist_path), cv2.IMREAD_GRAYSCALE)
        if dist_img is None:
            continue

        # Load or generate rectified image
        rect_path = TEST_RECTIFIED_DIR / dist_path.name
        if rect_path.exists():
            rect_img = cv2.imread(str(rect_path), cv2.IMREAD_GRAYSCALE)
        else:
            from authentication import rectify_with_ddrnet
            rect_img = rectify_with_ddrnet(dist_img)

        # 1. Genuine comparison (distorted vs clean, rectified vs clean)
        t0 = time.time()
        dist_tpl = MATCHER.extract_template(dist_img)
        rect_tpl = MATCHER.extract_template(rect_img)

        match_orig = MATCHER.match_1to1(dist_tpl, clean_ref_template)
        match_rect = MATCHER.match_1to1(rect_tpl, clean_ref_template)
        pair_duration_ms = (time.time() - t0) * 1000.0
        durations.append(pair_duration_ms)

        diff = float(match_rect.score - match_orig.score)
        if diff > 0.001:
            outcome = "improved"
            improved_count += 1
        elif diff < -0.001:
            outcome = "degraded"
            degraded_count += 1
        else:
            outcome = "unchanged"
            unchanged_count += 1

        labels.append(1)  # Genuine
        orig_scores.append(match_orig.score)
        rect_scores.append(match_rect.score)

        cursor.execute(
            insert_eval_sql,
            (
                match_orig.score,
                match_rect.score,
                match_orig.match,
                match_rect.match,
                diff,
                outcome,
                pair_duration_ms,
                1,
                dist_type,
            )
        )

        # 2. Impostor comparison (distorted probe vs different family clean reference)
        # Select an impostor reference
        for other_fam, other_mem in families:
            if other_fam != fam_id and other_mem == mem_type:
                imp_key = f"{other_fam}/{other_mem}"
                imp_ref = clean_templates.get(imp_key)
                if imp_ref:
                    imp_orig = MATCHER.match_1to1(dist_tpl, imp_ref)
                    imp_rect = MATCHER.match_1to1(rect_tpl, imp_ref)

                    labels.append(0)  # Impostor
                    orig_scores.append(imp_orig.score)
                    rect_scores.append(imp_rect.score)

                    cursor.execute(
                        insert_eval_sql,
                        (
                            imp_orig.score,
                            imp_rect.score,
                            imp_orig.match,
                            imp_rect.match,
                            imp_rect.score - imp_orig.score,
                            "unchanged",
                            pair_duration_ms,
                            0,
                            dist_type,
                        )
                    )
                break

        if idx % 25 == 0 or idx == len(distorted_files):
            db_conn.commit()
            print(f"Evaluated {idx}/{len(distorted_files)} test samples...")

    db_conn.commit()
    cursor.close()
    db_conn.close()

    # Calculate ROC, ROC-AUC, EER, FAR, FRR
    y_true = np.array(labels)
    y_rect = np.array(rect_scores)
    y_orig = np.array(orig_scores)

    # Path B (Rectified) Metrics
    fpr_rect, tpr_rect, thresh_rect = roc_curve(y_true, y_rect)
    auc_rect = float(roc_auc_score(y_true, y_rect))
    eer_rect, eer_thresh_rect = compute_eer(fpr_rect, tpr_rect, thresh_rect)

    # Path A (Original) Metrics
    fpr_orig, tpr_orig, thresh_orig = roc_curve(y_true, y_orig)
    auc_orig = float(roc_auc_score(y_true, y_orig))
    eer_orig, eer_thresh_orig = compute_eer(fpr_orig, tpr_orig, thresh_orig)

    # Operating point metrics at configured MATCH_THRESHOLD
    y_pred_rect = (y_rect >= MATCH_THRESHOLD).astype(int)
    y_pred_orig = (y_orig >= MATCH_THRESHOLD).astype(int)

    acc_rect = float(accuracy_score(y_true, y_pred_rect))
    prec_rect, rec_rect, f1_rect, _ = precision_recall_fscore_support(
        y_true, y_pred_rect, average="binary", zero_division=0
    )

    # FAR = FP / (FP + TN)
    # FRR = FN / (FN + TP)
    fp = int(np.sum((y_pred_rect == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred_rect == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred_rect == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred_rect == 1) & (y_true == 1)))

    far = float(fp / max(1, fp + tn))
    frr = float(fn / max(1, fn + tp))

    # Match rates
    genuine_mask = (y_true == 1)
    orig_match_rate = float(np.mean(y_orig[genuine_mask] >= MATCH_THRESHOLD))
    rect_match_rate = float(np.mean(y_rect[genuine_mask] >= MATCH_THRESHOLD))

    # Score distributions (bins for charting)
    gen_rect_scores = y_rect[y_true == 1].tolist()
    imp_rect_scores = y_rect[y_true == 0].tolist()

    bins = np.linspace(0.0, 1.0, 21).tolist()
    gen_hist, _ = np.histogram(gen_rect_scores, bins=bins)
    imp_hist, _ = np.histogram(imp_rect_scores, bins=bins)

    total_time = time.time() - t_start
    avg_match_time_ms = float(np.mean(durations)) if durations else 0.0
    median_match_time_ms = float(np.median(durations)) if durations else 0.0

    results = {
        "status": "completed",
        "timestamp": time.time(),
        "total_pairs_evaluated": len(labels),
        "genuine_pairs": int(np.sum(y_true == 1)),
        "impostor_pairs": int(np.sum(y_true == 0)),
        "threshold": MATCH_THRESHOLD,
        "rectified_metrics": {
            "roc_auc": round(auc_rect, 4),
            "eer": round(eer_rect, 4),
            "eer_threshold": round(eer_thresh_rect, 4),
            "accuracy": round(acc_rect, 4),
            "precision": round(float(prec_rect), 4),
            "recall": round(float(rec_rect), 4),
            "f1_score": round(float(f1_rect), 4),
            "far": round(far, 4),
            "frr": round(frr, 4),
            "match_rate": round(rect_match_rate, 4),
        },
        "original_metrics": {
            "roc_auc": round(auc_orig, 4),
            "eer": round(eer_orig, 4),
            "eer_threshold": round(eer_thresh_orig, 4),
            "match_rate": round(orig_match_rate, 4),
        },
        "rectification_outcomes": {
            "improved": improved_count,
            "degraded": degraded_count,
            "unchanged": unchanged_count,
        },
        "runtime": {
            "avg_match_time_ms": round(avg_match_time_ms, 2),
            "median_match_time_ms": round(median_match_time_ms, 2),
            "total_evaluation_time_s": round(total_time, 2),
        },
        "score_distribution": {
            "bin_edges": [round(b, 2) for b in bins],
            "genuine_counts": gen_hist.tolist(),
            "impostor_counts": imp_hist.tolist(),
        },
        "roc_curve": {
            "fpr": [round(float(v), 4) for v in fpr_rect[::max(1, len(fpr_rect) // 40)]],
            "tpr": [round(float(v), 4) for v in tpr_rect[::max(1, len(tpr_rect) // 40)]],
            "thresholds": [
                round(float(min(1.0, float(v))), 4) if np.isfinite(v) else 1.0
                for v in thresh_rect[::max(1, len(thresh_rect) // 40)]
            ],
        },
    }

    # Save to JSON cache
    METRICS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def get_cached_metrics() -> Dict[str, Any]:
    """Retrieve cached metrics or query MySQL evaluation_results."""
    if METRICS_CACHE_FILE.exists():
        try:
            with open(METRICS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # Check if records exist in MySQL
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) AS cnt FROM evaluation_results;")
        row = cursor.fetchone()
        count = row["cnt"] if row else 0
        cursor.close()
        conn.close()

        if count == 0:
            return {"status": "Evaluation not run"}
    except Exception:
        return {"status": "Evaluation not run"}

    return {"status": "Evaluation not run"}


if __name__ == "__main__":
    print("Testing evaluation on sample batch...")
    res = run_evaluation_protocol(sample_limit=20)
    print("Evaluation completed successfully:")
    print("ROC-AUC:", res["rectified_metrics"]["roc_auc"])
    print("EER:", res["rectified_metrics"]["eer"])
    print("FAR:", res["rectified_metrics"]["far"])
    print("FRR:", res["rectified_metrics"]["frr"])
    print("Outcomes:", res["rectification_outcomes"])

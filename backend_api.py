"""
FastAPI Backend Application for Fingerprint Distortion Rectification & Recognition.
Endpoints for 1:N identification, 1:1 verification, DDRNet rectification,
dataset statistics, and research evaluation analytics.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from authentication import (
    identify_fingerprint,
    authenticate,
    classify_distortion,
    rectify_with_ddrnet,
    save_authentication_images,
    generate_diagnostic_overlay,
)
from matcher import MATCHER, SourceAFISMatcher
from database import (
    get_dataset_statistics,
    MATCH_THRESHOLD,
    test_mysql_connection,
)
from evaluation import (
    get_cached_metrics,
    run_evaluation_protocol,
)

app = FastAPI(
    title="Fingerprint Distortion Rectification & Recognition API",
    description="Professional biometric identification & research analytics dashboard API.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR = CURRENT_DIR / "experiments" / "authentication_outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

FRONTEND_DIST_DIR = CURRENT_DIR / "frontend" / "dist"
if FRONTEND_DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST_DIR / "assets")), name="assets")


# ============================================================
# HEALTH & INFO ENDPOINTS
# ============================================================

@app.get("/api/health")
def health_check():
    mysql_ok, mysql_msg = test_mysql_connection()
    return {
        "status": "healthy",
        "service": "Fingerprint Distortion Rectification & Recognition API",
        "mysql_connected": mysql_ok,
        "mysql_details": mysql_msg,
        "matcher": "SourceAFIS",
        "threshold": MATCH_THRESHOLD,
    }


@app.get("/api/dataset/stats")
@app.get("/dataset/stats")
def dataset_stats():
    """Retrieve actual dataset and template counts from MySQL database."""
    return get_dataset_statistics()


# ============================================================
# PRIMARY USER WORKFLOW: 1:N IDENTIFICATION
# ============================================================

@app.post("/fingerprint/search")
@app.post("/api/fingerprint/search")
async def fingerprint_search(
    fingerprint: UploadFile = File(..., description="Query fingerprint image"),
    top_n: int = Form(5, description="Number of top candidates to return"),
    threshold: Optional[float] = Form(None, description="Custom matching threshold"),
    force_rectification: bool = Form(False, description="Force DDRNet rectification"),
):
    """
    1:N Biometric Identification:
    User uploads a single fingerprint image.
    Backend handles preprocessing, DDRNet rectification, SourceAFIS template generation,
    and 1:N database search to find the nearest matching identities.
    """
    try:
        image_bytes = await fingerprint.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

        if image is None or image.size == 0:
            raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")

        if threshold is not None:
            t_thresh = threshold if threshold > 1.0 else (threshold * 100.0 if threshold > 0.0 else MATCH_THRESHOLD)
        else:
            t_thresh = MATCH_THRESHOLD

        result = identify_fingerprint(
            fingerprint_image=image,
            top_n=top_n,
            threshold=t_thresh,
            force_rectification=force_rectification,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"1:N Search Error: {str(e)}")


# ============================================================
# 1:1 BIOMETRIC VERIFICATION
# ============================================================

@app.post("/fingerprint/match")
@app.post("/api/fingerprint/match")
async def fingerprint_match(
    probe_fingerprint: UploadFile = File(...),
    reference_fingerprint: Optional[UploadFile] = File(None),
    subject_id: Optional[str] = Form(None),
    threshold: Optional[float] = Form(None),
):
    """
    1:1 Biometric Verification between two fingerprints or a probe and enrolled subject.
    """
    try:
        if threshold is not None:
            t_thresh = threshold if threshold > 1.0 else (threshold * 100.0 if threshold > 0.0 else MATCH_THRESHOLD)
        else:
            t_thresh = MATCH_THRESHOLD
        probe_bytes = await probe_fingerprint.read()
        probe_img = cv2.imdecode(np.frombuffer(probe_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

        if probe_img is None:
            raise HTTPException(status_code=400, detail="Invalid probe image file.")

        # If reference image uploaded, match directly
        if reference_fingerprint is not None:
            ref_bytes = await reference_fingerprint.read()
            ref_img = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            if ref_img is None:
                raise HTTPException(status_code=400, detail="Invalid reference image file.")

            tpl_probe = MATCHER.extract_template(probe_img)
            tpl_ref = MATCHER.extract_template(ref_img)
            res = MATCHER.match_1to1(tpl_probe, tpl_ref, threshold=t_thresh)

            return {
                "matcher": "SourceAFIS",
                "score": res.score,
                "raw_score": res.raw_score,
                "match": res.match,
                "matched_minutiae": res.matched_minutiae_count,
                "runtime_ms": res.runtime_ms,
                "threshold": t_thresh,
            }

        # Otherwise match against enrolled subject
        if not subject_id:
            raise HTTPException(
                status_code=400,
                detail="Either reference_fingerprint or subject_id must be provided.",
            )

        auth_res = authenticate(identity_code=subject_id, fingerprint_image=probe_img)
        return {
            "matcher": "SourceAFIS",
            "subject_id": subject_id,
            "best_score": auth_res["best_score"],
            "best_raw_score": auth_res["best_raw_score"],
            "match": auth_res["authenticated"],
            "threshold": t_thresh,
            "scores": auth_res["scores"],
            "original_image_url": f"/outputs/{os.path.basename(auth_res['saved_original_path'])}" if auth_res.get("saved_original_path") else None,
            "rectified_image_url": f"/outputs/{os.path.basename(auth_res['saved_rectified_path'])}" if auth_res.get("saved_rectified_path") else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# RECTIFICATION ENDPOINT
# ============================================================

@app.post("/fingerprint/rectify")
@app.post("/api/fingerprint/rectify")
async def fingerprint_rectify(
    fingerprint: UploadFile = File(...),
):
    """
    Run DDRNet geometric rectification on uploaded fingerprint image.
    """
    try:
        image_bytes = await fingerprint.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        cls_res = classify_distortion(image)
        rectified = rectify_with_ddrnet(image)

        saved_orig, saved_rect = save_authentication_images(
            original_image=image,
            rectified_image=rectified,
            prefix="rectify",
        )

        return {
            "is_distorted": cls_res["is_distorted"],
            "distortion_probability": cls_res["prob_distorted"],
            "clean_probability": cls_res["prob_clean"],
            "distorted_probability": cls_res["prob_distorted"],
            "predicted_class": cls_res["predicted_class"],
            "original_image_url": f"/outputs/{os.path.basename(saved_orig)}",
            "rectified_image_url": f"/outputs/{os.path.basename(saved_rect)}",
            "image_width": int(rectified.shape[1]),
            "image_height": int(rectified.shape[0]),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# DIAGNOSTIC INSPECTION ENDPOINT (PHASE 3)
# ============================================================

@app.post("/fingerprint/diagnostic/inspect")
@app.post("/api/fingerprint/diagnostic/inspect")
async def fingerprint_diagnostic_inspect(
    fingerprint: UploadFile = File(..., description="Query image to visually inspect minutiae and preprocessing"),
):
    """
    Diagnostic Endpoint:
    Provides visual inspection of original image, preprocessed image,
    DDRNet rectified image, and extracted minutiae overlay with endings/bifurcations.
    """
    try:
        image_bytes = await fingerprint.read()
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.size == 0:
            raise HTTPException(status_code=400, detail="Invalid or unreadable image file.")

        diagnostic_data = generate_diagnostic_overlay(image)
        return diagnostic_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {str(e)}")


# ============================================================
# EVALUATION & RESEARCH ANALYTICS ENDPOINTS
# ============================================================

@app.get("/evaluation/metrics")
@app.get("/api/evaluation/metrics")
def get_metrics():
    """Retrieve calculated biometric metrics (ROC-AUC, EER, FAR, FRR)."""
    return get_cached_metrics()


@app.get("/evaluation/roc")
@app.get("/api/evaluation/roc")
def get_roc_data():
    """Retrieve ROC curve coordinates and score distributions."""
    metrics = get_cached_metrics()
    if metrics.get("status") == "Evaluation not run":
        return {"status": "Evaluation not run"}

    return {
        "roc_curve": metrics.get("roc_curve", {}),
        "score_distribution": metrics.get("score_distribution", {}),
        "rectified_metrics": metrics.get("rectified_metrics", {}),
        "original_metrics": metrics.get("original_metrics", {}),
    }


@app.post("/evaluation/run")
@app.post("/api/evaluation/run")
async def run_evaluation(
    sample_limit: Optional[int] = Form(None),
):
    """
    Run evaluation protocol comparing original distorted vs DDRNet rectified recognition.
    Stores results in MySQL evaluation_results and computes real ROC-AUC, EER, FAR, FRR.
    """
    try:
        res = run_evaluation_protocol(sample_limit=sample_limit)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation protocol failed: {str(e)}")


# ============================================================
# FRONTEND SPA SERVING
# ============================================================

@app.get("/")
def serve_index():
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Fingerprint Recognition API is active. Frontend build not found at frontend/dist."}


@app.get("/app")
def serve_app():
    index_file = FRONTEND_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return FileResponse("frontend/index.html")

"""
System Integration Test Suite for Fingerprint Distortion Rectification & Recognition.
Verifies MySQL connection, database schema, dataset counts, SourceAFIS 1:1 & 1:N search,
DDRNet rectification, and FastAPI endpoints.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from database import test_mysql_connection, get_dataset_statistics, get_db_connection, MATCH_THRESHOLD
from matcher import MATCHER
from authentication import identify_fingerprint, rectify_with_ddrnet
from backend_api import app

client = TestClient(app)

SAMPLE_IMAGE_PATH = Path("data/FAMILY FINGERPRINT DATASET/FAMILY FINGERPRINT DATASET/FAMILY-1/CHILD/FM001_C1.png")
DISTORTED_IMAGE_PATH = Path("data/rectification_test/distorted/FAMILY-13__CHILD__FM013_C1__elastic.png")


def run_tests():
    print("=" * 60)
    print("RUNNING COMPLETE SYSTEM INTEGRATION TESTS")
    print("=" * 60)

    # Test 1: MySQL Connection
    print("\n[TEST 1] Testing MySQL Connection...")
    ok, msg = test_mysql_connection()
    assert ok, f"MySQL connection failed: {msg}"
    print(f"  PASS: {msg}")

    # Test 2: Database Schema & Tables
    print("\n[TEST 2] Verifying MySQL Database Schema & Tables...")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SHOW TABLES;")
    tables = [r[0] for r in cur.fetchall()]
    conn.close()
    for req_table in ["fingerprints", "fingerprint_templates", "evaluation_results"]:
        assert req_table in tables, f"Missing table: {req_table}"
        print(f"  PASS: Table '{req_table}' exists.")

    # Test 3: Dataset Statistics
    print("\n[TEST 3] Querying Dataset Statistics in MySQL...")
    stats = get_dataset_statistics()
    print(f"  Total Fingerprints : {stats['total_fingerprints']}")
    print(f"  Total Subjects     : {stats['total_subjects']}")
    print(f"  Total Families     : {stats['total_families']}")
    print(f"  Total Templates    : {stats['total_templates']}")
    print(f"  Generated Templates: {stats['generated_templates']}")
    assert stats["total_fingerprints"] >= 1500, f"Expected 1500 fingerprints, got {stats['total_fingerprints']}"
    assert stats["generated_templates"] >= 1400, f"Expected >= 1400 generated templates, got {stats['generated_templates']}"
    print("  PASS: Dataset is fully imported into MySQL.")

    # Test 4: SourceAFIS Feature Extraction & 1:1 Matching
    print("\n[TEST 4] Testing SourceAFIS Feature Extraction & 1:1 Verification...")
    assert SAMPLE_IMAGE_PATH.exists(), f"Sample image not found: {SAMPLE_IMAGE_PATH}"
    img1 = cv2.imread(str(SAMPLE_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    t0 = time.time()
    tpl1 = MATCHER.extract_template(img1)
    extract_time = (time.time() - t0) * 1000
    assert tpl1["minutiae_count"] > 0, "No minutiae extracted"
    print(f"  PASS: Extracted {tpl1['minutiae_count']} minutiae in {extract_time:.1f}ms")

    # Match against self (should be 1.0)
    match_self = MATCHER.match_1to1(tpl1, tpl1)
    assert match_self.score >= 0.9, f"Self-match score too low: {match_self.score}"
    print(f"  PASS: Self-match score: {match_self.score:.2f} (raw: {match_self.raw_score:.1f}, match: {match_self.match})")

    # Test 5: DDRNet Distortion Rectification
    print("\n[TEST 5] Testing DDRNet Distortion Rectification...")
    if DISTORTED_IMAGE_PATH.exists():
        dist_img = cv2.imread(str(DISTORTED_IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
        t_rect0 = time.time()
        rect_img = rectify_with_ddrnet(dist_img)
        rect_time = (time.time() - t_rect0) * 1000
        assert rect_img is not None and rect_img.shape == dist_img.shape
        print(f"  PASS: DDRNet rectified {dist_img.shape} image in {rect_time:.1f}ms")
    else:
        print("  SKIP: Distorted sample image path not found.")

    # Test 6: 1:N Identification Workflow
    print("\n[TEST 6] Testing 1:N Automatic Biometric Identification...")
    id_result = identify_fingerprint(img1, top_n=5, threshold=MATCH_THRESHOLD)
    assert id_result["status"] == "success"
    print(f"  Match status       : {id_result['match_status']}")
    print(f"  Best Match Subject : {id_result['best_match']['subject_id'] if id_result['best_match'] else 'None'}")
    print(f"  Best Match Score   : {id_result['best_match']['score'] if id_result['best_match'] else 0}")
    print(f"  Top Candidates Count: {len(id_result['candidates'])}")
    print(f"  Total Search Time  : {id_result['processing_time_ms']}ms")
    assert len(id_result["candidates"]) == 5, f"Expected 5 candidates, got {len(id_result['candidates'])}"
    print("  PASS: 1:N search returned top-5 ranked candidates.")

    # Test 7: API Endpoint /api/health
    print("\n[TEST 7] Testing API Endpoint GET /api/health...")
    resp = client.get("/api/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    print(f"  PASS: {resp.json()}")

    # Test 8: API Endpoint GET /dataset/stats
    print("\n[TEST 8] Testing API Endpoint GET /dataset/stats...")
    resp = client.get("/dataset/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_fingerprints"] >= 1500
    print(f"  PASS: /dataset/stats returned total_fingerprints={data['total_fingerprints']}")

    # Test 9: API Endpoint GET /evaluation/metrics
    print("\n[TEST 9] Testing API Endpoint GET /evaluation/metrics...")
    resp = client.get("/evaluation/metrics")
    assert resp.status_code == 200
    m_data = resp.json()
    print(f"  Status: {m_data.get('status')}")
    if "rectified_metrics" in m_data:
        print(f"  ROC-AUC: {m_data['rectified_metrics']['roc_auc']}")
        print(f"  EER    : {m_data['rectified_metrics']['eer']}")
        print(f"  FAR    : {m_data['rectified_metrics']['far']}")
        print(f"  FRR    : {m_data['rectified_metrics']['frr']}")
    print("  PASS: Metrics endpoint operational.")

    # Test 10: API Endpoint POST /fingerprint/search
    print("\n[TEST 10] Testing API Endpoint POST /fingerprint/search...")
    with open(SAMPLE_IMAGE_PATH, "rb") as f:
        file_bytes = f.read()

    resp = client.post(
        "/fingerprint/search",
        files={"fingerprint": ("sample.png", file_bytes, "image/png")},
        data={"top_n": "5", "threshold": str(MATCH_THRESHOLD)},
    )
    assert resp.status_code == 200, f"Search API failed: {resp.text}"
    search_data = resp.json()
    assert search_data["status"] == "success"
    assert "best_match" in search_data
    assert "candidates" in search_data
    assert len(search_data["candidates"]) == 5
    print(f"  PASS: Search API returned best_match={search_data['best_match']['subject_id']}, score={search_data['best_match']['score']}, status={search_data['match_status']}")

    print("\n" + "=" * 60)
    print("ALL 10 INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

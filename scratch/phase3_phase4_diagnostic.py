"""
Phase 3 & Phase 4: Raw vs DDRNet 1:N Search Diagnostic
Runs 1:N search against the complete 1,484 gallery in MySQL for:
FM001_F1_bending.png
Under two conditions:
Condition A (Phase 3): Raw Query (No DDRNet) -> SourceAFIS -> 1:N Gallery Search
Condition B (Phase 4): DDRNet Rectified Query -> SourceAFIS -> 1:N Gallery Search

Records and compares the Top 20 candidates for each.
Calculates displacement field metrics (min/max/mean dx and dy) and image differences (MAE, MSE).
"""

import os
import sys
import time
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from matcher import SourceAFISMatcher
from authentication import (
    rectify_with_ddrnet,
    get_all_gallery_templates,
)

print("=" * 80)
print("PHASE 3 & 4: RAW VS DDRNET 1:N SEARCH DIAGNOSTIC (FM001_F1_bending.png)")
print("=" * 80)

# Load query image
query_path = os.path.join(
    PROJECT_ROOT, "DATA", "train_distorted", "distorted",
    "FAMILY-1", "FATHER", "FM001_F1_bending.png"
)
assert os.path.exists(query_path), f"Query image not found: {query_path}"
img_raw = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
print(f"Raw query image dimensions: {img_raw.shape[1]}x{img_raw.shape[0]} | std: {img_raw.std():.2f}")

# Load gallery templates
t0 = time.time()
gallery = get_all_gallery_templates(force_refresh=True)
print(f"Loaded {len(gallery)} gallery templates from MySQL in {(time.time() - t0)*1000.0:.1f} ms")

matcher = SourceAFISMatcher()

# ----------------------------------------------------
# CONDITION A (PHASE 3): RAW IMAGE -> SourceAFIS
# ----------------------------------------------------
print("\n" + "="*40)
print("CONDITION A: RAW IMAGE -> SOURCEAFIS (Phase 3)")
print("="*40)
t_tpl_raw0 = time.time()
tpl_raw = matcher.extract_template(img_raw)
t_tpl_raw = (time.time() - t_tpl_raw0) * 1000.0
print(f"Raw template minutiae: {tpl_raw['minutiae_count']} (extracted in {t_tpl_raw:.1f} ms)")

t_search_raw0 = time.time()
res_raw = matcher.identify_1toN(tpl_raw, gallery, top_n=20)
t_search_raw = (time.time() - t_search_raw0) * 1000.0
print(f"Raw 1:N search completed in {t_search_raw:.1f} ms")

print(f"\n{'Rank':<5} | {'Filename':<16} | {'Family':<12} | {'Member':<8} | {'Raw Score':<10} | {'Matched Minutiae'}")
print("-" * 75)
for idx, cand in enumerate(res_raw["candidates"]):
    print(f"#{idx+1:<4} | {cand.get('filename', '—'):<16} | {cand.get('family_id', '—'):<12} | {cand.get('member_type', '—'):<8} | {cand['raw_score']:<10.1f} | {cand['matched_minutiae']}")

# ----------------------------------------------------
# CONDITION B (PHASE 4): DDRNET -> SourceAFIS
# ----------------------------------------------------
print("\n" + "="*40)
print("CONDITION B: DDRNET RECTIFIED -> SOURCEAFIS (Phase 4)")
print("="*40)

t_rect0 = time.time()
img_rect = rectify_with_ddrnet(img_raw)
t_rect = (time.time() - t_rect0) * 1000.0
print(f"DDRNet rectification completed in {t_rect:.1f} ms")

# Objective image difference
diff = np.abs(img_rect.astype(float) - img_raw.astype(float))
mae = np.mean(diff)
mse = np.mean(diff ** 2)
changed_pct = np.count_nonzero(diff > 0) / diff.size * 100.0
print(f"Image Difference: MAE = {mae:.2f}, MSE = {mse:.2f}, Changed Pixels = {changed_pct:.1f}%")

t_tpl_rect0 = time.time()
tpl_rect = matcher.extract_template(img_rect)
t_tpl_rect = (time.time() - t_tpl_rect0) * 1000.0
print(f"Rectified template minutiae: {tpl_rect['minutiae_count']} (extracted in {t_tpl_rect:.1f} ms)")

t_search_rect0 = time.time()
res_rect = matcher.identify_1toN(tpl_rect, gallery, top_n=20)
t_search_rect = (time.time() - t_search_rect0) * 1000.0
print(f"Rectified 1:N search completed in {t_search_rect:.1f} ms")

print(f"\n{'Rank':<5} | {'Filename':<16} | {'Family':<12} | {'Member':<8} | {'Raw Score':<10} | {'Matched Minutiae'}")
print("-" * 75)
for idx, cand in enumerate(res_rect["candidates"]):
    print(f"#{idx+1:<4} | {cand.get('filename', '—'):<16} | {cand.get('family_id', '—'):<12} | {cand.get('member_type', '—'):<8} | {cand['raw_score']:<10.1f} | {cand['matched_minutiae']}")

print("=" * 80)

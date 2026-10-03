"""
Phase 1: SourceAFIS Self-Match & Basic Verification
Tests:
1. Self-match of FM001_F1_bending.png (template A vs template B from exact same image).
2. Comparison against the false-positive candidates:
   - FM010_C1.png
   - FM074_M2.png
   - FM008_M2.png
   - FM028_C4.png
   - FM044_M3.png
3. Check minutiae count, matched minutiae, and raw SourceAFIS scores.
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

matcher = SourceAFISMatcher()

# Paths to the specific images
query_path = os.path.join(
    PROJECT_ROOT, "DATA", "train_distorted", "distorted",
    "FAMILY-1", "FATHER", "FM001_F1_bending.png"
)

# Search for the candidate files in dataset
candidate_names = [
    "FM010_C1.png",
    "FM074_M2.png",
    "FM008_M2.png",
    "FM028_C4.png",
    "FM044_M3.png",
]

candidate_paths = {}
for root, dirs, files in os.walk(os.path.join(PROJECT_ROOT, "DATA")):
    for f in files:
        if f in candidate_names and f not in candidate_paths:
            candidate_paths[f] = os.path.join(root, f)

print("=" * 70)
print("PHASE 1: SOURCEAFIS SELF-MATCH AND FALSE POSITIVE COMPARISON")
print("=" * 70)

print(f"Query Image: {query_path}")
assert os.path.exists(query_path), f"Query image not found: {query_path}"

img_query = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
print(f"Query dimensions: {img_query.shape[1]}x{img_query.shape[0]} | dtype: {img_query.dtype}")

# 1. Self Match
t0 = time.time()
tpl_a = matcher.extract_template(img_query)
tpl_b = matcher.extract_template(img_query)
t_extract = (time.time() - t0) * 1000.0

print(f"Template A minutiae count: {tpl_a['minutiae_count']}")
print(f"Template B minutiae count: {tpl_b['minutiae_count']}")
print(f"Template extraction time (2x): {t_extract:.2f} ms")

self_res = matcher.match_1to1(tpl_a, tpl_b)
print(f"\n--- SELF MATCH RESULT ---")
print(f"Raw SourceAFIS Score: {self_res.raw_score}")
print(f"Matched Minutiae: {self_res.matched_minutiae_count} / {tpl_a['minutiae_count']}")
print(f"Match Decision (threshold={self_res.threshold}): {self_res.match}")
print(f"Runtime: {self_res.runtime_ms:.2f} ms")

# 2. Compare against specific candidates
print(f"\n--- CANDIDATE COMPARISONS (Query vs Specific Candidates) ---")
print(f"{'Filename':<16} | {'Status':<10} | {'Raw Score':<10} | {'Matched Minutiae':<18} | {'Path'}")
print("-" * 85)

for c_name in candidate_names:
    p = candidate_paths.get(c_name)
    if not p or not os.path.exists(p):
        print(f"{c_name:<16} | NOT FOUND  | -          | -                  | -")
        continue
    c_img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    c_tpl = matcher.extract_template(c_img)
    m_res = matcher.match_1to1(tpl_a, c_tpl)
    print(f"{c_name:<16} | Found      | {m_res.raw_score:<10.1f} | {m_res.matched_minutiae_count:<18} | {os.path.relpath(p, PROJECT_ROOT)}")

# Also compare against genuine uncompressed clean father FM001_F1.png
clean_f1_path = os.path.join(
    PROJECT_ROOT, "DATA", "FAMILY FINGERPRINT DATASET", "FAMILY FINGERPRINT DATASET",
    "FAMILY-1", "FATHER", "FM001_F1.png"
)
if os.path.exists(clean_f1_path):
    f1_img = cv2.imread(clean_f1_path, cv2.IMREAD_GRAYSCALE)
    f1_tpl = matcher.extract_template(f1_img)
    gen_res = matcher.match_1to1(tpl_a, f1_tpl)
    print(f"\nGENUINE SAME-SUBJECT (FM001_F1_bending.png vs FM001_F1.png):")
    print(f"Raw SourceAFIS Score: {gen_res.raw_score} | Matched Minutiae: {gen_res.matched_minutiae_count} | Decision: {gen_res.match}")

print("=" * 70)

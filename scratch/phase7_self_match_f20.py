"""
Phase 7: Self-Match and Genuine/Impostor Test for FAMILY-20 C1
Tests:
TEST 1: FM020_C1 image -> SourceAFIS -> its own stored template
TEST 2: FM020_C1 image -> SourceAFIS -> same-subject different-capture template (e.g. FM020_C2.png)
TEST 3: FM020_C1 image -> SourceAFIS -> unrelated fingerprint (e.g. FM001_F1.png, FM096_C1.png)
Also tests with DDRNet applied.
"""

import os
import sys
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from matcher import SourceAFISMatcher
from authentication import rectify_with_ddrnet
from database import get_db_connection
import pickle

matcher = SourceAFISMatcher()

f20_c1_path = os.path.join(
    PROJECT_ROOT, "DATA", "FAMILY FINGERPRINT DATASET", "FAMILY FINGERPRINT DATASET",
    "FAMILY-20", "CHILD", "FM020_C1.png"
)
f20_c2_path = os.path.join(
    PROJECT_ROOT, "DATA", "FAMILY FINGERPRINT DATASET", "FAMILY FINGERPRINT DATASET",
    "FAMILY-20", "CHILD", "FM020_C2.png"
)
unrelated_path = os.path.join(
    PROJECT_ROOT, "DATA", "FAMILY FINGERPRINT DATASET", "FAMILY FINGERPRINT DATASET",
    "FAMILY-1", "FATHER", "FM001_F1.png"
)

assert os.path.exists(f20_c1_path), f"Missing {f20_c1_path}"
img_c1 = cv2.imread(f20_c1_path, cv2.IMREAD_GRAYSCALE)

# Load stored template for FM020_C1 from MySQL
conn = get_db_connection()
cursor = conn.cursor(dictionary=True)
cursor.execute("""
    SELECT t.template_data, f.filename
    FROM fingerprint_templates t
    JOIN fingerprints f ON t.fingerprint_id = f.fingerprint_id
    WHERE f.filename = 'FM020_C1.png'
""")
row_c1 = cursor.fetchone()
cursor.close()
conn.close()

stored_tpl_c1 = pickle.loads(row_c1["template_data"])

# Fresh template from image
probe_tpl = matcher.extract_template(img_c1)

print("=" * 80)
print("PHASE 7: SELF-MATCH & GENUINE/IMPOSTOR TESTS (FAMILY-20 C1)")
print("=" * 80)

# TEST 1: Image vs its own stored template
res_self = matcher.match_1to1(probe_tpl, stored_tpl_c1)
print(f"TEST 1 (Self Match: FM020_C1 image vs its own stored template):")
print(f"  - Raw SourceAFIS Score: {res_self.raw_score}")
print(f"  - Matched Minutiae: {res_self.matched_minutiae_count} / {probe_tpl['minutiae_count']}")
print(f"  - Match Decision: {res_self.match} (threshold={res_self.threshold})")

# TEST 2: Same-subject different capture
if os.path.exists(f20_c2_path):
    img_c2 = cv2.imread(f20_c2_path, cv2.IMREAD_GRAYSCALE)
    tpl_c2 = matcher.extract_template(img_c2)
    res_gen = matcher.match_1to1(probe_tpl, tpl_c2)
    print(f"\nTEST 2 (Same-Subject Different Capture: FM020_C1 vs FM020_C2):")
    print(f"  - Raw SourceAFIS Score: {res_gen.raw_score}")
    print(f"  - Matched Minutiae: {res_gen.matched_minutiae_count}")
    print(f"  - Match Decision: {res_gen.match}")

# TEST 3: Unrelated fingerprint
if os.path.exists(unrelated_path):
    img_unrel = cv2.imread(unrelated_path, cv2.IMREAD_GRAYSCALE)
    tpl_unrel = matcher.extract_template(img_unrel)
    res_imp = matcher.match_1to1(probe_tpl, tpl_unrel)
    print(f"\nTEST 3 (Unrelated Impostor: FM020_C1 vs FM001_F1):")
    print(f"  - Raw SourceAFIS Score: {res_imp.raw_score}")
    print(f"  - Matched Minutiae: {res_imp.matched_minutiae_count}")
    print(f"  - Match Decision: {res_imp.match}")

# TEST with DDRNet
rect_c1 = rectify_with_ddrnet(img_c1)
tpl_rect = matcher.extract_template(rect_c1)
res_rect_self = matcher.match_1to1(tpl_rect, stored_tpl_c1)
print(f"\nTEST WITH DDRNET (FM020_C1 DDRNet-rectified vs clean stored template):")
print(f"  - Raw SourceAFIS Score: {res_rect_self.raw_score}")
print(f"  - Matched Minutiae: {res_rect_self.matched_minutiae_count}")
print(f"  - Match Decision: {res_rect_self.match}")

print("=" * 80)

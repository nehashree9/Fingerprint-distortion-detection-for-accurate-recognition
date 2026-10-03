"""
test_biometric_sanity.py
Automated Biometric and Pipeline Sanity Test Suite
Validates all 10 critical checks defined in Section 21.

1. Same fingerprint vs itself (self-match)
2. Genuine same-subject pair
3. Known impostor pair
4. Family-1/Father query (FM001_F1_bending.png without force rectification)
5. Family-96/Child impostor comparison
6. 1:N search candidate ordering
7. Template uniqueness check across MySQL database
8. Metadata-template mapping consistency
9. DDRNet output generation and displacement field check
10. Score and threshold scale consistency
"""

import os
import sys
import hashlib
import unittest
import numpy as np
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from matcher import SourceAFISMatcher
from database import get_db_connection, MATCH_THRESHOLD
from authentication import (
    identify_fingerprint,
    classify_distortion,
    rectify_with_ddrnet,
    get_all_gallery_templates,
)


class BiometricSanityTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matcher = SourceAFISMatcher()
        cls.f1_path = os.path.join(
            PROJECT_ROOT,
            "DATA", "FAMILY FINGERPRINT DATASET", "FAMILY FINGERPRINT DATASET",
            "FAMILY-1", "FATHER", "FM001_F1.png"
        )
        cls.f2_path = os.path.join(
            PROJECT_ROOT,
            "DATA", "FAMILY FINGERPRINT DATASET", "FAMILY FINGERPRINT DATASET",
            "FAMILY-1", "FATHER", "FM001_F2.png"
        )
        cls.c96_path = os.path.join(
            PROJECT_ROOT,
            "DATA", "FAMILY FINGERPRINT DATASET", "FAMILY FINGERPRINT DATASET",
            "FAMILY-96", "CHILD", "FM096_C1.png"
        )
        cls.bending_path = os.path.join(
            PROJECT_ROOT,
            "DATA", "train_distorted", "distorted",
            "FAMILY-1", "FATHER", "FM001_F1_bending.png"
        )

        assert os.path.exists(cls.f1_path), f"Missing {cls.f1_path}"
        assert os.path.exists(cls.f2_path), f"Missing {cls.f2_path}"
        assert os.path.exists(cls.c96_path), f"Missing {cls.c96_path}"
        assert os.path.exists(cls.bending_path), f"Missing {cls.bending_path}"

        cls.img_f1 = cv2.imread(cls.f1_path, cv2.IMREAD_GRAYSCALE)
        cls.img_f2 = cv2.imread(cls.f2_path, cv2.IMREAD_GRAYSCALE)
        cls.img_c96 = cv2.imread(cls.c96_path, cv2.IMREAD_GRAYSCALE)
        cls.img_bending = cv2.imread(cls.bending_path, cv2.IMREAD_GRAYSCALE)

        cls.tpl_f1 = cls.matcher.extract_template(cls.img_f1)
        cls.tpl_f2 = cls.matcher.extract_template(cls.img_f2)
        cls.tpl_c96 = cls.matcher.extract_template(cls.img_c96)
        cls.tpl_bending = cls.matcher.extract_template(cls.img_bending)

    def test_01_self_match(self):
        """Sanity Check 1: Same fingerprint vs itself must yield a high genuine score."""
        res = self.matcher.match_1to1(self.tpl_f1, self.tpl_f1)
        print(f"\n[Test 1: Self Match] Raw Score: {res.raw_score}, Matched Minutiae: {res.matched_minutiae_count}")
        self.assertGreaterEqual(res.raw_score, 100.0, "Self match score should be >= 100.0")
        self.assertTrue(res.match, "Self match decision must be True")
        self.assertGreaterEqual(res.matched_minutiae_count, 15, "Self match should pair significant minutiae")

    def test_02_genuine_same_subject_pair(self):
        """Sanity Check 2: Genuine same-subject pair must score higher than impostors."""
        res_genuine = self.matcher.match_1to1(self.tpl_f1, self.tpl_f2)
        res_impostor = self.matcher.match_1to1(self.tpl_f1, self.tpl_c96)
        print(f"[Test 2: Genuine vs Impostor] Genuine (F1 vs F2): {res_genuine.raw_score} | Impostor (F1 vs C96): {res_impostor.raw_score}")
        self.assertGreater(
            res_genuine.raw_score,
            res_impostor.raw_score,
            "Genuine score must strictly exceed impostor score"
        )

    def test_03_known_impostor_pair(self):
        """Sanity Check 3: Known impostor pair must yield raw score below default threshold."""
        res = self.matcher.match_1to1(self.tpl_f1, self.tpl_c96)
        print(f"[Test 3: Known Impostor] Impostor Score: {res.raw_score} (Threshold: {MATCH_THRESHOLD})")
        self.assertLess(
            res.raw_score,
            MATCH_THRESHOLD,
            f"Impostor score {res.raw_score} must be below threshold {MATCH_THRESHOLD}"
        )
        self.assertFalse(res.match, "Impostor pair should not be classified as a match")

    def test_04_family1_father_query(self):
        """Sanity Check 4: FM001_F1_bending.png query must match FAMILY-1/FATHER."""
        gallery = get_all_gallery_templates()
        res = self.matcher.identify_1toN(
            probe_template=self.tpl_bending,
            gallery=gallery,
            top_n=5,
            threshold=MATCH_THRESHOLD,
        )
        best = res.get("best_match")
        print(f"[Test 4: Family-1/Father Raw Query] Best Match: {best.get('subject_id') if best else None}, Raw: {best.get('raw_score') if best else None}")
        self.assertIsNotNone(best, "Best match must not be None")
        self.assertIn("FAMILY-1", best["subject_id"], "Top candidate must be from FAMILY-1")
        self.assertIn("FATHER", best["subject_id"], "Top candidate must be FATHER")
        self.assertGreaterEqual(
            best["raw_score"],
            MATCH_THRESHOLD,
            f"Genuine query score {best['raw_score']} must meet threshold {MATCH_THRESHOLD}"
        )
        self.assertTrue(best["match"], "Search must confirm genuine match")

    def test_05_family96_child_impostor_comparison(self):
        """Sanity Check 5: FM001_F1_bending.png must score far higher on FAMILY-1/FATHER than FAMILY-96/CHILD."""
        res_genuine = self.matcher.match_1to1(self.tpl_bending, self.tpl_f1)
        res_impostor = self.matcher.match_1to1(self.tpl_bending, self.tpl_c96)
        print(f"[Test 5: Family-96 Impostor Comparison] Genuine (Bend vs F1): {res_genuine.raw_score} | Impostor (Bend vs C96): {res_impostor.raw_score}")
        self.assertGreater(
            res_genuine.raw_score,
            res_impostor.raw_score * 2.0,
            "Genuine score should be more than double the impostor score for Family-96/Child"
        )
        self.assertLess(
            res_impostor.raw_score,
            MATCH_THRESHOLD,
            "Family-96/Child score must remain below genuine threshold"
        )

    def test_06_search_candidate_ordering(self):
        """Sanity Check 6: 1:N candidate list must be strictly sorted by raw_score descending."""
        gallery = get_all_gallery_templates()
        res = self.matcher.identify_1toN(
            probe_template=self.tpl_bending,
            gallery=gallery,
            top_n=5,
            threshold=MATCH_THRESHOLD,
        )
        candidates = res.get("candidates", [])
        self.assertGreaterEqual(len(candidates), 2, "Candidate list should contain multiple entries")
        scores = [c["raw_score"] for c in candidates]
        print(f"[Test 6: Candidate Ordering] Ranked scores: {scores}")
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(
                scores[i],
                scores[i + 1],
                f"Candidate at rank {i+1} ({scores[i]}) must be >= rank {i+2} ({scores[i+1]})"
            )

    def test_07_template_uniqueness(self):
        """Sanity Check 7: All enrolled templates in MySQL must be unique (0 duplicates)."""
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT template_id, template_data FROM fingerprint_templates WHERE generation_status = 'generated'")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        self.assertGreaterEqual(len(rows), 1400, "Gallery should contain at least 1400 valid templates")
        hashes = set()
        duplicates = []
        for r in rows:
            h = hashlib.sha256(r["template_data"]).hexdigest()
            if h in hashes:
                duplicates.append(r["template_id"])
            hashes.add(h)

        print(f"[Test 7: Template Uniqueness] Verified {len(rows)} templates. Unique hashes: {len(hashes)}, Duplicates: {len(duplicates)}")
        self.assertEqual(len(duplicates), 0, f"Found duplicate template IDs: {duplicates}")

    def test_08_metadata_template_mapping(self):
        """Sanity Check 8: Metadata mapping in database must match expected family/subject structures."""
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT f.fingerprint_id, f.subject_id, f.family_id, f.member_type, f.filename, LENGTH(t.template_data) AS tpl_len
            FROM fingerprints f
            JOIN fingerprint_templates t ON f.fingerprint_id = t.fingerprint_id
            WHERE f.family_id IN ('FAMILY-1', 'FAMILY-96')
            ORDER BY f.fingerprint_id
            LIMIT 10
        """)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        self.assertGreater(len(rows), 0, "Should retrieve metadata rows")
        for r in rows:
            self.assertIn(r["member_type"], ["FATHER", "MOTHER", "CHILD"])
            self.assertTrue(r["subject_id"].startswith(r["family_id"]))
            self.assertGreater(r["tpl_len"], 10, "Template blob must not be empty")
        print(f"[Test 8: Metadata Mapping] Sample {len(rows)} rows validated successfully.")

    def test_09_ddrnet_output_generation(self):
        """Sanity Check 9: DDRNet must produce a valid transformed image of matching dimensions and dynamic range."""
        rectified = rectify_with_ddrnet(self.img_bending)
        self.assertIsNotNone(rectified, "Rectified image must not be None")
        self.assertEqual(rectified.shape, self.img_bending.shape, "Rectified shape must match original input")
        self.assertEqual(rectified.dtype, np.uint8, "Rectified image must be uint8")
        self.assertGreater(rectified.std(), 10.0, "Rectified image must have realistic contrast/variance")
        # Check that rectification altered pixels
        diff = np.abs(rectified.astype(float) - self.img_bending.astype(float))
        mean_diff = np.mean(diff)
        print(f"[Test 9: DDRNet Output] Input shape: {self.img_bending.shape} | Output shape: {rectified.shape} | Mean pixel delta: {mean_diff:.2f}")
        self.assertGreater(mean_diff, 1.0, "DDRNet output should exhibit non-zero deformation from input")

    def test_10_score_threshold_scale_consistency(self):
        """Sanity Check 10: Score and threshold scales must be consistent across matcher and database."""
        print(f"[Test 10: Score Scale Consistency] Matcher threshold: {self.matcher.default_threshold}, DB threshold: {MATCH_THRESHOLD}")
        self.assertEqual(
            self.matcher.default_threshold,
            MATCH_THRESHOLD,
            "Matcher default threshold should match MATCH_THRESHOLD"
        )
        self.assertGreaterEqual(MATCH_THRESHOLD, 30.0, "MATCH_THRESHOLD must be on raw score scale (>= 30.0)")
        self.assertLessEqual(MATCH_THRESHOLD, 100.0, "MATCH_THRESHOLD must be on raw score scale (<= 100.0)")


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING BIOMETRIC & PIPELINE SANITY TEST SUITE (10 CHECKS)")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(BiometricSanityTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nALL 10 SANITY CHECKS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\nSOME SANITY CHECKS FAILED.")
        sys.exit(1)

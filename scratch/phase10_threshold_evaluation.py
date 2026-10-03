"""
Phase 10: Biometric Threshold Calibration & ROC / EER Evaluation
Evaluates:
- Genuine score distribution
- Impostor score distribution
- FAR, FRR, EER across candidate thresholds [20, 25, 30, 35, 40, 45, 50, 55, 60]
- ROC-AUC
"""

import os
import sys
import random
import cv2
import numpy as np
from sklearn.metrics import roc_curve, auc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from matcher import SourceAFISMatcher
from database import get_db_connection
import pickle

matcher = SourceAFISMatcher()

# Load gallery templates from MySQL
conn = get_db_connection()
cursor = conn.cursor(dictionary=True)
cursor.execute("""
    SELECT f.fingerprint_id, f.subject_id, f.family_id, f.member_type, f.filename, t.template_data
    FROM fingerprints f
    JOIN fingerprint_templates t ON f.fingerprint_id = t.fingerprint_id
    WHERE t.generation_status = 'generated'
""")
db_rows = cursor.fetchall()
cursor.close()
conn.close()

templates_by_subject = {}
all_templates = []

for r in db_rows:
    try:
        tpl = pickle.loads(r["template_data"])
        item = {
            "fingerprint_id": r["fingerprint_id"],
            "subject_id": r["subject_id"],
            "filename": r["filename"],
            "template": tpl,
        }
        all_templates.append(item)
        templates_by_subject.setdefault(r["subject_id"], []).append(item)
    except Exception:
        continue

print(f"Loaded {len(all_templates)} valid templates across {len(templates_by_subject)} subjects.")

# Build genuine pairs (different impressions of the same subject)
genuine_scores = []
impostor_scores = []

# Collect genuine pairs
for subj, items in templates_by_subject.items():
    if len(items) >= 2:
        for i in range(len(items)):
            for j in range(i + 1, min(len(items), i + 3)):
                res = matcher.match_1to1(items[i]["template"], items[j]["template"])
                genuine_scores.append(res.raw_score)

# Collect impostor pairs (between different subjects)
random.seed(42)
subjects_list = list(templates_by_subject.keys())
for i in range(min(500, len(genuine_scores) * 3)):
    s1, s2 = random.sample(subjects_list, 2)
    item1 = random.choice(templates_by_subject[s1])
    item2 = random.choice(templates_by_subject[s2])
    res = matcher.match_1to1(item1["template"], item2["template"])
    impostor_scores.append(res.raw_score)

print(f"Evaluated {len(genuine_scores)} genuine pairs and {len(impostor_scores)} impostor pairs.")

# Compute statistics
gen_arr = np.array(genuine_scores)
imp_arr = np.array(impostor_scores)

print(f"\n--- SCORE DISTRIBUTIONS ---")
print(f"Genuine Scores:  Mean={gen_arr.mean():.2f}, Std={gen_arr.std():.2f}, Min={gen_arr.min():.1f}, Max={gen_arr.max():.1f}, Median={np.median(gen_arr):.1f}")
print(f"Impostor Scores: Mean={imp_arr.mean():.2f}, Std={imp_arr.std():.2f}, Min={imp_arr.min():.1f}, Max={imp_arr.max():.1f}, Median={np.median(imp_arr):.1f}")

# Threshold analysis table
candidate_thresholds = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0]
print(f"\n{'Threshold':<10} | {'FAR (%)':<10} | {'FRR (%)':<10} | {'Decision Accuracy (%)'}")
print("-" * 55)

for th in candidate_thresholds:
    far = float(np.mean(imp_arr >= th)) * 100.0
    frr = float(np.mean(gen_arr < th)) * 100.0
    acc = float((np.sum(gen_arr >= th) + np.sum(imp_arr < th)) / (len(gen_arr) + len(imp_arr))) * 100.0
    print(f"{th:<10.1f} | {far:<10.2f} | {frr:<10.2f} | {acc:<10.2f}")

# ROC and EER
y_true = np.concatenate([np.ones_like(gen_arr), np.zeros_like(imp_arr)])
y_scores = np.concatenate([gen_arr, imp_arr])

fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
fnr = 1.0 - tpr
eer_idx = np.nanargmin(np.abs(fpr - fnr))
eer = float(fpr[eer_idx])
eer_thresh = float(roc_thresholds[eer_idx])

print(f"\n--- BIOMETRIC EVALUATION METRICS ---")
print(f"ROC-AUC:       {roc_auc:.4f}")
print(f"EER:           {eer * 100.0:.2f}%")
print(f"EER Threshold: {eer_thresh:.2f}")

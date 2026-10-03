"""
Test Fast 1:N Search Architecture
"""
import os
import sys
import time
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from matcher import SourceAFISMatcher
import afis.algorithms.safis as safis
from authentication import get_all_gallery_templates

matcher = SourceAFISMatcher()
gallery = get_all_gallery_templates()

query_img = cv2.imread(r'DATA/train_distorted/distorted/FAMILY-1/FATHER/FM001_F1_bending.png', cv2.IMREAD_GRAYSCALE)
probe_tpl = matcher.extract_template(query_img)
probe_minu = probe_tpl['minutiae']

params = safis.SafisParams(max_roots=10)

# Pre-index gallery
for item in gallery:
    cand_tpl = item.get('template', item)
    if '_edges' not in cand_tpl:
        c_minu = cand_tpl.get('minutiae')
        if c_minu is not None and len(c_minu) >= 2:
            edges = safis.build_edges(c_minu, params)
            buckets = {}
            ltol = float(params.length_tol) or 1e-6
            for e in edges:
                k = int(round(e.length / ltol))
                buckets.setdefault(k, []).append(e)
            cand_tpl['_edges'] = edges
            cand_tpl['_buckets'] = buckets
        else:
            cand_tpl['_edges'] = []
            cand_tpl['_buckets'] = {}

# Time probe indexing
t0 = time.time()
ea = safis.build_edges(probe_minu, params)
length_tol = float(params.length_tol) or 1e-6
t_probe_idx = (time.time() - t0) * 1000.0

# Time 1:N search
t_search0 = time.time()
results = []
for item in gallery:
    cand_tpl = item.get('template', item)
    cand_edges = cand_tpl['_edges']
    cand_buckets = cand_tpl['_buckets']
    if not cand_edges:
        results.append((item, 0.0, 0))
        continue
    roots = []
    for pe in ea:
        key = int(round(pe.length / length_tol))
        for k in (key - 1, key, key + 1):
            b = cand_buckets.get(k)
            if not b: continue
            for ce in b:
                if safis._edges_compatible(pe, ce, params):
                    close = 1.0 - (
                        abs(pe.length - ce.length) / length_tol
                        + safis._angle_diff(pe.ref_angle, ce.ref_angle) / params.angle_tol
                        + safis._angle_diff(pe.neighbor_angle, ce.neighbor_angle) / params.angle_tol
                    ) / 3.0
                    roots.append((close, pe, ce))
    if not roots:
        results.append((item, 0.0, 0))
        continue
    roots.sort(key=lambda t: t[0], reverse=True)
    roots = roots[:params.max_roots]
    best_n = 0
    best_pairs = 0
    for _c, pe, ce in roots:
        pairing, pairs = safis._crawl(ea, cand_edges, pe.ref, ce.ref, pe.neighbor, ce.neighbor, params)
        if len(pairing) > best_n:
            best_n = len(pairing)
            best_pairs = len(pairs)
    raw = float(best_n) * 5.0
    if best_n >= 2:
        raw += float(best_pairs) * 1.5
    results.append((item, raw, best_pairs))

total_search_ms = (time.time() - t_search0) * 1000.0
results.sort(key=lambda x: x[1], reverse=True)

print(f"1:N Search across {len(gallery)} templates took: {total_search_ms:.2f} ms ({total_search_ms/len(gallery):.3f} ms/candidate)!")
print("Top 5 results:")
for i, (item, score, pc) in enumerate(results[:5]):
    fn = item.get('filename')
    subj = f"{item.get('family_id')}/{item.get('member_type')}"
    print(f"#{i+1}: {fn} | {subj} | Score: {score} | Pairs: {pc}")

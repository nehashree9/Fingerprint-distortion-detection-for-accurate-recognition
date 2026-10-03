"""
Benchmark 1:N Optimization
Compares:
1. Unoptimized (Repeated edge building, GIL thread contention)
2. Optimized SourceAFIS 1:N pattern:
   - ProbeMatcher indexes probe edges ONCE
   - Gallery templates cache invariant edges ONCE
   - Fast root edge pairing + tree crawl
Measures runtime for 100, 500, 1000, and 1484 templates.
"""

import time
import math
import numpy as np
import afis.algorithms.safis as safis
from authentication import get_all_gallery_templates
from matcher import SourceAFISMatcher

print("=" * 70)
print("SOURCEAFIS 1:N PATTERN BENCHMARK & OPTIMIZATION")
print("=" * 70)

# Load gallery
gallery = get_all_gallery_templates()
print(f"Total loaded gallery templates: {len(gallery)}")

# Extract probe template from query
import cv2
query_img = cv2.imread(r"DATA/train_distorted/distorted/FAMILY-1/FATHER/FM001_F1_bending.png", cv2.IMREAD_GRAYSCALE)
base_matcher = SourceAFISMatcher()
probe_tpl = base_matcher.extract_template(query_img)
probe_minu = probe_tpl["minutiae"]
print(f"Probe minutiae count: {probe_minu.shape[0]}")

params = safis.SafisParams(
    max_neighbors=8,
    max_edge_length=300.0,
    length_tol=8.0,
    angle_tol=0.35,
    max_roots=10,
)

# ---------------------------------------------------------
# OPTIMIZED MATCHER CLASS (SourceAFIS Pattern)
# ---------------------------------------------------------
class OptimizedProbeMatcher:
    def __init__(self, probe_minutiae: np.ndarray, params: safis.SafisParams):
        self.params = params
        self.probe_edges = safis.build_edges(probe_minutiae, params)
        self.length_tol = float(params.length_tol) or 1e-6
        self.angle_tol = params.angle_tol
        self.max_roots = params.max_roots

    def match(self, cand_edges: list, cand_buckets: dict) -> tuple[float, int]:
        if not self.probe_edges or not cand_edges:
            return 0.0, 0

        roots = []
        for pe in self.probe_edges:
            key = int(round(pe.length / self.length_tol))
            for k in (key - 1, key, key + 1):
                bucket = cand_buckets.get(k)
                if not bucket:
                    continue
                for ce in bucket:
                    if safis._edges_compatible(pe, ce, self.params):
                        close = 1.0 - (
                            abs(pe.length - ce.length) / self.length_tol
                            + safis._angle_diff(pe.ref_angle, ce.ref_angle) / self.angle_tol
                            + safis._angle_diff(pe.neighbor_angle, ce.neighbor_angle) / self.angle_tol
                        ) / 3.0
                        roots.append((close, pe, ce))

        if not roots:
            return 0.0, 0

        roots.sort(key=lambda t: t[0], reverse=True)
        roots = roots[: self.max_roots]

        best_n = 0
        best_pairs_count = 0
        for _close, pe, ce in roots:
            pairing, pairs = safis._crawl(
                self.probe_edges, cand_edges, pe.ref, ce.ref, pe.neighbor, ce.neighbor, self.params,
            )
            if len(pairing) > best_n:
                best_n = len(pairing)
                best_pairs_count = len(pairs)
                # Early exit if strong genuine match found
                if best_n >= 20:
                    break

        raw = float(best_n) * 5.0
        if best_n >= 2:
            raw += float(best_pairs_count) * 1.5

        return raw, best_pairs_count

# Pre-cache candidate edges & buckets in gallery
t_prep0 = time.time()
for item in gallery:
    cand_tpl = item.get("template", item)
    minu = cand_tpl.get("minutiae")
    if minu is not None and len(minu) >= 2:
        edges = safis.build_edges(minu, params)
        buckets = {}
        for e in edges:
            k = int(round(e.length / (params.length_tol or 1e-6)))
            buckets.setdefault(k, []).append(e)
        cand_tpl["_edges"] = edges
        cand_tpl["_buckets"] = buckets
    else:
        cand_tpl["_edges"] = []
        cand_tpl["_buckets"] = {}
t_prep = (time.time() - t_prep0) * 1000.0
print(f"Pre-indexing {len(gallery)} gallery templates took {t_prep:.1f} ms ({t_prep/len(gallery):.3f} ms/cand)")

# Create probe matcher ONCE
t_probe0 = time.time()
probe_matcher = OptimizedProbeMatcher(probe_minu, params)
t_probe = (time.time() - t_probe0) * 1000.0
print(f"ProbeMatcher initialization: {t_probe:.2f} ms")

# Benchmark for 100, 500, 1000, 1484 templates
bench_counts = [100, 500, 1000, len(gallery)]

print(f"\n{'N Templates':<12} | {'Total Time (ms)':<16} | {'Avg Time/Candidate (ms)':<25} | {'Candidates/sec'}")
print("-" * 75)

for n in bench_counts:
    subset = gallery[:n]
    t0 = time.time()
    results = []
    for item in subset:
        cand_tpl = item.get("template", item)
        score, pairs_cnt = probe_matcher.match(cand_tpl["_edges"], cand_tpl["_buckets"])
        results.append((item.get("filename"), score, pairs_cnt))
    elapsed_ms = (time.time() - t0) * 1000.0
    avg_ms = elapsed_ms / n
    fps = n / (elapsed_ms / 1000.0)
    print(f"{n:<12} | {elapsed_ms:<16.2f} | {avg_ms:<25.4f} | {fps:.1f}")

# Sort and check top 5 from the full 1484 search
results.sort(key=lambda x: x[1], reverse=True)
print(f"\nTop 5 Candidates with Optimized Matcher (1,484 templates):")
for r, (fname, sc, pc) in enumerate(results[:5]):
    print(f"#{r+1}: {fname} | Score: {sc} | Matched: {pc}")

print("=" * 70)

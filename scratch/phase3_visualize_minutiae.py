"""
Phase 3: Minutiae Quality Investigation and Visual Diagnostic Generator
Inspects and visualizes:
1. Original image
2. Preprocessed image
3. DDRNet rectified image (where applicable)
4. Extracted minutiae overlay (ridge endings/bifurcations, orientation vectors)
For:
A) FM001_F1_bending.png
B) FM020_C1.png (Family 20 successful)
C) FM044_F2.png (<=5 minutiae sample)
D) FM001_C1.png (high minutiae sample)

Saves detailed diagnostic composite images to experiments/diagnostic_minutiae_overlays/
"""

import os
import sys
import math
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from authentication import rectify_with_ddrnet
from matcher import SourceAFISMatcher
import afis.algorithms.segmentation as seg
import afis.algorithms.orientation as ori
import afis.algorithms.frequency as freq
import afis.algorithms.enhancement as enh
import afis.algorithms.skeleton as skel
import afis.algorithms.minutiae as minu

output_dir = os.path.join(PROJECT_ROOT, "experiments", "diagnostic_minutiae_overlays")
os.makedirs(output_dir, exist_ok=True)

samples = [
    ("sample_a_FM001_F1_bending", os.path.join(PROJECT_ROOT, "DATA", "train_distorted", "distorted", "FAMILY-1", "FATHER", "FM001_F1_bending.png"), True),
    ("sample_b_FM020_C1", os.path.join(PROJECT_ROOT, "DATA", "FAMILY FINGERPRINT DATASET", "FAMILY FINGERPRINT DATASET", "FAMILY-20", "CHILD", "FM020_C1.png"), False),
    ("sample_c_FM044_F2_low_minutiae", os.path.join(PROJECT_ROOT, "DATA", "FAMILY FINGERPRINT DATASET", "FAMILY FINGERPRINT DATASET", "FAMILY-44", "FATHER", "FM044_F2.png"), False),
    ("sample_d_FM001_C1_high_minutiae", os.path.join(PROJECT_ROOT, "DATA", "FAMILY FINGERPRINT DATASET", "FAMILY FINGERPRINT DATASET", "FAMILY-1", "CHILD", "FM001_C1.png"), False),
]

def analyze_and_render_overlay(image_gray: np.ndarray, title: str) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Runs full biometric pipeline:
    1. Resize to target size (384, 384)
    2. Segmentation (mask)
    3. Orientation & Frequency
    4. Gabor enhancement
    5. Binarization & Skeletonization
    6. Minutiae extraction & pruning
    Returns: (preprocessed_img, overlay_rgb, minutiae_count, foreground_area_pct)
    """
    target_img = cv2.resize(image_gray, (384, 384), interpolation=cv2.INTER_AREA)

    normed = (target_img.astype(np.float64) - float(np.mean(target_img))) / (float(np.std(target_img)) + 1e-6)

    mask, _ = seg.segment(normed, block_size=16)
    foreground_pct = float(np.mean(mask)) * 100.0

    orientation = ori.estimate_orientation(normed, block_size=16)
    frequency = freq.estimate_frequency(normed, orientation, block_size=16)
    frequency = freq.fill_invalid(frequency, mask)
    enhanced = enh.enhance(normed, orientation, frequency, mask=mask, block_size=16, angle_steps=8)

    binary = skel.binarize(enhanced, mask=mask, block_size=16)
    skeleton = skel.skeletonize(binary)
    skeleton = skel.clean_skeleton(skeleton)

    raw_minutiae = minu.extract_minutiae(skeleton, orientation, mask=mask, border_margin=12)
    pruned = minu.prune_false_minutiae(raw_minutiae, skeleton, mask=mask, min_distance=8.0, max_minutiae=80)
    minutiae_arr = minu.minutiae_to_array(pruned)

    # Render color overlay on enhanced image
    overlay = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)

    # Draw foreground mask contour
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)

    for m in minutiae_arr:
        x, y, theta = int(m[0]), int(m[1]), float(m[2])
        kind = int(m[3]) if m.shape[0] > 3 else 0

        # Red for Ending, Green for Bifurcation
        color = (0, 0, 255) if kind == 0 else (0, 255, 0)
        cv2.circle(overlay, (x, y), 3, color, 1)

        # Orientation tail
        tx = int(x + 10.0 * math.cos(theta))
        ty = int(y - 10.0 * math.sin(theta))
        cv2.line(overlay, (x, y), (tx, ty), color, 1)

    return enhanced, overlay, len(minutiae_arr), foreground_pct

print("=" * 80)
print("PHASE 3: MINUTIAE QUALITY INVESTIGATION")
print("=" * 80)

for tag, path, do_ddr in samples:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    orig_h, orig_w = img.shape

    enhanced_img, overlay, count, fg_pct = analyze_and_render_overlay(img, tag)

    print(f"\n[{tag}]")
    print(f"  - Source Path: {path}")
    print(f"  - Native Dimensions: {orig_w}x{orig_h}")
    print(f"  - Usable Foreground Area: {fg_pct:.1f}%")
    print(f"  - Extracted Minutiae Count: {count}")
    if fg_pct < 15.0:
        print(f"  -> DIAGNOSIS: Genuine lack of usable fingerprint area ({fg_pct:.1f}% foreground). Image is mostly empty margin/crop.")
    elif count >= 20:
        print(f"  -> DIAGNOSIS: Excellent biometric quality with {count} reliable minutiae.")

    def to_u8(arr):
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return ((arr - mn) / (mx - mn) * 255.0).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    # Save visualization composite
    disp_orig = cv2.resize(img, (384, 384))
    disp_orig_rgb = cv2.cvtColor(disp_orig, cv2.COLOR_GRAY2BGR)
    cv2.putText(disp_orig_rgb, f"Original ({orig_w}x{orig_h})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    enh_u8 = to_u8(enhanced_img)
    enh_rgb = cv2.cvtColor(enh_u8, cv2.COLOR_GRAY2BGR)
    cv2.putText(enh_rgb, f"Enhanced (FG: {fg_pct:.1f}%)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(overlay, f"Minutiae: {count} (Red=End, Grn=Bif)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    composite = np.hstack([disp_orig_rgb, enh_rgb, overlay])
    out_file = os.path.join(output_dir, f"{tag}_diagnostic.png")
    cv2.imwrite(out_file, composite)
    print(f"  - Saved diagnostic composite: {out_file}")

    if do_ddr:
        rect = rectify_with_ddrnet(img)
        enhanced_r, overlay_r, count_r, fg_pct_r = analyze_and_render_overlay(rect, f"{tag}_rectified")
        disp_rect = cv2.resize(rect, (384, 384))
        disp_rect_rgb = cv2.cvtColor(disp_rect, cv2.COLOR_GRAY2BGR)
        cv2.putText(disp_rect_rgb, f"DDRNet Rectified", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(overlay_r, f"Rect Minutiae: {count_r}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        comp_r = np.hstack([disp_rect_rgb, cv2.cvtColor(to_u8(enhanced_r), cv2.COLOR_GRAY2BGR), overlay_r])
        out_file_r = os.path.join(output_dir, f"{tag}_rectified_diagnostic.png")
        cv2.imwrite(out_file_r, comp_r)
        print(f"  - DDRNet Rectified Minutiae: {count_r}, FG: {fg_pct_r:.1f}%")
        print(f"  - Saved DDRNet diagnostic composite: {out_file_r}")

print("=" * 80)

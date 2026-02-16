#!/usr/bin/env python3
"""Grid search for COP-HAC clustering parameters in the visual ReID pipeline.

Loads a cached interview session, reads real video frames from the disk frame
cache, pre-computes HSV color histograms for all accepted crops, then sweeps
over distance_threshold, appearance_weight, spatial_weight, color_weight, and
tau to find parameter combos that produce the expected number of identity
clusters.

Usage (inside Docker container):
    python3 grid_search_clustering.py
    python3 grid_search_clustering.py --cache-key p225664_t245750647 --target-k 4
    python3 grid_search_clustering.py --cache-key p225664_t245750647 --target-k 4 --top 30
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `interview.*` imports work.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from complete_reid import _compute_hist, _rgb_to_hsv
from interview.cache_manager import load_session
from interview.reid_pipeline import (
    collapse_runs,
    compute_cooccurrence,
    compute_spatial_temporal,
    enrich_crops_t1,
    extract_cannot_links,
    over_cluster,
    propagate_cannot_links_to_runs,
    segment_runs,
    validate_and_split_runs,
)
from interview.state import CropData, CropLabel, CueWeights, EnrichedCrop, RunGroup


# ---------------------------------------------------------------------------
# Real frame reader -- reads from disk frame cache.
# ---------------------------------------------------------------------------

def _make_disk_cache_read_frame(cache_root, cache_key, stride=3):
    """Return a read_frame callable that reads from the disk frame cache.

    Frames are stored as /cache_root/cache_key/frames/%08d.jpg at every
    `stride` frames.  For a requested frame_idx, try the nearest cache
    frame first, then the exact frame.
    """
    frames_dir = os.path.join(cache_root, cache_key, "frames")

    def read_frame(video_path, frame_idx, cache_key=None):
        nearest = round(frame_idx / stride) * stride
        fpath = os.path.join(frames_dir, "%08d.jpg" % nearest)
        if not os.path.exists(fpath):
            fpath = os.path.join(frames_dir, "%08d.jpg" % frame_idx)
        if not os.path.exists(fpath):
            return None
        return Image.open(fpath)

    return read_frame


def _precompute_histograms(crops, read_frame, video_path, cache_key,
                           frame_w, frame_h, bins=(8, 8, 8)):
    """Pre-compute HSV histograms on CropData objects using real frames.

    Mutates crop.histogram in place so that enrich_crops_t1() will find
    them at line 211 (color_hist=crop.histogram if modality == "rgb").
    """
    # Group by frame to minimize disk reads
    frame_to_crops = {}
    for crop in crops:
        if crop.features is None:
            continue
        frame_to_crops.setdefault(crop.frame_idx, []).append(crop)

    computed = 0
    skipped = 0
    for frame_idx in sorted(frame_to_crops.keys()):
        pil_frame = read_frame(video_path, frame_idx, cache_key=cache_key)
        if pil_frame is None:
            skipped += len(frame_to_crops[frame_idx])
            continue

        frame_rgb = np.array(pil_frame.convert("RGB"))
        h, w = frame_rgb.shape[:2]

        for crop in frame_to_crops[frame_idx]:
            if crop.histogram is not None:
                continue  # already has one
            x1, y1, x2, y2 = crop.xyxy.astype(int)
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            crop_rgb = frame_rgb[y1:y2, x1:x2]
            if crop_rgb.size == 0:
                continue
            crop.histogram = _compute_hist(crop_rgb, bins)
            computed += 1

    return computed, skipped


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cluster_entropy(sizes):
    """Shannon entropy of cluster size distribution (higher = more balanced)."""
    total = sum(sizes)
    if total == 0:
        return 0.0
    probs = [s / total for s in sizes]
    ent = 0.0
    for p in probs:
        if p > 0:
            ent -= p * np.log2(p)
    return ent


def balance_score(sizes):
    """Balance metric: inverse of max/min ratio times entropy.

    Higher is better (more balanced clusters with higher entropy).
    Returns 0.0 for degenerate single-element clusters.
    """
    if not sizes or min(sizes) == 0:
        return 0.0
    ratio = max(sizes) / min(sizes)
    ent = cluster_entropy(sizes)
    return ent / ratio


# ---------------------------------------------------------------------------
# Core grid search logic (operates on pre-enriched crops)
# ---------------------------------------------------------------------------

def evaluate_combo(
    enriched,
    crop_map,
    cooccurrence,
    distance_threshold,
    appearance_weight,
    spatial_weight,
    color_weight,
    tau,
    run_threshold=0.5,
):
    """Run the clustering pipeline for one parameter combo.

    Returns a dict with n_clusters, cluster_sizes, entropy, balance, and params.
    """
    # Build CueWeights: appearance, spatial, and color are specified;
    # remaining weight is split evenly among body, context, temporal.
    remaining = max(1.0 - appearance_weight - spatial_weight - color_weight, 0.0)
    each = remaining / 3.0
    weights = CueWeights(
        appearance=appearance_weight,
        spatial=spatial_weight,
        body=each,
        color=color_weight,
        context=each,
        temporal=each,
    )
    weights.normalize()

    # Recompute spatial-temporal with this tau
    st = compute_spatial_temporal(enriched, tau=tau)

    # Segment runs
    raw_runs = segment_runs(enriched, st, threshold=run_threshold)
    crop_cannot = extract_cannot_links(cooccurrence, enriched)

    # Validate/split runs, collapse into RunGroups
    validated = validate_and_split_runs(raw_runs, crop_map)
    run_groups = collapse_runs(validated, crop_map)

    # Propagate cannot-links to run level
    run_cannot = propagate_cannot_links_to_runs(run_groups, crop_cannot)

    # Over-cluster (COP-HAC)
    clusters = over_cluster(run_groups, run_cannot, weights, distance_threshold)

    # Compute cluster sizes
    # clusters maps cluster_id -> [run_ids]; expand to crop counts
    run_map = {rg.run_id: rg for rg in run_groups}
    sizes = []
    for cid, run_ids in sorted(clusters.items()):
        n_crops = sum(len(run_map[rid].crop_ids) for rid in run_ids)
        sizes.append(n_crops)
    sizes.sort(reverse=True)

    n_clusters = len(clusters)
    ent = cluster_entropy(sizes)
    bal = balance_score(sizes)

    return {
        "n_clusters": n_clusters,
        "cluster_sizes": sizes,
        "entropy": ent,
        "balance": bal,
        "distance_threshold": distance_threshold,
        "appearance_weight": appearance_weight,
        "spatial_weight": spatial_weight,
        "color_weight": color_weight,
        "tau": tau,
        "weights": weights.to_dict(),
        "n_runs": len(run_groups),
        "n_run_cannot_links": len(run_cannot),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Grid search COP-HAC clustering parameters for visual ReID."
    )
    parser.add_argument(
        "--cache-key", default="p225664_t245750647",
        help="Session cache key to load (default: p225664_t245750647)",
    )
    parser.add_argument(
        "--target-k", type=int, default=4,
        help="Expected number of identity clusters (default: 4)",
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of top combos to print (default: 30)",
    )
    parser.add_argument(
        "--cache-root", default=None,
        help="Override INTERVIEW_CACHE_ROOT (default: from env or /data/adapters)",
    )
    parser.add_argument(
        "--frame-cache-stride", type=int, default=3,
        help="Stride of cached frames in disk frame cache (default: 3)",
    )
    args = parser.parse_args()

    cache_root = args.cache_root or os.getenv("INTERVIEW_CACHE_ROOT", "/data/adapters")
    if args.cache_root:
        os.environ["INTERVIEW_CACHE_ROOT"] = args.cache_root
        import interview.cache_manager as cm
        cm.CACHE_ROOT = args.cache_root

    # ------------------------------------------------------------------
    # 1. Load session
    # ------------------------------------------------------------------
    print("Loading session from cache_key=%r ..." % args.cache_key)
    session = load_session(args.cache_key)
    if session is None:
        print("ERROR: No cached session found for key %r" % args.cache_key)
        print("  Cache root: %s" % cache_root)
        sys.exit(1)

    accepted = [
        c for c in session.crops.values()
        if c.label == CropLabel.ACCEPTED and c.features is not None
    ]
    print("  Session loaded: %d total crops, %d accepted with features"
          % (len(session.crops), len(accepted)))
    print("  Video: %dx%d, %d frames"
          % (session.width, session.height, session.frames_count))

    if not accepted:
        print("ERROR: No accepted crops with features found. Nothing to cluster.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Pre-compute histograms from REAL frames (disk cache)
    # ------------------------------------------------------------------
    frames_dir = os.path.join(cache_root, args.cache_key, "frames")
    if os.path.isdir(frames_dir):
        n_cached = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        print("  Disk frame cache: %s (%d frames)" % (frames_dir, n_cached))
    else:
        print("  WARNING: No disk frame cache at %s" % frames_dir)
        n_cached = 0

    read_frame = _make_disk_cache_read_frame(
        cache_root, args.cache_key, stride=args.frame_cache_stride,
    )

    print("Pre-computing HSV histograms from real frames ...")
    t0_hist = time.time()
    n_computed, n_skipped = _precompute_histograms(
        accepted, read_frame, session.video_path, session.cache_key,
        session.width, session.height,
    )
    t_hist = time.time() - t0_hist
    n_with_hist = sum(1 for c in accepted if c.histogram is not None)
    print("  Computed %d histograms in %.1fs (%d skipped, %d/%d have histograms)"
          % (n_computed, t_hist, n_skipped, n_with_hist, len(accepted)))

    # ------------------------------------------------------------------
    # 3. Enrich crops ONCE (uses real frames now)
    # ------------------------------------------------------------------
    print("Enriching crops (Tier 1) ...")
    t0_enrich = time.time()
    enriched = enrich_crops_t1(
        accepted, session.width, session.height,
        read_frame, session.video_path, session.cache_key,
    )
    t_enrich = time.time() - t0_enrich
    print("  Enriched %d crops in %.1fs" % (len(enriched), t_enrich))

    # Verify histograms propagated
    n_with_color = sum(1 for c in enriched if c.color_hist is not None)
    print("  Enriched crops with color_hist: %d/%d" % (n_with_color, len(enriched)))

    if not enriched:
        print("ERROR: Enrichment produced zero crops.")
        sys.exit(1)

    # Precompute co-occurrence (does not depend on tau)
    crop_map = {c.crop_id: c for c in enriched}
    cooccurrence = compute_cooccurrence(enriched)

    n_cooc = int(cooccurrence.sum()) // 2
    print("  Co-occurring crop pairs (same frame): %d" % n_cooc)

    # Print DINOv3 distance statistics
    n = len(enriched)
    if n > 1:
        feats = np.stack([c.dinov3_cls for c in enriched])
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        feats_normed = feats / norms
        cosine_sim = feats_normed @ feats_normed.T
        triu_idx = np.triu_indices(n, k=1)
        sims = cosine_sim[triu_idx]
        print("")
        print("  DINOv3 cosine similarity stats (%d pairs):" % len(sims))
        print("    min=%.3f  max=%.3f  mean=%.3f  median=%.3f  std=%.3f"
              % (sims.min(), sims.max(), sims.mean(), np.median(sims), sims.std()))
        dists = 1.0 - sims
        print("  DINOv3 cosine distance stats:")
        print("    min=%.3f  max=%.3f  mean=%.3f  median=%.3f  std=%.3f"
              % (dists.min(), dists.max(), dists.mean(), np.median(dists), dists.std()))

    # Print color histogram distance statistics (if available)
    if n_with_color > 1:
        color_crops = [c for c in enriched if c.color_hist is not None]
        hists = np.stack([c.color_hist for c in color_crops])
        # Compute pairwise histogram intersection
        n_c = len(color_crops)
        color_dists = []
        for i in range(n_c):
            for j in range(i + 1, n_c):
                intersection = np.minimum(hists[i], hists[j]).sum()
                color_dists.append(1.0 - intersection)
        color_dists = np.array(color_dists)
        print("")
        print("  Color histogram distance stats (%d pairs):" % len(color_dists))
        print("    min=%.3f  max=%.3f  mean=%.3f  median=%.3f  std=%.3f"
              % (color_dists.min(), color_dists.max(), color_dists.mean(),
                 np.median(color_dists), color_dists.std()))

    # ------------------------------------------------------------------
    # 4. Define grid (now includes color_weight as explicit dimension)
    # ------------------------------------------------------------------
    grid_distance_threshold = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    grid_appearance_weight = [0.20, 0.30, 0.40, 0.50, 0.60]
    grid_spatial_weight = [0.05, 0.10, 0.15, 0.20]
    grid_color_weight = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
    grid_tau = [25]  # tau had zero effect in previous run; fix at 25

    # Filter combos where appearance + spatial + color > 0.95 (leave room for others)
    combos = []
    for dt, aw, sw, cw, tau in itertools.product(
        grid_distance_threshold,
        grid_appearance_weight,
        grid_spatial_weight,
        grid_color_weight,
        grid_tau,
    ):
        if aw + sw + cw > 0.95:
            continue
        combos.append((dt, aw, sw, cw, tau))

    total = len(combos)
    print("")
    print("Grid search: %d parameter combinations" % total)
    print("  distance_threshold: %s" % grid_distance_threshold)
    print("  appearance_weight:  %s" % grid_appearance_weight)
    print("  spatial_weight:     %s" % grid_spatial_weight)
    print("  color_weight:       %s" % grid_color_weight)
    print("  tau:                %s (fixed)" % grid_tau)

    # ------------------------------------------------------------------
    # 5. Run grid search
    # ------------------------------------------------------------------
    results = []
    t0_grid = time.time()

    for i, (dt, aw, sw, cw, tau) in enumerate(combos):
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0_grid
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print("  [%d/%d] %.1fs elapsed, %.1f combos/s, ETA %.0fs"
                  % (i + 1, total, elapsed, rate, eta))

        result = evaluate_combo(
            enriched=enriched,
            crop_map=crop_map,
            cooccurrence=cooccurrence,
            distance_threshold=dt,
            appearance_weight=aw,
            spatial_weight=sw,
            color_weight=cw,
            tau=tau,
        )
        results.append(result)

    t_grid = time.time() - t0_grid
    print("")
    print("Grid search complete: %d combos in %.1fs (%.1f combos/s)"
          % (total, t_grid, total / t_grid))

    # ------------------------------------------------------------------
    # 6. Summary: cluster count distribution per distance_threshold
    # ------------------------------------------------------------------
    print("")
    print("=" * 80)
    print("SUMMARY: Cluster count distribution by distance_threshold")
    print("=" * 80)

    bucket_labels = ["1", "2", "3", "4", "5", "6-10", "11-20", "21+"]

    def bucket_key(n_clusters):
        if n_clusters <= 5:
            return str(n_clusters)
        elif n_clusters <= 10:
            return "6-10"
        elif n_clusters <= 20:
            return "11-20"
        else:
            return "21+"

    dt_groups = {}
    for r in results:
        dt = r["distance_threshold"]
        dt_groups.setdefault(dt, []).append(r)

    header = "%6s" % "dt"
    for bl in bucket_labels:
        header += "  %5s" % bl
    header += "  %5s" % "total"
    print(header)
    print("-" * len(header))

    for dt in sorted(dt_groups.keys()):
        group = dt_groups[dt]
        counter = Counter(bucket_key(r["n_clusters"]) for r in group)
        row = "%6.2f" % dt
        for bl in bucket_labels:
            row += "  %5d" % counter.get(bl, 0)
        row += "  %5d" % len(group)
        print(row)

    # ------------------------------------------------------------------
    # 7. Filter to target k, rank by balance
    # ------------------------------------------------------------------
    target_k = args.target_k
    target_results = [r for r in results if r["n_clusters"] == target_k]

    print("")
    print("=" * 80)
    print("COMBOS PRODUCING EXACTLY %d CLUSTERS: %d" % (target_k, len(target_results)))
    print("=" * 80)

    if not target_results:
        print("No combos produced exactly %d clusters." % target_k)
        all_k = Counter(r["n_clusters"] for r in results)
        print("Available cluster counts: %s" % dict(sorted(all_k.items())))

        nearest = min(all_k.keys(), key=lambda k: abs(k - target_k))
        print("")
        print("Showing top %d combos for nearest k=%d:" % (args.top, nearest))
        target_results = [r for r in results if r["n_clusters"] == nearest]
        target_k = nearest

    # Sort by balance (descending)
    target_results.sort(key=lambda r: -r["balance"])
    top_n = target_results[:args.top]

    print("")
    print("Top %d combos by balance score (k=%d):" % (len(top_n), target_k))
    print("")
    print("%3s  %5s  %5s  %5s  %5s  %4s  %6s  %7s  %s"
          % ("#", "dt", "app", "spat", "color", "tau", "bal", "entropy", "sizes"))
    print("-" * 90)

    for rank, r in enumerate(top_n, 1):
        sizes_str = ",".join(str(s) for s in r["cluster_sizes"])
        print("%3d  %5.2f  %5.2f  %5.2f  %5.2f  %4.0f  %6.3f  %7.3f  [%s]"
              % (rank, r["distance_threshold"],
                 r["appearance_weight"], r["spatial_weight"],
                 r["color_weight"],
                 r["tau"], r["balance"],
                 r["entropy"], sizes_str))

    # ------------------------------------------------------------------
    # 8. Compare color=0 vs color>0 for same dt/app/spatial
    # ------------------------------------------------------------------
    print("")
    print("=" * 80)
    print("EFFECT OF COLOR WEIGHT (comparing color=0 vs color>0 for same dt/app/spat)")
    print("=" * 80)

    # Group by (dt, app, spatial), compare cluster counts with and without color
    key_groups = {}
    for r in results:
        key = (r["distance_threshold"], r["appearance_weight"], r["spatial_weight"])
        key_groups.setdefault(key, []).append(r)

    color_diff_count = 0
    for key, group in sorted(key_groups.items()):
        no_color = [r for r in group if r["color_weight"] == 0.0]
        with_color = [r for r in group if r["color_weight"] > 0.0]
        if not no_color or not with_color:
            continue
        nc_k = no_color[0]["n_clusters"]
        for r in with_color:
            if r["n_clusters"] != nc_k:
                color_diff_count += 1
                print("  dt=%.2f app=%.2f spat=%.2f: color=0 -> k=%d, color=%.2f -> k=%d  sizes=%s"
                      % (key[0], key[1], key[2], nc_k,
                         r["color_weight"], r["n_clusters"], r["cluster_sizes"]))
                if color_diff_count >= 30:
                    break
        if color_diff_count >= 30:
            break

    if color_diff_count == 0:
        print("  No differences found (color cue had no effect on cluster count)")
    else:
        print("  ... %d parameter combos where color changed the cluster count" % color_diff_count)

    # ------------------------------------------------------------------
    # 9. Best combo details
    # ------------------------------------------------------------------
    if top_n:
        best = top_n[0]
        print("")
        print("Best combo details:")
        print("  distance_threshold = %s" % best["distance_threshold"])
        print("  appearance_weight  = %s" % best["appearance_weight"])
        print("  spatial_weight     = %s" % best["spatial_weight"])
        print("  color_weight       = %s" % best["color_weight"])
        print("  tau                = %s" % best["tau"])
        print("  n_runs             = %s" % best["n_runs"])
        print("  n_run_cannot_links = %s" % best["n_run_cannot_links"])
        print("  Final CueWeights:")
        for k, v in best["weights"].items():
            print("    %12s = %.4f" % (k, v))
        print("  cluster_sizes      = %s" % best["cluster_sizes"])
        print("  entropy            = %.4f" % best["entropy"])
        print("  balance            = %.4f" % best["balance"])

    total_time = time.time() - t0_enrich
    print("")
    print("Total time: %.1fs" % total_time)


if __name__ == "__main__":
    main()

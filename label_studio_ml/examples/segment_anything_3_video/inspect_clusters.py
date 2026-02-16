#!/usr/bin/env python3
"""Generate an HTML page showing cluster contents for visual inspection.

Runs the clustering pipeline with specified parameters and outputs an HTML
file with crop thumbnails grouped by cluster. Served via a simple HTTP server.

Usage (inside Docker container):
    python3 inspect_clusters.py
    python3 inspect_clusters.py --dt 0.35 --app-weight 0.20 --spatial-weight 0.20
    python3 inspect_clusters.py --port 9099
"""

from __future__ import annotations

import argparse
import base64
import html
import http.server
import io
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

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
from interview.state import CropLabel, CueWeights, EnrichedCrop


def _make_fake_read_frame(width, height):
    gray_frame = Image.new("RGB", (width, height), (128, 128, 128))
    def read_frame(video_path, frame_idx, cache_key=None):
        return gray_frame
    return read_frame


def _read_crop_thumbnail(crop_data, session, frame_cache) -> str:
    """Read video frame, crop the bounding box, return base64 PNG."""
    frame_idx = crop_data.frame_idx
    if frame_idx not in frame_cache:
        try:
            import av
            container = av.open(session.video_path)
            stream = container.streams.video[0]
            # Seek to nearest keyframe before target
            target_pts = int(frame_idx * stream.time_base.denominator
                           / (stream.time_base.numerator * stream.average_rate))
            container.seek(target_pts, stream=stream)
            for packet in container.demux(stream):
                for frame in packet.decode():
                    if frame.pts is not None:
                        actual_idx = int(frame.pts * stream.time_base * stream.average_rate)
                        if actual_idx >= frame_idx:
                            frame_cache[frame_idx] = frame.to_image()
                            container.close()
                            break
                if frame_idx in frame_cache:
                    break
            else:
                container.close()
        except Exception:
            pass

    if frame_idx not in frame_cache:
        return ""

    pil_frame = frame_cache[frame_idx]
    x1, y1, x2, y2 = [int(v) for v in crop_data.xyxy]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(pil_frame.width, x2)
    y2 = min(pil_frame.height, y2)
    if x2 <= x1 or y2 <= y1:
        return ""

    crop_img = pil_frame.crop((x1, y1, x2, y2))
    # Resize to thumbnail
    crop_img.thumbnail((120, 120), Image.LANCZOS)
    buf = io.BytesIO()
    crop_img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _crop_to_thumb_html(crop_data, session, idx: int, frame_cache: dict) -> str:
    """Generate HTML for one crop thumbnail."""
    b64 = _read_crop_thumbnail(crop_data, session, frame_cache)
    frame = crop_data.frame_idx
    crop_id = crop_data.crop_id
    bbox = crop_data.xyxy
    w = int(bbox[2] - bbox[0])
    h = int(bbox[3] - bbox[1])

    if b64:
        img_tag = f'<img src="data:image/jpeg;base64,{b64}" style="max-width:120px;max-height:120px;object-fit:contain;border:1px solid #555;border-radius:4px;">'
    else:
        img_tag = '<div style="width:100px;height:100px;background:#333;border:1px solid #555;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#888;font-size:10px;">no thumb</div>'

    return f'''<div style="display:inline-block;text-align:center;margin:4px;vertical-align:top;">
        {img_tag}
        <div style="font-size:10px;color:#aaa;max-width:120px;overflow:hidden;">f{frame} ({w}x{h})</div>
        <div style="font-size:9px;color:#666;">{html.escape(crop_id[:12])}</div>
    </div>'''


def generate_html(session, clusters_by_crop, cluster_sizes, params) -> str:
    """Generate the full HTML page."""
    # Sort clusters by size descending
    sorted_clusters = sorted(clusters_by_crop.items(), key=lambda x: -len(x[1]))

    # Shared frame cache to avoid re-reading the same frame for multiple crops
    frame_cache = {}
    print("  Reading video frames for thumbnails ...")

    crops_html = []
    for cluster_id, crop_ids in sorted_clusters:
        thumbs = []
        for i, cid in enumerate(sorted(crop_ids, key=lambda c: session.crops[c].frame_idx if c in session.crops else 0)):
            if cid in session.crops:
                thumbs.append(_crop_to_thumb_html(session.crops[cid], session, i, frame_cache))
            else:
                thumbs.append(f'<div style="display:inline-block;margin:4px;color:#888;">?{cid[:8]}</div>')

        crops_html.append(f'''
        <div style="background:#1a1a2e;border:1px solid #444;border-radius:8px;padding:12px;margin:12px 0;">
            <h3 style="color:#e0e0e0;margin:0 0 8px 0;">
                Cluster {cluster_id}
                <span style="color:#888;font-size:14px;font-weight:normal;">({len(crop_ids)} crops)</span>
            </h3>
            <div style="display:flex;flex-wrap:wrap;gap:2px;">
                {"".join(thumbs)}
            </div>
        </div>''')

    params_html = " | ".join(f"{k}={v}" for k, v in params.items())

    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Cluster Inspection</title>
    <style>
        body {{ background: #0f0f23; color: #e0e0e0; font-family: -apple-system, sans-serif; padding: 20px; }}
        h1 {{ color: #fff; }}
        .params {{ background: #16213e; padding: 10px 16px; border-radius: 6px; margin: 12px 0; font-family: monospace; font-size: 13px; color: #aaa; }}
        .stats {{ display: flex; gap: 20px; margin: 12px 0; }}
        .stat {{ background: #16213e; padding: 8px 16px; border-radius: 6px; text-align: center; }}
        .stat-val {{ font-size: 24px; font-weight: bold; color: #4fc3f7; }}
        .stat-label {{ font-size: 11px; color: #888; }}
    </style>
</head>
<body>
    <h1>Cluster Inspection</h1>
    <div class="params">{html.escape(params_html)}</div>
    <div class="stats">
        <div class="stat"><div class="stat-val">{len(sorted_clusters)}</div><div class="stat-label">Clusters</div></div>
        <div class="stat"><div class="stat-val">{sum(len(v) for v in clusters_by_crop.values())}</div><div class="stat-label">Total Crops</div></div>
        <div class="stat"><div class="stat-val">{" / ".join(str(len(c)) for _, c in sorted_clusters)}</div><div class="stat-label">Sizes</div></div>
    </div>
    {"".join(crops_html)}
    <div style="color:#555;font-size:11px;margin-top:20px;">Generated {time.strftime("%Y-%m-%d %H:%M:%S")}</div>
</body>
</html>'''


def _make_disk_cache_read_frame(cache_root, cache_key, stride=3):
    """Read frames from disk cache instead of video decoding."""
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
    """Pre-compute HSV histograms from real frames."""
    from complete_reid import _compute_hist

    frame_to_crops = {}
    for crop in crops:
        if crop.features is None:
            continue
        frame_to_crops.setdefault(crop.frame_idx, []).append(crop)

    computed = 0
    for frame_idx in sorted(frame_to_crops.keys()):
        pil_frame = read_frame(video_path, frame_idx, cache_key=cache_key)
        if pil_frame is None:
            continue
        frame_rgb = np.array(pil_frame.convert("RGB"))
        h, w = frame_rgb.shape[:2]
        for crop in frame_to_crops[frame_idx]:
            if crop.histogram is not None:
                continue
            x1, y1, x2, y2 = [int(v) for v in crop.xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop_rgb = frame_rgb[y1:y2, x1:x2]
            if crop_rgb.size == 0:
                continue
            crop.histogram = _compute_hist(crop_rgb, bins)
            computed += 1
    return computed


def main():
    parser = argparse.ArgumentParser(description="Inspect cluster contents visually.")
    parser.add_argument("--cache-key", default="p225664_t245750647")
    parser.add_argument("--dt", type=float, default=0.35, help="distance_threshold")
    parser.add_argument("--app-weight", type=float, default=0.30, help="appearance weight")
    parser.add_argument("--spatial-weight", type=float, default=0.05, help="spatial weight")
    parser.add_argument("--color-weight", type=float, default=0.15, help="color weight")
    parser.add_argument("--tau", type=float, default=25)
    parser.add_argument("--port", type=int, default=9099, help="HTTP server port")
    parser.add_argument("--no-serve", action="store_true", help="Write HTML file only, don't start server")
    parser.add_argument("--cache-root", default=None)
    args = parser.parse_args()

    cache_root = args.cache_root or os.getenv("INTERVIEW_CACHE_ROOT", "/data/adapters")
    if args.cache_root:
        os.environ["INTERVIEW_CACHE_ROOT"] = args.cache_root
        import interview.cache_manager as cm
        cm.CACHE_ROOT = args.cache_root

    # Load session
    print("Loading session ...")
    session = load_session(args.cache_key)
    if session is None:
        print("ERROR: No session for key %r" % args.cache_key)
        sys.exit(1)

    accepted = [c for c in session.crops.values()
                if c.label == CropLabel.ACCEPTED and c.features is not None]
    print("  %d accepted crops" % len(accepted))

    # Pre-compute histograms from disk cache
    read_frame = _make_disk_cache_read_frame(cache_root, args.cache_key)
    n_hist = _precompute_histograms(
        accepted, read_frame, session.video_path, session.cache_key,
        session.width, session.height,
    )
    print("  Computed %d histograms" % n_hist)

    # Enrich with real frames
    print("Enriching ...")
    enriched = enrich_crops_t1(
        accepted, session.width, session.height,
        read_frame, session.video_path, session.cache_key,
    )

    # Cluster with specified params
    remaining = max(1.0 - args.app_weight - args.spatial_weight - args.color_weight, 0.0)
    each = remaining / 3.0  # split among body, context, temporal
    weights = CueWeights(
        appearance=args.app_weight, spatial=args.spatial_weight,
        body=each, color=args.color_weight, context=each, temporal=each,
    )
    weights.normalize()

    crop_map = {c.crop_id: c for c in enriched}
    cooccurrence = compute_cooccurrence(enriched)
    st = compute_spatial_temporal(enriched, tau=args.tau)
    raw_runs = segment_runs(enriched, st, threshold=0.5)
    crop_cannot = extract_cannot_links(cooccurrence, enriched)
    validated = validate_and_split_runs(raw_runs, crop_map)
    run_groups = collapse_runs(validated, crop_map)
    run_cannot = propagate_cannot_links_to_runs(run_groups, crop_cannot)
    clusters = over_cluster(run_groups, run_cannot, weights, args.dt)

    # Map cluster_id -> list of crop_ids
    run_map = {rg.run_id: rg for rg in run_groups}
    clusters_by_crop = {}
    for cid, run_ids in clusters.items():
        crop_ids = []
        for rid in run_ids:
            crop_ids.extend(run_map[rid].crop_ids)
        clusters_by_crop[cid] = crop_ids

    sizes = sorted([len(v) for v in clusters_by_crop.values()], reverse=True)
    print("  %d clusters: %s" % (len(clusters), sizes))

    params = {
        "cache_key": args.cache_key,
        "distance_threshold": args.dt,
        "appearance_weight": args.app_weight,
        "spatial_weight": args.spatial_weight,
        "color_weight": args.color_weight,
        "tau": args.tau,
    }

    html_content = generate_html(session, clusters_by_crop, sizes, params)

    # Write to static dir for Flask serving
    out_path = "/app/interview/static/cluster_inspection.html"
    try:
        with open(out_path, "w") as f:
            f.write(html_content)
        print("  HTML written to %s" % out_path)
    except OSError:
        out_path = "/tmp/cluster_inspection.html"
        with open(out_path, "w") as f:
            f.write(html_content)
        print("  HTML written to %s" % out_path)

    if args.no_serve:
        return

    # Serve
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))
        def log_message(self, format, *a):
            pass  # quiet

    server = http.server.HTTPServer(("0.0.0.0", args.port), Handler)
    print("\n  Serving at http://localhost:%d" % args.port)
    print("  Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

"""Optional Targeted Local CV disambiguator for unresolved UNDETERMINED intervals."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, asset_paths, half_video_blocks, video_path  # noqa: E402
from starter.load_homography import load as load_homography, project  # noqa: E402
from src import NOBODY, UNDETERMINED  # noqa: E402


def project_pitch_roi_to_pixels(H_matrix, x_norm, y_norm, patch_radius_px=32, frame_wh=(1920, 1080)):
    """Project unscaled [0, 1] pitch coordinates via homography into clamped pixel ROI bounds.

    Coordinates passed to starter.load_homography.project MUST be unscaled [0, 1].
    Returns None if the homography matrix or projected coordinates are invalid/non-finite.
    """
    if H_matrix is None or np.isnan(H_matrix).any():
        return None
    if np.isnan(x_norm) or np.isnan(y_norm):
        return None

    px, py = project(H_matrix, float(x_norm), float(y_norm))
    if not (np.isfinite(px) and np.isfinite(py)):
        return None

    cx, cy = int(round(float(px))), int(round(float(py)))
    w, h = frame_wh
    if cx < -patch_radius_px or cx >= w + patch_radius_px or cy < -patch_radius_px or cy >= h + patch_radius_px:
        return None

    x0 = max(0, cx - patch_radius_px)
    y0 = max(0, cy - patch_radius_px)
    x1 = min(w, cx + patch_radius_px)
    y1 = min(h, cy + patch_radius_px)
    if x1 <= x0 or y1 <= y0:
        return None
    return cx, cy, x0, y0, x1, y1


def _select_ambiguous_intervals(features, labels_pass1, game=GAME, max_intervals=40):
    """Select high-priority UNDETERMINED intervals (e.g., Team=N/A restarts and NaN gaps)."""
    n_frames = features["n_frames"]
    is_set_piece = features["is_set_piece"]
    set_piece_team = features["set_piece_team"]
    ball_vis = features["ball_visible"]

    candidates = []
    for _, first_vf, last_vf in half_video_blocks(game):
        h_s = max(0, int(first_vf))
        h_e = min(n_frames - 1, int(last_vf))
        t = h_s
        while t <= h_e:
            if labels_pass1[t] != UNDETERMINED:
                t += 1
                continue
            j = t
            while j + 1 <= h_e and labels_pass1[j + 1] == UNDETERMINED:
                j += 1

            length = j - t + 1
            left_lbl = int(labels_pass1[t - 1]) if t > h_s else UNDETERMINED
            right_lbl = int(labels_pass1[j + 1]) if j < h_e else UNDETERMINED
            has_na_sp = bool(np.any(is_set_piece[t : j + 1] & (set_piece_team[t : j + 1] == NOBODY)))
            is_turnover = left_lbl in (0, 1) and right_lbl in (0, 1) and left_lbl != right_lbl
            nan_frac = float(np.mean(~ball_vis[t : j + 1]))

            priority = (3.0 if has_na_sp else 0.0) + (2.0 if is_turnover else 0.0) + nan_frac
            if length >= 5 and (has_na_sp or is_turnover or nan_frac > 0.5):
                candidates.append((priority, t, j))
            t = j + 1

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [(t, j) for _, t, j in candidates[:max_intervals]]


def _extract_patch_cues(frame_bgr, roi_bounds):
    """Extract lightweight deterministic ball-contrast & jersey-hue cues from a small ROI crop."""
    _, _, x0, y0, x1, y1 = roi_bounds
    patch = frame_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None

    # White/bright circular ball contrast against green pitch turf.
    b = patch[:, :, 0].astype(np.float32)
    g = patch[:, :, 1].astype(np.float32)
    r = patch[:, :, 2].astype(np.float32)
    brightness = (r + g + b) / 3.0
    green_excess = g - 0.5 * (r + b)
    white_score = brightness - np.maximum(np.abs(r - g), np.abs(g - b))
    ball_detected = bool(np.mean((white_score > 165.0) & (green_excess < 25.0)) > 0.015)

    # Coarse jersey hue cues (Home blaugrana/dark blue-red vs Away high-contrast/light).
    non_grass = green_excess < 15.0
    if np.sum(non_grass) >= 12:
        r_m = float(np.mean(r[non_grass]))
        b_m = float(np.mean(b[non_grass]))
        luma = float(np.mean(brightness[non_grass]))
        # Higher red+blue saturation vs neutral brightness heuristic; soft signal only.
        blaugrana_cue = np.clip((r_m + b_m) / 260.0 - luma / 420.0 + 0.5, 0.2, 0.8)
        team_prob_H = float(blaugrana_cue)
        team_prob_A = float(1.0 - team_prob_H)
    else:
        team_prob_H = 0.5
        team_prob_A = 0.5

    return {
        "team_prob_H": team_prob_H,
        "team_prob_A": team_prob_A,
        "ball_detected": ball_detected,
    }


def extract_targeted_cv_cues(
    features,
    labels_pass1,
    game=GAME,
    data_dir=None,
    enabled=False,
    max_intervals=40,
):
    """Extract optional local CV cues on unresolved UNDETERMINED intervals.

    Gracefully returns {} if BAR-ATH_tactical.mp4, homography CSV, or cv2 is unavailable.
    """
    if not enabled:
        return {}

    vid_path = video_path(game=game, data_dir=data_dir, source="tactical")
    paths = asset_paths(game=game, data_dir=data_dir, source="tactical")
    homog_path = paths.get("homography")
    if not (os.path.isfile(vid_path) and homog_path and os.path.isfile(homog_path)):
        return {}

    try:
        import cv2
    except ImportError:
        return {}

    intervals = _select_ambiguous_intervals(features, labels_pass1, game=game, max_intervals=max_intervals)
    if not intervals:
        return {}

    try:
        H_mats, _, valid = load_homography(game=game, data_dir=data_dir, source="tactical")
    except Exception:
        return {}

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        return {}

    length, width = features["pitch"]
    ball_norm_xy = features["ball_norm_xy"]
    ball_xy_interp = features["ball_xy_interp"]
    cues = {}

    try:
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        for t_s, t_e in intervals:
            # Sample up to 15 evenly spaced frames per ambiguous gap to avoid slow random seeks.
            step = max(1, (t_e - t_s + 1) // 15)
            for t in range(t_s, t_e + 1, step):
                if t >= len(H_mats) or not valid[t]:
                    continue
                x_n, y_n = ball_norm_xy[t, 0], ball_norm_xy[t, 1]
                if np.isnan(x_n) or np.isnan(y_n):
                    bx_m, by_m = ball_xy_interp[t, 0], ball_xy_interp[t, 1]
                    if np.isnan(bx_m) or np.isnan(by_m):
                        continue
                    x_n = float(bx_m) / float(length)
                    y_n = float(by_m) / float(width)

                roi = project_pitch_roi_to_pixels(
                    H_mats[t], x_n, y_n, patch_radius_px=32, frame_wh=(frame_w, frame_h)
                )
                if roi is None:
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    continue
                cue = _extract_patch_cues(frame_bgr, roi)
                if cue is not None:
                    cues[t] = cue
    finally:
        cap.release()

    return cues


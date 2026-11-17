"""Pass 3: Football rule post-processing (deflection/duel merge & Zero Graded X)."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks  # noqa: E402
from src import A, D, H, UNDETERMINED, X  # noqa: E402


def _extract_segments(arr):
    """Return list of [start, end_exclusive, code] for a 1D integer array."""
    if len(arr) == 0:
        return []
    breaks = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(arr)]))
    return [
        [int(bounds[i]), int(bounds[i + 1]), int(arr[bounds[i]])]
        for i in range(len(bounds) - 1)
    ]


def _fill_residual_inside_half(window):
    """Forward-fill then back-fill any UNDETERMINED or X inside a half slice."""
    valid_mask = (window == H) | (window == A) | (window == D)
    if np.all(valid_mask):
        return window
    known = np.flatnonzero(valid_mask)
    if len(known) == 0:
        window[:] = D
        return window
    carry = np.maximum.accumulate(np.where(valid_mask, np.arange(len(window)), -1))
    carry[carry < 0] = known[0]
    window[:] = window[carry]
    return window


def _has_verified_control(features, seg_start, seg_end, team_code):
    """Check whether a short live spell exhibits genuine controlled carry/pass rather than a deflection."""
    sl = slice(seg_start, seg_end)
    ball_vis = features["ball_visible"][sl]
    ball_oob = features["ball_oob"][sl]
    d_own = features["d_H"][sl] if team_code == H else features["d_A"][sl]
    d_opp = features["d_A"][sl] if team_code == H else features["d_H"][sl]

    uncontested = (
        ball_vis
        & (~ball_oob)
        & np.isfinite(d_own)
        & (d_own <= 1.5)
        & (np.isnan(d_opp) | (d_opp > 2.5))
    )
    if int(np.sum(uncontested)) >= 20:
        return True

    # Two distinct uncontested touches/passes or explicit offensive Smart Tagging confirmation.
    off_agree = int(np.sum(features["off_phase_team"][sl] == team_code))
    if int(np.sum(uncontested)) >= 14 and off_agree >= 15:
        return True

    return False


def run_pass3_postprocess(
    features,
    labels_pass2,
    game=GAME,
    min_spell_frames=30,
    min_dead_frames=35,
):
    """Pass 3: Suppress <1.2s uncontrolled deflections/duels and guarantee Zero Graded X."""
    labels = labels_pass2.copy()
    n_frames = features["n_frames"]

    # Mask everything outside half_video_blocks strictly as X.
    in_any_half = np.zeros(n_frames, dtype=bool)

    for _, first_vf, last_vf in half_video_blocks(game):
        h_start = max(0, int(first_vf))
        h_end = min(n_frames - 1, int(last_vf))
        if h_end < h_start:
            continue

        in_any_half[h_start : h_end + 1] = True
        win = labels[h_start : h_end + 1].copy()
        _fill_residual_inside_half(win)

        # 1. Suppress brief spurious dead-ball flickers lacking SET PIECES or OOB evidence.
        changed = True
        while changed:
            changed = False
            segs = _extract_segments(win)
            for idx, (s, e, code) in enumerate(segs):
                if code != D:
                    continue
                seg_len = e - s
                if seg_len >= min_dead_frames:
                    continue
                abs_s, abs_e = h_start + s, h_start + e
                has_sp = bool(np.any(features["is_set_piece"][abs_s:abs_e]))
                has_oob = int(np.sum(features["ball_oob"][abs_s:abs_e])) >= 3
                if has_sp or has_oob:
                    continue

                if 0 < idx < len(segs) - 1:
                    l_code = segs[idx - 1][2]
                    r_code = segs[idx + 1][2]
                    if l_code in (H, A) and l_code == r_code:
                        win[s:e] = l_code
                        changed = True
                        break
                    if l_code in (H, A) and r_code in (H, A) and seg_len < 15:
                        mid = (s + e) // 2
                        win[s:mid] = l_code
                        win[mid:e] = r_code
                        changed = True
                        break

        # 2. Deflection & Duel Filter: suppress isolated < min_spell_frames uncontrolled touches.
        changed = True
        while changed:
            changed = False
            segs = _extract_segments(win)
            for idx, (s, e, code) in enumerate(segs):
                if code not in (H, A):
                    continue
                seg_len = e - s
                if seg_len >= min_spell_frames:
                    continue
                abs_s, abs_e = h_start + s, h_start + e
                if _has_verified_control(features, abs_s, abs_e, code):
                    continue

                l_code = segs[idx - 1][2] if idx > 0 else None
                r_code = segs[idx + 1][2] if idx + 1 < len(segs) else None

                # Sandwiched in-play deflection (H -> brief A -> H) or
                # pre-stoppage deflection out of bounds (H -> brief A -> D).
                if l_code in (H, A) and code == (1 - l_code) and (r_code == l_code or r_code == D):
                    win[s:e] = l_code
                    changed = True
                    break

                # Post-stoppage restart (D -> brief A -> H): preserve valid short restarts,
                # merging only micro-noise (< 15f without matching set_piece_team) or
                # uncontrolled first touches contradicting a named set_piece_team == r_code.
                if l_code == D and r_code in (H, A) and code == (1 - r_code):
                    sp_win = features["set_piece_team"][max(h_start, abs_s - 10) : abs_e]
                    sp_team_is_code = bool(np.any(sp_win == code))
                    sp_team_is_r = bool(np.any(sp_win == r_code))
                    if (seg_len < 15 and not sp_team_is_code) or (sp_team_is_r and not sp_team_is_code):
                        win[s:e] = r_code
                        changed = True
                        break

                # Isolated micro-blip between two dead-ball runs (D -> <15f H/A -> D).
                if l_code == D and r_code == D and seg_len < 15:
                    win[s:e] = D
                    changed = True
                    break

        # Final zero-X guarantee inside this half.
        _fill_residual_inside_half(win)
        if np.any((win != H) & (win != A) & (win != D)):
            raise AssertionError("Graded half [{}, {}] contains invalid/X labels".format(h_start, h_end))
        labels[h_start : h_end + 1] = win

    labels[~in_any_half] = X
    return labels


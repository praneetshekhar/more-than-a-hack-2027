"""Pass 0 (structural halves & algorithmic kickoff gating) and Pass 1 (high-confidence anchors)."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks  # noqa: E402
from src import A, D, H, NOBODY, UNDETERMINED, X  # noqa: E402


def _contiguous_runs(mask):
    """Return list of (start_idx, end_idx_inclusive) for True runs in a 1D boolean mask."""
    if len(mask) == 0:
        return []
    m = mask.astype(np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _detect_kickoff_frame(features, w_start, w_end, game=GAME):
    """Algorithmically locate the kickoff strike frame inside a Kick Off window.

    Uses center-circle ball rest-to-motion kinematics (zero hardcoded frame constants),
    with a collective player acceleration fallback if the ball is occluded.
    """
    if w_end <= w_start:
        return w_start

    length, width = GAMES[game]["pitch"]
    cx, cy = length / 2.0, width / 2.0

    bx = features["ball_xy_interp"][w_start:w_end, 0]
    by = features["ball_xy_interp"][w_start:w_end, 1]
    bspeed = features["ball_speed"][w_start:w_end]
    mean_spd = features["mean_speed_all"][w_start:w_end]

    dist_center = np.hypot(bx - cx, by - cy)
    near_center_rest = (
        np.isfinite(dist_center)
        & (dist_center <= 4.5)
        & (np.isnan(bspeed) | (bspeed < 2.0))
    )

    if np.any(near_center_rest):
        # Use the primary center-circle stationary block before the ball departs.
        rest_runs = _contiguous_runs(near_center_rest)
        best_run = max(rest_runs, key=lambda r: r[1] - r[0])
        rest_bx = float(np.nanmedian(bx[best_run[0] : best_run[1] + 1]))
        rest_by = float(np.nanmedian(by[best_run[0] : best_run[1] + 1]))
        disp = np.hypot(bx - rest_bx, by - rest_by)
        raw_spd = features["ball_speed_raw"][w_start:w_end]

        scan_from = best_run[0]
        # Look for exact departure frame after stationary rest starts.
        for offset in range(scan_from, w_end - w_start):
            disp_val = disp[offset]
            raw_val = raw_spd[offset]
            spd_val = bspeed[offset]
            moved = np.isfinite(disp_val) and disp_val >= 0.30
            fast_raw = np.isfinite(raw_val) and raw_val >= 2.0
            fast_interp = (
                np.isfinite(spd_val)
                and spd_val >= 2.2
                and (np.isnan(raw_val) or raw_val >= 1.0)
                and (np.isnan(disp_val) or disp_val >= 0.15)
            )
            if moved or fast_raw or fast_interp:
                return w_start + offset
        return w_start + min(best_run[1] + 1, (w_end - w_start) - 1)

    # Fallback when center-circle ball tracking is occluded: transition from standby to play.
    valid_spd = np.isfinite(mean_spd)
    if np.any(valid_spd):
        standby = valid_spd & (mean_spd < 1.35)
        active = valid_spd & (mean_spd >= 1.50)
        if np.any(standby) and np.any(active):
            first_standby = int(np.flatnonzero(standby)[0])
            active_after = np.flatnonzero(active & (np.arange(len(active)) > first_standby))
            if len(active_after) > 0:
                return w_start + int(active_after[0])

    return w_start


def run_pass0_structural_and_kickoff(features, game=GAME):
    """Pass 0: mask ungraded frames as X and algorithmically detect pre-kickoff D per half."""
    n_frames = features["n_frames"]
    labels = np.full(n_frames, X, dtype=np.int8)

    set_piece_events = features["set_piece_events"]
    goal_events = features["goal_events"]
    mean_spd = features["mean_speed_all"]
    n_tracked = features["n_tracked"]
    home_cx = features["home_centroid_x"]
    away_cx = features["away_centroid_x"]

    for _, first_vf, last_vf in half_video_blocks(game):
        h_start = max(0, int(first_vf))
        h_end = min(n_frames - 1, int(last_vf))
        if h_end < h_start:
            continue

        labels[h_start : h_end + 1] = UNDETERMINED

        # 1. Algorithmic opening kickoff detection inside this half.
        search_limit = min(h_end + 1, h_start + 6000)
        ko_events = [
            ev for ev in set_piece_events
            if ev["sp_type"] == "kick off (start)"
            and ev["end_frame"] > h_start
            and ev["start_frame"] < search_limit
        ]

        if ko_events:
            ev = ko_events[0]
            w_start = max(h_start, ev["start_frame"] - 25)
            w_end = min(h_end + 1, ev["end_frame"] + 25)
            t_kickoff = _detect_kickoff_frame(features, w_start, w_end, game=game)

            if t_kickoff > h_start and (t_kickoff - h_start) >= 15:
                # Confirm pre-kickoff segment reflects dead-ball staging rather than live play.
                pre_spd = mean_spd[h_start:t_kickoff]
                finite_pre = pre_spd[np.isfinite(pre_spd)]
                low_pre_spd = (len(finite_pre) == 0) or (float(np.median(finite_pre)) < 1.65)
                if ev["start_frame"] > h_start + 10 or low_pre_spd:
                    labels[h_start:t_kickoff] = D
        else:
            # Pure kinematic standby fallback if Kick Off (Start) tag is absent.
            sep = np.abs(home_cx[h_start:search_limit] - away_cx[h_start:search_limit])
            standby_mask = (
                (n_tracked[h_start:search_limit] < 10)
                | (
                    (np.isnan(mean_spd[h_start:search_limit]) | (mean_spd[h_start:search_limit] < 1.35))
                    & (np.isnan(sep) | (sep > 10.0))
                )
            )
            if len(standby_mask) > 0 and standby_mask[0]:
                runs = _contiguous_runs(standby_mask)
                if runs and runs[0][0] == 0 and (runs[0][1] - runs[0][0] + 1) >= 25:
                    labels[h_start : h_start + runs[0][1] + 1] = D

        # 2. Algorithmic post-goal kickoff restarts inside this half.
        ko_after_goal = [
            ev for ev in set_piece_events
            if ev["sp_type"] == "kick off (after goal)"
            and ev["start_frame"] >= h_start
            and ev["end_frame"] <= h_end + 1
        ]
        for ko_ev in ko_after_goal:
            w_s = max(h_start, ko_ev["start_frame"] - 10)
            w_e = min(h_end + 1, ko_ev["end_frame"] + 15)
            t_restart = _detect_kickoff_frame(features, w_s, w_e, game=game)
            if t_restart > ko_ev["start_frame"]:
                labels[ko_ev["start_frame"]:t_restart] = D

            # Bridge preceding goal celebration stretch up to the kickoff restart when inactive.
            matching_goals = [
                g for g in goal_events
                if h_start <= g["start_frame"] <= ko_ev["start_frame"]
                and (ko_ev["start_frame"] - g["end_frame"]) <= 2500
            ]
            if matching_goals:
                g_ev = matching_goals[-1]
                # Start dead ball only once ball leaves pitch or celebration walk begins inside goal window.
                scan_s = max(h_start, g_ev["start_frame"])
                scan_e = min(h_end + 1, ko_ev["start_frame"])
                if scan_e > scan_s:
                    oob_slice = features["ball_oob"][scan_s:scan_e]
                    low_slice = (
                        (~features["ball_visible"][scan_s:scan_e] | oob_slice)
                        & (np.isnan(mean_spd[scan_s:scan_e]) | (mean_spd[scan_s:scan_e] < 1.65))
                    )
                    first_dead = None
                    if np.any(oob_slice):
                        first_dead = scan_s + int(np.flatnonzero(oob_slice)[0])
                    elif np.any(low_slice):
                        first_dead = scan_s + int(np.flatnonzero(low_slice)[0])
                    if first_dead is not None and first_dead < ko_ev["start_frame"]:
                        labels[first_dead:t_restart] = D

    return labels


def _find_set_piece_restart(features, ev_start, ev_end, h_start, h_end):
    """Locate the restart strike/release frame inside a SET PIECES window."""
    w_s = max(h_start, ev_start)
    w_e = min(h_end + 1, ev_end)
    if w_e <= w_s:
        return w_s

    length, width = features["pitch"]
    bspeed = features["ball_speed"][w_s:w_e]
    baccel = features["ball_accel"][w_s:w_e]
    bdv = features["ball_dv"][w_s:w_e]
    boob = features["ball_oob"][w_s:w_e]
    bvis = features["ball_visible"][w_s:w_e]
    bxy_i = features["ball_xy_interp"][w_s:w_e]
    mean_spd = features["mean_speed_all"][w_s:w_e]
    p75_spd = features["p75_speed_all"][w_s:w_e]
    d_min = np.fmin(features["d_H_interp"][w_s:w_e], features["d_A_interp"][w_s:w_e])

    interp_in_bounds = (
        np.isfinite(bxy_i[:, 0])
        & np.isfinite(bxy_i[:, 1])
        & (bxy_i[:, 0] >= 0.0)
        & (bxy_i[:, 0] <= float(length))
        & (bxy_i[:, 1] >= 0.0)
        & (bxy_i[:, 1] <= float(width))
    )

    # Identify dead staging frames inside the tag window.
    stationary_or_oob = (
        boob
        | ~bvis
        | (np.isfinite(bspeed) & (bspeed < 1.8))
    )
    saw_dead_staging = False
    for i in range(w_e - w_s):
        is_in_bounds = bvis[i] and (not boob[i])
        spd_val = bspeed[i]
        acc_val = baccel[i]
        dv_val = bdv[i]
        m_val = mean_spd[i]
        p75_val = p75_spd[i]
        d_val = d_min[i]

        if stationary_or_oob[i] or (np.isfinite(m_val) and m_val < 1.55):
            saw_dead_staging = True

        if not saw_dead_staging:
            continue

        active_players = (
            (np.isfinite(p75_val) and p75_val >= 2.10)
            or (np.isfinite(m_val) and m_val >= 1.50)
        )
        kick_speed = is_in_bounds and np.isfinite(spd_val) and spd_val >= 2.8
        kick_interp_nan = (
            (not bvis[i])
            and interp_in_bounds[i]
            and np.isfinite(spd_val)
            and spd_val >= 2.8
            and active_players
        )
        vis_player_control = (
            is_in_bounds
            and np.isfinite(d_val)
            and d_val <= 1.5
            and np.isfinite(m_val)
            and m_val >= 1.50
        )
        kick_impulse = (
            is_in_bounds
            and ((np.isfinite(acc_val) and acc_val >= 12.0) or (np.isfinite(dv_val) and dv_val >= 2.5))
            and (np.isnan(d_val) or d_val <= 2.5)
        )
        # Distinguish genuine restart kicks/throws from slow dead-ball repositioning rolls.
        if kick_speed or kick_interp_nan or vis_player_control or kick_impulse:
            post_win = mean_spd[i : min(w_e - w_s, i + 20)]
            finite_post = post_win[np.isfinite(post_win)]
            active_post = (len(finite_post) == 0) or (float(np.mean(finite_post)) >= 1.45)
            strong_kick = np.isfinite(spd_val) and spd_val >= 4.5
            if not (strong_kick or active_post):
                continue
            # Verify ball is not immediately stepping back OOB on next frame.
            if i + 1 < (w_e - w_s) and boob[i + 1]:
                continue
            return w_s + i

    # Fallback: if the second half of the set-piece window enters active play.
    mid = (w_e - w_s) // 2
    for i in range(mid, w_e - w_s):
        if not boob[i]:
            high_p75 = np.isfinite(p75_spd[i]) and p75_spd[i] >= 2.25
            high_mean = np.isfinite(mean_spd[i]) and mean_spd[i] >= 1.65
            if high_p75 or high_mean:
                return w_s + i

    return w_e


def _bridge_short_true_gaps(mask, allow_bridge, max_gap=2):
    """Fill tiny 1-2 frame False blips inside True runs when allow_bridge holds."""
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        gap_len = j - i
        if i > 0 and j < n and gap_len <= max_gap and np.all(allow_bridge[i:j]):
            out[i:j] = True
        i = j
    return out


def run_pass1_high_confidence_anchors(features, labels_pass0, game=GAME):
    """Pass 1: lock in >95% high-confidence H, A, and D anchors inside each half."""
    labels = labels_pass0.copy()
    n_frames = features["n_frames"]

    ball_vis = features["ball_visible"]
    ball_oob = features["ball_oob"]
    ball_spd = features["ball_speed"]
    d_H = features["d_H"]
    d_A = features["d_A"]
    mean_spd = features["mean_speed_all"]
    p75_spd = features["p75_speed_all"]
    off_phase_team = features["off_phase_team"]
    is_set_piece = features["is_set_piece"]
    set_piece_events = features["set_piece_events"]

    for _, first_vf, last_vf in half_video_blocks(game):
        h_start = max(0, int(first_vf))
        h_end = min(n_frames - 1, int(last_vf))
        if h_end < h_start:
            continue

        # --- 1A. Definitive Dead Ball (D) Anchors ---
        # Sustained out-of-bounds ball (>= 3 consecutive frames).
        for rs, re in _contiguous_runs(ball_oob[h_start : h_end + 1]):
            if (re - rs + 1) >= 3:
                cur = labels[h_start + rs : h_start + re + 1]
                labels[h_start + rs : h_start + re + 1] = np.where(cur == UNDETERMINED, D, cur)

        # Smart Tagging SET PIECES staging windows prior to restart strike.
        for ev in set_piece_events:
            if ev["end_frame"] <= h_start or ev["start_frame"] > h_end:
                continue
            if ev["sp_type"] in ("kick off (start)", "kick off (after goal)"):
                continue

            ev_s = max(h_start, ev["start_frame"])
            ev_e = min(h_end + 1, ev["end_frame"])
            t_restart = _find_set_piece_restart(features, ev_s, ev_e, h_start, h_end)

            # Anchor confirmed dead staging frames leading up to t_restart.
            anchor_start = None
            for t in range(ev_s, t_restart):
                dead_like = (
                    ball_oob[t]
                    or (not ball_vis[t])
                    or (np.isfinite(mean_spd[t]) and mean_spd[t] < 1.60)
                    or (np.isfinite(ball_spd[t]) and ball_spd[t] < 1.60)
                )
                if dead_like:
                    if anchor_start is None:
                        anchor_start = t
                elif anchor_start is None:
                    # Live play at start of tag before dead staging begins; keep searching.
                    continue

            if anchor_start is not None and anchor_start < t_restart:
                # Extend backward while collective speed remains slow / ball dead.
                ext = anchor_start
                while ext > h_start and (anchor_start - ext) < 250:
                    prev = ext - 1
                    if labels[prev] != UNDETERMINED:
                        break
                    prev_dead = (
                        ball_oob[prev]
                        or (
                            (np.isnan(mean_spd[prev]) or mean_spd[prev] < 1.45)
                            and (not ball_vis[prev] or np.isnan(ball_spd[prev]) or ball_spd[prev] < 1.2)
                        )
                    )
                    if not prev_dead:
                        break
                    ext = prev

                seg = labels[ext:t_restart]
                labels[ext:t_restart] = np.where(seg == UNDETERMINED, D, seg)

        # --- 1B. Definitive Team Control (H / A) Anchors (>95% Certainty) ---
        # Run prior to unmarked slow-walk stoppage checks so ball-at-feet buildup is preserved.
        sl = slice(h_start, h_end + 1)
        und = labels[sl] == UNDETERMINED
        controlled_spd = np.isnan(ball_spd[sl]) | (ball_spd[sl] <= 9.5)
        in_play_ball = ball_vis[sl] & (~ball_oob[sl]) & und & controlled_spd

        # Exclude static dead-ball setups where a player stands holding a stationary ball.
        static_setup = (
            is_set_piece[sl]
            & (np.isnan(mean_spd[sl]) | (mean_spd[sl] < 1.45))
            & (np.isnan(ball_spd[sl]) | (ball_spd[sl] < 1.0))
        )
        in_play_ball = in_play_ball & (~static_setup)

        dH_sl = d_H[sl]
        dA_sl = d_A[sl]
        off_sl = off_phase_team[sl]

        strict_H = in_play_ball & np.isfinite(dH_sl) & (dH_sl <= 1.5) & (np.isnan(dA_sl) | (dA_sl > 2.5))
        strict_A = in_play_ball & np.isfinite(dA_sl) & (dA_sl <= 1.5) & (np.isnan(dH_sl) | (dH_sl > 2.5))

        reinf_H = (
            in_play_ball
            & np.isfinite(dH_sl)
            & (dH_sl <= 1.85)
            & (np.isnan(dA_sl) | (dA_sl > 2.8))
            & (off_sl == H)
        )
        reinf_A = (
            in_play_ball
            & np.isfinite(dA_sl)
            & (dA_sl <= 1.85)
            & (np.isnan(dH_sl) | (dH_sl > 2.8))
            & (off_sl == A)
        )

        cand_H = strict_H | reinf_H
        cand_A = strict_A | reinf_A

        # Mutual exclusivity safety.
        both = cand_H & cand_A
        cand_H[both] = False
        cand_A[both] = False

        # Bridge 1-2 frame occlusion blips inside active control runs.
        allow_H = und & (~ball_oob[sl]) & (~cand_A) & (np.isnan(dA_sl) | (dA_sl > 2.5))
        allow_A = und & (~ball_oob[sl]) & (~cand_H) & (np.isnan(dH_sl) | (dH_sl > 2.5))
        cand_H_b = _bridge_short_true_gaps(cand_H, allow_H, max_gap=2)
        cand_A_b = _bridge_short_true_gaps(cand_A, allow_A, max_gap=2)

        for rs, re in _contiguous_runs(cand_H_b):
            if (re - rs + 1) >= 10:
                cur = labels[h_start + rs : h_start + re + 1]
                labels[h_start + rs : h_start + re + 1] = np.where(cur == UNDETERMINED, H, cur)

        for rs, re in _contiguous_runs(cand_A_b):
            if (re - rs + 1) >= 10:
                cur = labels[h_start + rs : h_start + re + 1]
                labels[h_start + rs : h_start + re + 1] = np.where(cur == UNDETERMINED, A, cur)

        # Connect two crisp controlled touches (>= 8 frames each) separated by a clean short pass (<= 25 frames).
        for team_code, cand_arr, d_opp in ((H, cand_H_b, dA_sl), (A, cand_A_b, dH_sl)):
            runs = [r for r in _contiguous_runs(cand_arr) if (r[1] - r[0] + 1) >= 8]
            for idx_r in range(len(runs) - 1):
                r1_s, r1_e = runs[idx_r]
                r2_s, r2_e = runs[idx_r + 1]
                gap = r2_s - r1_e - 1
                if 0 < gap <= 25:
                    gap_sl = slice(h_start + r1_e + 1, h_start + r2_s)
                    if np.all(labels[gap_sl] == UNDETERMINED) and not np.any(ball_oob[gap_sl]):
                        opp_far = np.all(np.isnan(d_opp[r1_e + 1 : r2_s]) | (d_opp[r1_e + 1 : r2_s] > 2.3))
                        if opp_far:
                            labels[h_start + r1_s : h_start + r2_e + 1] = team_code

        # --- 1C. Unmarked Kinematic Stoppages (After 1B Team Control) ---
        # Require >= 75 frames (3.0 s) of collective standstill with no player visibly near ball.
        und_after_1b = labels[sl] == UNDETERMINED
        player_near_vis_ball = (
            ball_vis[sl]
            & (~ball_oob[sl])
            & ((np.isfinite(dH_sl) & (dH_sl <= 2.0)) | (np.isfinite(dA_sl) & (dA_sl <= 2.0)))
        )
        kinematic_dead = (
            und_after_1b
            & (~player_near_vis_ball)
            & (np.isnan(mean_spd[sl]) | (mean_spd[sl] < 1.20))
            & (np.isnan(p75_spd[sl]) | (p75_spd[sl] < 1.65))
            & (
                ball_oob[sl]
                | (~ball_vis[sl])
                | (np.isnan(ball_spd[sl]) | (ball_spd[sl] < 1.0))
            )
        )
        for rs, re in _contiguous_runs(kinematic_dead):
            if (re - rs + 1) >= 75:
                labels[h_start + rs : h_start + re + 1] = D

    return labels

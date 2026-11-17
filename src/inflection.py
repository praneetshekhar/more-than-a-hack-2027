"""Pass 2: Bidirectional [x - n, x + m] inflection-point & gap resolution."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks  # noqa: E402
from src import A, D, H, NOBODY, UNDETERMINED  # noqa: E402


def assign_possession_confidence(
    features,
    t_start,
    t_end,
    half_start,
    half_end,
    context_n=25,
    context_m=25,
    cv_cues=None,
):
    """Compute per-frame H/A/D confidence & inflection signals on inclusive [t_start, t_end].

    Uses bidirectional context [max(half_start, t_start - context_n),
    min(half_end, t_end + context_m)] strictly within the current half.
    """
    sl = slice(t_start, t_end + 1)
    width = t_end - t_start + 1

    d_H = features["d_H"][sl]
    d_A = features["d_A"][sl]
    d_H_interp = features["d_H_interp"][sl]
    d_A_interp = features["d_A_interp"][sl]

    ball_vis = features["ball_visible"][sl]
    d_H_eff = np.where(np.isfinite(d_H), d_H, d_H_interp).astype(np.float64)
    d_A_eff = np.where(np.isfinite(d_A), d_A, d_A_interp).astype(np.float64)

    # Attenuate proximity and suppress margin bonus when ball position is only interpolated across NaN.
    raw_prox_H = np.where(np.isfinite(d_H_eff), np.exp(-d_H_eff / 2.2), 0.20)
    raw_prox_A = np.where(np.isfinite(d_A_eff), np.exp(-d_A_eff / 2.2), 0.20)
    prox_H = np.where(ball_vis, raw_prox_H, 0.20 + 0.30 * (raw_prox_H - 0.20))
    prox_A = np.where(ball_vis, raw_prox_A, 0.20 + 0.30 * (raw_prox_A - 0.20))

    both_d = np.isfinite(d_H_eff) & np.isfinite(d_A_eff)
    margin = np.where(both_d, d_A_eff - d_H_eff, 0.0)
    d_closer = np.where(both_d, np.minimum(d_H_eff, d_A_eff), 10.0)
    reach_weight = np.where(ball_vis, np.exp(-np.maximum(d_closer - 2.0, 0.0) / 2.0), 0.0)
    score_H = prox_H + 0.35 * reach_weight * np.clip(margin / 3.0, 0.0, 1.0)
    score_A = prox_A + 0.35 * reach_weight * np.clip(-margin / 3.0, 0.0, 1.0)

    ball_oob = features["ball_oob"][sl]
    ball_spd = features["ball_speed"][sl].astype(np.float64)
    mean_spd = features["mean_speed_all"][sl].astype(np.float64)
    nearest_H_spd = features["nearest_H_speed"][sl].astype(np.float64)
    nearest_A_spd = features["nearest_A_speed"][sl].astype(np.float64)

    # Player-ball velocity coupling during active carries.
    coupled_H = (
        ball_vis
        & (~ball_oob)
        & np.isfinite(d_H_eff)
        & (d_H_eff <= 2.0)
        & np.isfinite(ball_spd)
        & np.isfinite(nearest_H_spd)
        & (np.abs(ball_spd - nearest_H_spd) <= 2.2)
    )
    coupled_A = (
        ball_vis
        & (~ball_oob)
        & np.isfinite(d_A_eff)
        & (d_A_eff <= 2.0)
        & np.isfinite(ball_spd)
        & np.isfinite(nearest_A_spd)
        & (np.abs(ball_spd - nearest_A_spd) <= 2.2)
    )
    score_H += np.where(coupled_H, 0.20, 0.0)
    score_A += np.where(coupled_A, 0.20, 0.0)

    # Smart Tagging tactical phase priors (stronger for explicit offensive tags).
    off_pt = features["off_phase_team"][sl]
    pt = features["phase_team"][sl]
    sp_team = features["set_piece_team"][sl]
    is_sp = features["is_set_piece"][sl]

    score_H += np.where(off_pt == H, 0.25, np.where(pt == H, 0.10, 0.0))
    score_A += np.where(off_pt == A, 0.25, np.where(pt == A, 0.10, 0.0))

    # Bidirectional collective context shift (closing velocity toward ball over [x - n, x + m]).
    ctx_s = max(half_start, t_start - context_n)
    ctx_e = min(half_end, t_end + context_m)
    dH_ctx = np.where(
        np.isfinite(features["d_H"][ctx_s : ctx_e + 1]),
        features["d_H"][ctx_s : ctx_e + 1],
        features["d_H_interp"][ctx_s : ctx_e + 1],
    ).astype(np.float64)
    dA_ctx = np.where(
        np.isfinite(features["d_A"][ctx_s : ctx_e + 1]),
        features["d_A"][ctx_s : ctx_e + 1],
        features["d_A_interp"][ctx_s : ctx_e + 1],
    ).astype(np.float64)

    rel_s = t_start - ctx_s
    for idx in range(width):
        c_idx = rel_s + idx
        post_hi = min(len(dH_ctx), c_idx + min(8, context_m) + 1)
        pre_lo = max(0, c_idx - min(8, context_n))
        if post_hi - c_idx >= 2 and c_idx - pre_lo >= 2:
            pre_h = dH_ctx[pre_lo:c_idx]
            post_h = dH_ctx[c_idx:post_hi]
            pre_a = dA_ctx[pre_lo:c_idx]
            post_a = dA_ctx[c_idx:post_hi]
            if np.any(np.isfinite(pre_h)) and np.any(np.isfinite(post_h)):
                if np.nanmean(post_h) < np.nanmean(pre_h) - 0.8 and np.nanmean(post_h) < 2.5:
                    score_H[idx] += 0.08
            if np.any(np.isfinite(pre_a)) and np.any(np.isfinite(post_a)):
                if np.nanmean(post_a) < np.nanmean(pre_a) - 0.8 and np.nanmean(post_a) < 2.5:
                    score_A[idx] += 0.08

    # Optional Targeted Local CV cues.
    if cv_cues:
        for idx in range(width):
            cue = cv_cues.get(t_start + idx)
            if cue is not None:
                score_H[idx] += 0.35 * (cue.get("team_prob_H", 0.5) - 0.5)
                score_A[idx] += 0.35 * (cue.get("team_prob_A", 0.5) - 0.5)
                if cue.get("ball_detected") and (sp_team[idx] == H):
                    score_H[idx] += 0.10
                elif cue.get("ball_detected") and (sp_team[idx] == A):
                    score_A[idx] += 0.10

    # Dead-ball score.
    slow_players = np.where(np.isfinite(mean_spd), np.clip((1.55 - mean_spd) / 0.65, 0.0, 1.0), 0.25)
    slow_ball = np.where(np.isfinite(ball_spd), np.clip((1.5 - ball_spd) / 1.5, 0.0, 1.0), 0.15)
    active_control = (~ball_oob) & ((d_H_eff <= 1.7) | (d_A_eff <= 1.7)) & (
        np.isnan(mean_spd) | (mean_spd >= 1.65)
    )
    score_D = (
        np.where(ball_oob, 1.15, 0.0)
        + np.where(is_sp & ((slow_players > 0.25) | (slow_ball > 0.45)), 0.45, 0.0)
        + 0.35 * slow_players
        + 0.25 * slow_ball
        - np.where(active_control, 0.35, 0.0)
    )
    score_D = np.clip(score_D, 0.0, 1.5)

    # Kinematic touch / inflection signal (velocity change or acceleration coinciding with proximity).
    bdv = np.nan_to_num(features["ball_dv"][sl].astype(np.float64), nan=0.0)
    bacc = np.nan_to_num(features["ball_accel"][sl].astype(np.float64), nan=0.0)
    kick_sig = np.maximum(np.clip(bdv / 8.0, 0.0, 1.5), np.clip(bacc / 30.0, 0.0, 1.5))

    near_H = np.where(np.isfinite(d_H_eff), np.exp(-np.maximum(d_H_eff - 1.2, 0.0) / 1.5), 0.25)
    near_A = np.where(np.isfinite(d_A_eff), np.exp(-np.maximum(d_A_eff - 1.2, 0.0) / 1.5), 0.25)
    inflection_H = kick_sig * near_H
    inflection_A = kick_sig * near_A

    return {
        "score_H": score_H,
        "score_A": score_A,
        "score_D": score_D,
        "inflection_H": inflection_H,
        "inflection_A": inflection_A,
        "d_H_eff": d_H_eff,
        "d_A_eff": d_A_eff,
    }


def _longest_contiguous_true_span(mask, max_bridge=3):
    """Return (start, end_inclusive) of the longest True run, bridging gaps <= max_bridge."""
    n = len(mask)
    best = None
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n:
            if mask[j + 1]:
                j += 1
                continue
            # Look ahead up to max_bridge frames
            nxt = None
            for k in range(1, max_bridge + 1):
                if j + 1 + k < n and mask[j + 1 + k]:
                    nxt = j + 1 + k
                    break
            if nxt is not None:
                j = nxt
            else:
                break
        if best is None or (j - i) > (best[1] - best[0]):
            best = (i, j)
        i = j + 1
    return best


def _find_best_split_index(left_scores, right_scores, split_bonus=None):
    """Find split index k in [0, W] maximizing sum(left[:k]) + sum(right[k:]) + bonus[k]."""
    w = len(left_scores)
    if w == 0:
        return 0
    pref_l = np.concatenate(([0.0], np.cumsum(left_scores)))
    suff_r = np.concatenate((np.cumsum(right_scores[::-1])[::-1], [0.0]))
    obj = pref_l + suff_r
    if split_bonus is not None:
        if len(split_bonus) == w:
            obj[:-1] += split_bonus
        elif len(split_bonus) == w + 1:
            obj += split_bonus
    return int(np.argmax(obj))


def _extrapolate_oob_exit(features, t_start, t_end, half_start):
    """Extrapolate last visible in-bounds ball velocity to estimate boundary crossing frame."""
    length, width = features["pitch"]
    ball_vis = features["ball_visible"]
    ball_oob = features["ball_oob"]
    ball_xy = features["ball_xy"]
    vx = features["ball_vx"]
    vy = features["ball_vy"]
    fps = features["fps"]

    search_lo = max(half_start, t_start - 15)
    last_vis = None
    for t in range(t_end, search_lo - 1, -1):
        if ball_vis[t] and not ball_oob[t]:
            last_vis = t
            break

    if last_vis is None or np.isnan(vx[last_vis]) or np.isnan(vy[last_vis]):
        return None

    vx_val = float(vx[last_vis])
    vy_val = float(vy[last_vis])
    if np.hypot(vx_val, vy_val) < 1.5:
        return None

    bx = float(ball_xy[last_vis, 0])
    by = float(ball_xy[last_vis, 1])
    dts = []
    if vx_val > 0.5:
        dts.append((length - bx) / vx_val)
    elif vx_val < -0.5:
        dts.append((0.0 - bx) / vx_val)
    if vy_val > 0.5:
        dts.append((width - by) / vy_val)
    elif vy_val < -0.5:
        dts.append((0.0 - by) / vy_val)

    pos_dts = [dt for dt in dts if 0.0 <= dt <= 2.5]
    if not pos_dts:
        return None
    dt_min = min(pos_dts)
    t_exit = int(round(last_vis + dt_min * fps))
    if t_start <= t_exit <= t_end + 1:
        return t_exit
    return None


def _unanchored_set_piece_mask(features, t_start, t_end, labels=None):
    """Mask of SET PIECES overlap inside [t_start, t_end] not already anchored as D in Pass 1."""
    width = t_end - t_start + 1
    mask = np.zeros(width, dtype=bool)
    n_frames = features["n_frames"]
    for ev in features.get("set_piece_events", ()):
        ev_s = int(ev["start_frame"])
        ev_e = int(ev["end_frame"])
        if ev_e <= t_start or ev_s > t_end:
            continue
        if labels is not None:
            chk_s = max(0, ev_s - 25)
            chk_e = min(n_frames, ev_e + 1)
            if np.any(labels[chk_s:chk_e] == D):
                # Pass 1 already anchored this set-piece stoppage; remaining overlap is live flight.
                continue
        ov_s = max(t_start, ev_s) - t_start
        ov_e = min(t_end + 1, ev_e) - t_start
        if ov_e > ov_s:
            mask[ov_s:ov_e] = True
    return mask


def _detect_intermediate_stoppage(features, conf, t_start, t_end, labels=None):
    """Return inclusive [d0, d1] dead-ball sub-interval inside [t_start, t_end], or None."""
    sl = slice(t_start, t_end + 1)
    width = t_end - t_start + 1
    if width < 10:
        return None

    unanchored_sp = _unanchored_set_piece_mask(features, t_start, t_end, labels=labels)
    has_sp = bool(np.any(unanchored_sp))
    oob_cnt = int(np.sum(features["ball_oob"][sl]))
    mean_spd = features["mean_speed_all"][sl]
    score_D = conf["score_D"]
    score_live = np.maximum(conf["score_H"], conf["score_A"])

    dead_cand = (
        features["ball_oob"][sl]
        | (
            (score_D > score_live + 0.12)
            & (np.isnan(mean_spd) | (mean_spd < 1.55))
        )
    )
    n_dead_cand = int(np.sum(dead_cand))
    if not (has_sp or oob_cnt >= 6 or n_dead_cand >= 75):
        return None

    min_span_len = 6 if (has_sp or oob_cnt >= 6) else 50

    # Find longest / strongest dead-ball core.
    best_span = None
    i = 0
    while i < width:
        if not dead_cand[i]:
            i += 1
            continue
        j = i
        while j + 1 < width and (dead_cand[j + 1] or (j + 2 < width and dead_cand[j + 2])):
            j += 1
        span_len = j - i + 1
        if span_len >= min_span_len and (best_span is None or span_len > (best_span[1] - best_span[0] + 1)):
            best_span = (t_start + i, t_start + j)
        i = j + 1

    if best_span is None and has_sp:
        sp_idx = np.flatnonzero(unanchored_sp)
        if len(sp_idx) > 0:
            best_span = (t_start + int(sp_idx[0]), t_start + int(sp_idx[-1]))

    return best_span


def _resolve_to_dead(labels, features, conf, t_start, t_end, half_start, team_L):
    """Resolve H/A -> ? -> D stoppage boundary inside [t_start, t_end]."""
    sl = slice(t_start, t_end + 1)
    score_L = conf["score_H"] if team_L == H else conf["score_A"]
    score_D = conf["score_D"].copy()

    # Penalize keeping live team after out-of-bounds begins.
    oob = features["ball_oob"][sl]
    if np.any(oob):
        first_oob = int(np.flatnonzero(oob)[0])
        score_D[first_oob:] += 0.8

    bonus = np.zeros(t_end - t_start + 1, dtype=np.float64)
    t_exit = _extrapolate_oob_exit(features, t_start, t_end, half_start)
    if t_exit is not None and t_start <= t_exit <= t_end:
        bonus[t_exit - t_start] += 0.45

    k = _find_best_split_index(score_L, score_D, split_bonus=bonus)
    t_star = t_start + k
    if t_star > t_start:
        labels[t_start:t_star] = team_L
    if t_star <= t_end:
        labels[t_star : t_end + 1] = D


def _resolve_from_dead(labels, features, conf, t_start, t_end, team_R):
    """Resolve D -> ? -> H/A restart boundary inside [t_start, t_end]."""
    sl = slice(t_start, t_end + 1)
    width = t_end - t_start + 1
    score_R = conf["score_H"] if team_R == H else conf["score_A"]
    score_D = conf["score_D"]

    ball_vis = features["ball_visible"][sl]
    ball_oob = features["ball_oob"][sl]
    ball_spd = features["ball_speed"][sl]
    ball_acc = features["ball_accel"][sl]
    ball_dv = features["ball_dv"][sl]
    touchline = features["ball_near_touchline"][sl]
    is_sp = features["is_set_piece"][sl]
    sp_type = features["set_piece_type"][sl]

    # Locate restart kick / throw-in release cue.
    restart_bonus = np.zeros(width, dtype=np.float64)
    for i in range(width):
        if ball_oob[i]:
            continue
        is_throw = is_sp[i] and (sp_type[i] == "throw-in")
        if is_throw and touchline[i] and np.isfinite(ball_spd[i]) and ball_spd[i] < 2.0:
            continue
        spd_spike = ball_vis[i] and np.isfinite(ball_spd[i]) and ball_spd[i] >= 2.5
        acc_spike = ball_vis[i] and (
            (np.isfinite(ball_acc[i]) and ball_acc[i] >= 10.0)
            or (np.isfinite(ball_dv[i]) and ball_dv[i] >= 2.2)
        )
        if spd_spike or acc_spike:
            restart_bonus[i] += 0.55

    k = _find_best_split_index(score_D, score_R, split_bonus=restart_bonus)
    t_star = t_start + k
    if t_star > t_start:
        labels[t_start:t_star] = D
    if t_star <= t_end:
        labels[t_star : t_end + 1] = team_R


def resolve_gap(
    labels,
    features,
    t_start,
    t_end,
    half_start,
    half_end,
    context_n=25,
    context_m=25,
    cv_cues=None,
):
    """Resolve a single contiguous UNDETERMINED run [t_start, t_end] in-place using anchors L and R."""
    if t_end < t_start:
        return

    left_anchor = int(labels[t_start - 1]) if t_start > half_start else UNDETERMINED
    right_anchor = int(labels[t_end + 1]) if t_end < half_end else UNDETERMINED

    conf = assign_possession_confidence(
        features,
        t_start,
        t_end,
        half_start,
        half_end,
        context_n=context_n,
        context_m=context_m,
        cv_cues=cv_cues,
    )
    width = t_end - t_start + 1

    # Case 1: Same-team bridge (H -> ? -> H or A -> ? -> A).
    if left_anchor in (H, A) and right_anchor == left_anchor:
        stop_span = _detect_intermediate_stoppage(features, conf, t_start, t_end, labels=labels)
        if stop_span is not None:
            d0, d1 = stop_span
            if d0 > t_start:
                c_left = assign_possession_confidence(
                    features, t_start, d0 - 1, half_start, half_end, context_n, context_m, cv_cues
                )
                _resolve_to_dead(labels, features, c_left, t_start, d0 - 1, half_start, left_anchor)
            labels[d0 : d1 + 1] = D
            if d1 < t_end:
                c_right = assign_possession_confidence(
                    features, d1 + 1, t_end, half_start, half_end, context_n, context_m, cv_cues
                )
                _resolve_from_dead(labels, features, c_right, d1 + 1, t_end, right_anchor)
            return

        # Long gaps (> 75 frames) with sustained visible, controlled opposing play split cleanly.
        opp = 1 - left_anchor
        sl = slice(t_start, t_end + 1)
        score_own = conf["score_H"] if left_anchor == H else conf["score_A"]
        score_opp = conf["score_H"] if opp == H else conf["score_A"]
        d_opp_raw = features["d_H"][sl] if opp == H else features["d_A"][sl]
        d_own_raw = features["d_H"][sl] if left_anchor == H else features["d_A"][sl]
        bvis_sl = features["ball_visible"][sl]
        boob_sl = features["ball_oob"][sl]
        bspd_sl = features["ball_speed"][sl]
        opp_dom = (
            bvis_sl
            & (~boob_sl)
            & (score_opp - score_own > 0.35)
            & np.isfinite(d_opp_raw)
            & (d_opp_raw <= 1.8)
            & (np.isnan(d_own_raw) | (d_own_raw > 2.8))
            & (np.isnan(bspd_sl) | (bspd_sl <= 8.5))
        )
        if width > 75 and np.sum(opp_dom) >= 30:
            span = _longest_contiguous_true_span(opp_dom, max_bridge=3)
            if span is not None and (span[1] - span[0] + 1) >= 30:
                s_opp, e_opp = span
                labels[t_start : t_start + s_opp] = left_anchor
                labels[t_start + s_opp : t_start + e_opp + 1] = opp
                labels[t_start + e_opp + 1 : t_end + 1] = right_anchor
                return

        labels[t_start : t_end + 1] = left_anchor
        return

    # Case 2: Opposite-team turnover (H -> ? -> A or A -> ? -> H).
    if left_anchor in (H, A) and right_anchor in (H, A) and left_anchor != right_anchor:
        stop_span = _detect_intermediate_stoppage(features, conf, t_start, t_end, labels=labels)
        if stop_span is not None:
            d0, d1 = stop_span
            if d0 > t_start:
                c_left = assign_possession_confidence(
                    features, t_start, d0 - 1, half_start, half_end, context_n, context_m, cv_cues
                )
                _resolve_to_dead(labels, features, c_left, t_start, d0 - 1, half_start, left_anchor)
            labels[d0 : d1 + 1] = D
            if d1 < t_end:
                c_right = assign_possession_confidence(
                    features, d1 + 1, t_end, half_start, half_end, context_n, context_m, cv_cues
                )
                _resolve_from_dead(labels, features, c_right, d1 + 1, t_end, right_anchor)
            return

        # Persistence bias keeps live play with left_anchor during NaN flight or while
        # right_anchor is not yet within visible ball reach (d_R_raw <= 2.2 m).
        sl = slice(t_start, t_end + 1)
        d_R_raw = features["d_H"][sl] if right_anchor == H else features["d_A"][sl]
        bvis_sl = features["ball_visible"][sl]
        in_flight_away = (~bvis_sl) | np.isnan(d_R_raw) | (d_R_raw > 2.2)
        score_L = (
            (conf["score_H"] if left_anchor == H else conf["score_A"])
            + 0.08
            + np.where(in_flight_away, 0.35, 0.0)
        )
        score_R = conf["score_H"] if right_anchor == H else conf["score_A"]
        if not np.all(in_flight_away):
            score_R = np.where(in_flight_away, np.minimum(score_R, 0.20), score_R)
        infl_R = conf["inflection_H"] if right_anchor == H else conf["inflection_A"]

        k = _find_best_split_index(score_L, score_R, split_bonus=0.60 * infl_R)
        t_star = t_start + k
        if t_star > t_start:
            labels[t_start:t_star] = left_anchor
        if t_star <= t_end:
            labels[t_star : t_end + 1] = right_anchor
        return

    # Case 3: Stoppage entry (H/A -> ? -> D).
    if left_anchor in (H, A) and right_anchor == D:
        _resolve_to_dead(labels, features, conf, t_start, t_end, half_start, left_anchor)
        return

    # Case 4: Stoppage restart (D -> ? -> H/A), including Team=N/A throw-ins.
    if left_anchor == D and right_anchor in (H, A):
        _resolve_from_dead(labels, features, conf, t_start, t_end, right_anchor)
        return

    # Case 5: Dead-to-dead (D -> ? -> D).
    if left_anchor == D and right_anchor == D:
        score_H = conf["score_H"]
        score_A = conf["score_A"]
        score_D = conf["score_D"]
        live_best = np.maximum(score_H, score_A)
        live_mask = (live_best > score_D + 0.25) & (~features["ball_oob"][t_start : t_end + 1])
        if int(np.sum(live_mask)) >= 30:
            span = _longest_contiguous_true_span(live_mask, max_bridge=3)
            if span is not None and (span[1] - span[0] + 1) >= 30:
                s_l, e_l = span
                seg_h = float(np.sum(score_H[s_l : e_l + 1]))
                seg_a = float(np.sum(score_A[s_l : e_l + 1]))
                team = H if seg_h >= seg_a else A
                labels[t_start : t_start + s_l] = D
                labels[t_start + s_l : t_start + e_l + 1] = team
                labels[t_start + e_l + 1 : t_end + 1] = D
                return
        labels[t_start : t_end + 1] = D
        return

    # Case 6: Half boundary open edge(s) where L or R is UNDETERMINED.
    if left_anchor in (H, A, D) and right_anchor == UNDETERMINED:
        if left_anchor in (H, A):
            stop_span = _detect_intermediate_stoppage(features, conf, t_start, t_end, labels=labels)
            if stop_span is not None:
                d0, _ = stop_span
                labels[t_start:d0] = left_anchor
                labels[d0 : t_end + 1] = D
                return
        labels[t_start : t_end + 1] = left_anchor
        return

    if left_anchor == UNDETERMINED and right_anchor in (H, A, D):
        if right_anchor in (H, A):
            stop_span = _detect_intermediate_stoppage(features, conf, t_start, t_end, labels=labels)
            if stop_span is not None:
                _, d1 = stop_span
                labels[t_start : d1 + 1] = D
                labels[d1 + 1 : t_end + 1] = right_anchor
                return
        labels[t_start : t_end + 1] = right_anchor
        return

    # Fallback if neither endpoint was anchored (e.g., very short synthetic clip).
    stacked = np.vstack([conf["score_H"], conf["score_A"], conf["score_D"]])
    labels[t_start : t_end + 1] = np.argmax(stacked, axis=0).astype(np.int8)


def run_pass2_inflection_resolution(
    features,
    labels_pass1,
    game=GAME,
    context_n=25,
    context_m=25,
    cv_cues=None,
):
    """Pass 2: Resolve all UNDETERMINED intervals strictly per half."""
    labels = labels_pass1.copy()
    n_frames = features["n_frames"]

    for _, first_vf, last_vf in half_video_blocks(game):
        h_start = max(0, int(first_vf))
        h_end = min(n_frames - 1, int(last_vf))
        if h_end < h_start:
            continue

        t = h_start
        while t <= h_end:
            if labels[t] != UNDETERMINED:
                t += 1
                continue
            j = t
            while j + 1 <= h_end and labels[j + 1] == UNDETERMINED:
                j += 1

            resolve_gap(
                labels,
                features,
                t,
                j,
                h_start,
                h_end,
                context_n=context_n,
                context_m=context_m,
                cv_cues=cv_cues,
            )
            t = j + 1

    return labels

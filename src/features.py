"""Streaming ATD loader, gitignored .npz cache, Smart Tagging alignment, and per-half kinematics."""

import os
import sys
import warnings
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging_xml  # noqa: E402
from src import A, D, H, NOBODY  # noqa: E402

FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING_CHAR = "N"

OFFENSIVE_PHASE_CODES = frozenset({
    "BUILD UP",
    "PROGRESSION",
    "CREATING CHANCES",
    "BALL IN FINAL THIRD",
    "BALL IN THE BOX",
    "PLAYERS IN THE BOX",
    "ATTACKING TRANSITION",
    "LONG BALL",
})

DEFENSIVE_PHASE_CODES = frozenset({
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENSIVE TRANSITION",
})


def normalize_team_name(name):
    """Normalize ATD and Smart Tagging team strings to H (0), A (1), or NOBODY (-1).

    Handles the Smart Tagging away-team spelling ('Atletic Club') as well as ATD XML
    ('Athletic Club') and 'N/A' on throw-ins and unassigned tags.
    """
    if name is None:
        return NOBODY
    text = str(name).strip().lower()
    if not text or text in ("n/a", "na", "none", "nobody"):
        return NOBODY
    if "barc" in text:
        return H
    if "atlet" in text or "athlet" in text:
        return A
    return NOBODY


def parse_atd_meta(meta_path):
    """Parse FIFA/EPTS metadata XML to map each track slot to H (0) or A (1)."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))

    home_id = teams[0].get("id")
    team_names = tuple((team.findtext("Name") or "").strip() for team in teams)

    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries found".format(meta_path))

    team_of_slot = np.array(
        [H if team_id == home_id else A for team_id in slots],
        dtype=np.int8,
    )
    return team_of_slot, team_names


def _parse_group_triplet(group_str):
    """Parse a single 'x,y,speed' group into (x, y, speed) floats."""
    if not group_str or group_str[0] in ("N", "n"):
        return np.nan, np.nan, np.nan
    parts = group_str.split(",")
    x_val = float(parts[0])
    y_val = float(parts[1]) if len(parts) > 1 else np.nan
    spd_val = float(parts[2]) if len(parts) > 2 else np.nan
    return x_val, y_val, spd_val


def _stream_raw_atd(raw_path, n_frames, n_slots, pitch):
    """Stream raw _atd.txt line-by-line into float32 metric arrays."""
    length, width = float(pitch[0]), float(pitch[1])
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball_norm_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_speed_raw = np.full(n_frames, np.nan, dtype=np.float32)
    ball_visible = np.zeros(n_frames, dtype=bool)

    expected_vals = n_slots * 3
    with open(raw_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            first_colon = line.find(FIELD_SEP)
            second_colon = line.find(FIELD_SEP, first_colon + 1)
            if first_colon < 0 or second_colon < 0:
                continue

            vf = int(line[:first_colon]) - 1
            if vf < 0 or vf >= n_frames:
                continue

            players_part = line[first_colon + 1:second_colon]
            if players_part.endswith(GROUP_SEP):
                players_part = players_part[:-1]

            arr = np.fromstring(
                players_part.replace(GROUP_SEP, ","),
                dtype=np.float32,
                sep=",",
            )
            if arr.size == expected_vals:
                players[vf] = arr.reshape(n_slots, 3)
            else:
                groups = players_part.split(GROUP_SEP)
                for slot_idx, grp in enumerate(groups[:n_slots]):
                    if not grp or grp[0] in ("N", "n"):
                        continue
                    players[vf, slot_idx] = _parse_group_triplet(grp)

            ball_part = line[second_colon + 1:].strip()
            if ball_part and ball_part[0] not in ("N", "n"):
                bx, by, bs = _parse_group_triplet(ball_part)
                if not np.isnan(bx) and not np.isnan(by):
                    ball_norm_xy[vf, 0] = bx
                    ball_norm_xy[vf, 1] = by
                    ball_speed_raw[vf] = bs
                    ball_visible[vf] = True

    players[:, :, 0] *= np.float32(length)
    players[:, :, 1] *= np.float32(width)
    ball_xy = np.empty_like(ball_norm_xy)
    ball_xy[:, 0] = ball_norm_xy[:, 0] * np.float32(length)
    ball_xy[:, 1] = ball_norm_xy[:, 1] * np.float32(width)

    return {
        "players": players,
        "ball_xy": ball_xy,
        "ball_norm_xy": ball_norm_xy,
        "ball_speed_raw": ball_speed_raw,
        "ball_visible": ball_visible,
    }


def atd_cache_path(game=GAME, data_dir=None):
    """Return the gitignored .npz cache path inside data/.cache/ (or MTAH_DATA_DIR/.cache/)."""
    root_dir = data_dir or os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")
    return os.path.join(root_dir, ".cache", "{}_tactical_atd.npz".format(game))


def load_or_build_atd_cache(game=GAME, data_dir=None, use_cache=True):
    """Load ATD tracking arrays from .npz cache or stream raw _atd.txt and cache them."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    pitch = GAMES[game]["pitch"]
    team_of_slot, team_names = parse_atd_meta(meta_path)
    n_slots = len(team_of_slot)
    cache_file = atd_cache_path(game, data_dir)

    if use_cache and os.path.isfile(cache_file):
        try:
            with np.load(cache_file, allow_pickle=False) as cached:
                cached_pitch = cached["pitch"] if "pitch" in cached else None
                if (
                    cached["players"].shape == (n_frames, n_slots, 3)
                    and cached["team_of_slot"].shape == (n_slots,)
                    and np.array_equal(cached["team_of_slot"], team_of_slot)
                    and cached_pitch is not None
                    and np.allclose(cached_pitch, np.array(pitch, dtype=np.float32))
                ):
                    return {
                        "players": cached["players"],
                        "ball_xy": cached["ball_xy"],
                        "ball_norm_xy": cached["ball_norm_xy"],
                        "ball_speed_raw": cached["ball_speed_raw"],
                        "ball_visible": cached["ball_visible"].astype(bool),
                        "team_of_slot": cached["team_of_slot"].astype(np.int8),
                        "team_names": team_names,
                        "cache_hit": True,
                        "cache_path": cache_file,
                    }
        except Exception:
            pass

    data = _stream_raw_atd(raw_path, n_frames, n_slots, pitch)
    data["team_of_slot"] = team_of_slot
    data["team_names"] = team_names
    data["cache_hit"] = False
    data["cache_path"] = cache_file

    if use_cache:
        cache_dir = os.path.dirname(cache_file)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(
            cache_file,
            players=data["players"],
            ball_xy=data["ball_xy"],
            ball_norm_xy=data["ball_norm_xy"],
            ball_speed_raw=data["ball_speed_raw"],
            ball_visible=data["ball_visible"],
            team_of_slot=team_of_slot,
            pitch=np.array(pitch, dtype=np.float32),
        )
    return data


def load_smart_tagging_features(game=GAME, data_dir=None, n_frames=None):
    """Parse Smart Tagging XML into frame-aligned boolean/categorical priors and event lists."""
    if n_frames is None:
        n_frames = source_frames("tactical", game)

    events = load_smart_tagging_xml(game=game, data_dir=data_dir)
    off_votes = np.zeros((n_frames, 2), dtype=np.int16)
    def_votes = np.zeros((n_frames, 2), dtype=np.int16)

    is_set_piece = np.zeros(n_frames, dtype=bool)
    set_piece_team = np.full(n_frames, NOBODY, dtype=np.int8)
    set_piece_type = np.full(n_frames, "", dtype=object)
    is_goal = np.zeros(n_frames, dtype=bool)

    set_piece_events = []
    goal_events = []

    for ev in events:
        code = (ev.get("code") or "").strip().upper()
        s = max(0, int(ev["start_frame"]))
        e = min(n_frames, int(ev["end_frame"]))
        if e <= s:
            continue
        tags = ev.get("tags") or {}
        team = normalize_team_name(tags.get("Team"))

        if code in OFFENSIVE_PHASE_CODES and team in (H, A):
            off_votes[s:e, team] += 1
        elif code in DEFENSIVE_PHASE_CODES and team in (H, A):
            def_votes[s:e, team] += 1
        elif code == "SET PIECES":
            sp_type = (tags.get("Type") or "").strip().lower()
            is_set_piece[s:e] = True
            if team in (H, A):
                set_piece_team[s:e] = team
            if sp_type:
                set_piece_type[s:e] = sp_type
            set_piece_events.append({
                "start_frame": s,
                "end_frame": e,
                "team": team,
                "sp_type": sp_type,
                "half": (tags.get("Half") or "").strip(),
            })
        elif code == "GOALS":
            is_goal[s:e] = True
            goal_events.append({
                "start_frame": s,
                "end_frame": e,
                "team": team,
                "half": (tags.get("Half") or "").strip(),
            })

    off_phase_team = np.full(n_frames, NOBODY, dtype=np.int8)
    off_h_win = off_votes[:, H] > off_votes[:, A]
    off_a_win = off_votes[:, A] > off_votes[:, H]
    off_phase_team[off_h_win] = H
    off_phase_team[off_a_win] = A

    phase_team = off_phase_team.copy()
    tie_mask = phase_team == NOBODY
    # Defending by Away implies Home is attacking, and vice versa.
    phase_team[tie_mask & (def_votes[:, A] > def_votes[:, H])] = H
    phase_team[tie_mask & (def_votes[:, H] > def_votes[:, A])] = A

    set_piece_events.sort(key=lambda item: (item["start_frame"], item["end_frame"]))
    goal_events.sort(key=lambda item: (item["start_frame"], item["end_frame"]))

    return {
        "phase_team": phase_team,
        "off_phase_team": off_phase_team,
        "off_votes": off_votes,
        "def_votes": def_votes,
        "is_set_piece": is_set_piece,
        "set_piece_team": set_piece_team,
        "set_piece_type": set_piece_type,
        "is_goal": is_goal,
        "set_piece_events": set_piece_events,
        "goal_events": goal_events,
    }


def _nearest_team_metrics(players, ball_xy, ball_visible, team_mask):
    """Compute nearest/second-nearest distance and nearest player speed for one team."""
    n_frames = players.shape[0]
    if not np.any(team_mask):
        nan_arr = np.full(n_frames, np.nan, dtype=np.float32)
        return nan_arr, nan_arr.copy(), nan_arr.copy()

    team_players = players[:, team_mask, :]
    valid = (
        ball_visible[:, None]
        & ~np.isnan(team_players[:, :, 0])
        & ~np.isnan(team_players[:, :, 1])
    )
    dx = team_players[:, :, 0] - ball_xy[:, 0:1]
    dy = team_players[:, :, 1] - ball_xy[:, 1:2]
    with np.errstate(invalid="ignore"):
        dists = np.where(valid, np.hypot(dx, dy), np.inf)

    if team_players.shape[1] >= 2:
        part = np.partition(dists, 1, axis=1)
        d1 = np.where(np.isfinite(part[:, 0]), part[:, 0], np.nan).astype(np.float32)
        d2 = np.where(np.isfinite(part[:, 1]), part[:, 1], np.nan).astype(np.float32)
    else:
        min_d = np.min(dists, axis=1)
        d1 = np.where(np.isfinite(min_d), min_d, np.nan).astype(np.float32)
        d2 = np.full(n_frames, np.nan, dtype=np.float32)

    idx_min = np.argmin(dists, axis=1)
    nearest_spd = np.where(
        np.isfinite(d1),
        team_players[np.arange(n_frames), idx_min, 2],
        np.nan,
    ).astype(np.float32)
    return d1, d2, nearest_spd


def _box_smooth_1d(arr, half_win=2):
    """NaN-aware moving mean strictly within a single half slice."""
    n = len(arr)
    if n == 0:
        return arr.copy()
    valid = (~np.isnan(arr)).astype(np.float64)
    vals = np.where(valid > 0, arr.astype(np.float64), 0.0)
    csum = np.concatenate(([0.0], np.cumsum(vals)))
    ccnt = np.concatenate(([0.0], np.cumsum(valid)))
    idx = np.arange(n)
    lo = np.maximum(0, idx - half_win)
    hi = np.minimum(n, idx + half_win + 1)
    cnt = ccnt[hi] - ccnt[lo]
    out = np.where(cnt > 0, (csum[hi] - csum[lo]) / np.maximum(cnt, 1.0), np.nan)
    return out.astype(np.float32)


def compute_kinematic_features(atd_data, st_data, game=GAME, max_interp_gap=75):
    """Compute per-frame player/ball metric kinematics strictly isolated per half."""
    players = atd_data["players"]
    ball_xy = atd_data["ball_xy"]
    ball_norm_xy = atd_data["ball_norm_xy"]
    ball_speed_raw = atd_data["ball_speed_raw"]
    ball_visible = atd_data["ball_visible"]
    team_of_slot = atd_data["team_of_slot"]

    n_frames = players.shape[0]
    fps = float(GAMES[game]["fps"])
    pitch = GAMES[game]["pitch"]
    length, width = pitch
    center_x, center_y = length / 2.0, width / 2.0

    home_mask = team_of_slot == H
    away_mask = team_of_slot == A

    valid_p = ~np.isnan(players[:, :, 0]) & ~np.isnan(players[:, :, 1])
    n_tracked = valid_p.sum(axis=1).astype(np.int16)

    spd = players[:, :, 2]
    spd_valid = ~np.isnan(spd)
    spd_cnt = spd_valid.sum(axis=1)
    spd_sum = np.where(spd_valid, spd, 0.0).sum(axis=1)
    mean_speed_all = np.where(spd_cnt > 0, spd_sum / np.maximum(spd_cnt, 1), np.nan).astype(np.float32)
    p75_speed_all = np.full(n_frames, np.nan, dtype=np.float32)

    d_H, d2_H, nearest_H_speed = _nearest_team_metrics(players, ball_xy, ball_visible, home_mask)
    d_A, d2_A, nearest_A_speed = _nearest_team_metrics(players, ball_xy, ball_visible, away_mask)

    ball_oob = np.zeros(n_frames, dtype=bool)
    ball_oob[ball_visible] = (
        (ball_xy[ball_visible, 0] < 0.0)
        | (ball_xy[ball_visible, 0] > length)
        | (ball_xy[ball_visible, 1] < 0.0)
        | (ball_xy[ball_visible, 1] > width)
    )
    ball_near_touchline = np.zeros(n_frames, dtype=bool)
    ball_near_touchline[ball_visible] = (
        (ball_xy[ball_visible, 1] < 2.2) | (ball_xy[ball_visible, 1] > width - 2.2)
    )
    ball_near_center = np.zeros(n_frames, dtype=bool)
    ball_near_center[ball_visible] = (
        np.hypot(ball_xy[ball_visible, 0] - center_x, ball_xy[ball_visible, 1] - center_y) <= 4.0
    )

    ball_xy_interp = np.full((n_frames, 2), np.nan, dtype=np.float32)
    interp_gap_len = np.full(n_frames, n_frames + 1, dtype=np.int32)
    ball_vx = np.full(n_frames, np.nan, dtype=np.float32)
    ball_vy = np.full(n_frames, np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    ball_ax = np.full(n_frames, np.nan, dtype=np.float32)
    ball_ay = np.full(n_frames, np.nan, dtype=np.float32)
    ball_accel = np.full(n_frames, np.nan, dtype=np.float32)
    ball_dv = np.full(n_frames, np.nan, dtype=np.float32)

    home_centroid_x = np.full(n_frames, np.nan, dtype=np.float32)
    away_centroid_x = np.full(n_frames, np.nan, dtype=np.float32)
    team_vx_H = np.full(n_frames, np.nan, dtype=np.float32)
    team_vx_A = np.full(n_frames, np.nan, dtype=np.float32)

    if np.any(home_mask):
        hx = players[:, home_mask, 0]
        h_val = ~np.isnan(hx)
        h_cnt = h_val.sum(axis=1)
        home_centroid_x = np.where(
            h_cnt > 0,
            np.where(h_val, hx, 0.0).sum(axis=1) / np.maximum(h_cnt, 1),
            np.nan,
        ).astype(np.float32)
    if np.any(away_mask):
        ax_p = players[:, away_mask, 0]
        a_val = ~np.isnan(ax_p)
        a_cnt = a_val.sum(axis=1)
        away_centroid_x = np.where(
            a_cnt > 0,
            np.where(a_val, ax_p, 0.0).sum(axis=1) / np.maximum(a_cnt, 1),
            np.nan,
        ).astype(np.float32)

    # Strict per-half isolation for all temporal interpolations, gradients, and percentiles.
    for _, first_vf, last_vf in half_video_blocks(game):
        if first_vf >= n_frames or last_vf < 0:
            continue
        s = max(0, int(first_vf))
        e = min(n_frames, int(last_vf) + 1)
        n_half = e - s
        if n_half <= 0:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            has_spd = spd_cnt[s:e] > 0
            if np.any(has_spd):
                p75_sub = np.full(n_half, np.nan, dtype=np.float32)
                p75_sub[has_spd] = np.nanpercentile(spd[s:e][has_spd], 75, axis=1).astype(np.float32)
                p75_speed_all[s:e] = p75_sub

        # Team collective x-velocity within this half only.
        if n_half >= 3:
            hc = _box_smooth_1d(home_centroid_x[s:e], half_win=2)
            ac = _box_smooth_1d(away_centroid_x[s:e], half_win=2)
            tv_h = np.full(n_half, np.nan, dtype=np.float32)
            tv_a = np.full(n_half, np.nan, dtype=np.float32)
            tv_h[1:-1] = (hc[2:] - hc[:-2]) * np.float32(fps / 2.0)
            tv_a[1:-1] = (ac[2:] - ac[:-2]) * np.float32(fps / 2.0)
            team_vx_H[s:e] = tv_h
            team_vx_A[s:e] = tv_a

        vis = ball_visible[s:e]
        gap_lens = np.full(n_half, n_half + 1, dtype=np.int32)
        gap_lens[vis] = 0

        not_vis = (~vis).astype(np.int8)
        edges = np.diff(np.concatenate(([0], not_vis, [0])))
        run_starts = np.flatnonzero(edges == 1)
        run_ends = np.flatnonzero(edges == -1)
        for rs, re in zip(run_starts, run_ends):
            if rs > 0 and re < n_half:
                gap_lens[rs:re] = re - rs
        interp_gap_len[s:e] = gap_lens

        vis_idx = np.flatnonzero(vis)
        if len(vis_idx) == 0:
            continue

        bx_raw = ball_xy[s:e, 0]
        by_raw = ball_xy[s:e, 1]
        bx_i = np.full(n_half, np.nan, dtype=np.float32)
        by_i = np.full(n_half, np.nan, dtype=np.float32)
        if len(vis_idx) >= 2:
            grid = np.arange(n_half)
            interp_x = np.interp(grid, vis_idx, bx_raw[vis_idx]).astype(np.float32)
            interp_y = np.interp(grid, vis_idx, by_raw[vis_idx]).astype(np.float32)
            bounded = gap_lens <= max_interp_gap
            bx_i[bounded] = interp_x[bounded]
            by_i[bounded] = interp_y[bounded]
        else:
            bx_i[vis_idx] = bx_raw[vis_idx]
            by_i[vis_idx] = by_raw[vis_idx]

        ball_xy_interp[s:e, 0] = bx_i
        ball_xy_interp[s:e, 1] = by_i

        # Gap-aware central-difference velocity & acceleration inside this half.
        if n_half >= 3:
            vx_h = np.full(n_half, np.nan, dtype=np.float32)
            vy_h = np.full(n_half, np.nan, dtype=np.float32)
            ok_c = ~np.isnan(bx_i[:-2]) & ~np.isnan(bx_i[2:])
            vx_h[1:-1] = np.where(ok_c, (bx_i[2:] - bx_i[:-2]) * np.float32(fps / 2.0), np.nan)
            vy_h[1:-1] = np.where(ok_c, (by_i[2:] - by_i[:-2]) * np.float32(fps / 2.0), np.nan)
            # One-sided step for endpoints adjacent to valid frames
            if not np.isnan(bx_i[0]) and not np.isnan(bx_i[1]):
                vx_h[0] = (bx_i[1] - bx_i[0]) * np.float32(fps)
                vy_h[0] = (by_i[1] - by_i[0]) * np.float32(fps)
            if not np.isnan(bx_i[-1]) and not np.isnan(bx_i[-2]):
                vx_h[-1] = (bx_i[-1] - bx_i[-2]) * np.float32(fps)
                vy_h[-1] = (by_i[-1] - by_i[-2]) * np.float32(fps)

            # Fill isolated 1-frame visible velocities with raw tracker speed when central diff is NaN.
            spd_h = np.hypot(vx_h, vy_h).astype(np.float32)
            raw_fallback = np.isnan(spd_h) & ~np.isnan(ball_speed_raw[s:e])
            spd_h[raw_fallback] = ball_speed_raw[s:e][raw_fallback]

            ax_h = np.full(n_half, np.nan, dtype=np.float32)
            ay_h = np.full(n_half, np.nan, dtype=np.float32)
            dv_h = np.full(n_half, np.nan, dtype=np.float32)
            if n_half >= 5:
                ok_a = ~np.isnan(vx_h[:-4]) & ~np.isnan(vx_h[4:])
                dvx = np.where(ok_a, vx_h[4:] - vx_h[:-4], np.nan)
                dvy = np.where(ok_a, vy_h[4:] - vy_h[:-4], np.nan)
                scale_a = np.float32(fps / 4.0)
                ax_h[2:-2] = dvx * scale_a
                ay_h[2:-2] = dvy * scale_a
                dv_h[2:-2] = np.hypot(dvx, dvy).astype(np.float32)

            ball_vx[s:e] = vx_h
            ball_vy[s:e] = vy_h
            ball_speed[s:e] = spd_h
            ball_ax[s:e] = ax_h
            ball_ay[s:e] = ay_h
            ball_accel[s:e] = np.hypot(ax_h, ay_h).astype(np.float32)
            ball_dv[s:e] = dv_h

    interp_valid = ~np.isnan(ball_xy_interp[:, 0]) & ~np.isnan(ball_xy_interp[:, 1])
    d_H_interp, _, _ = _nearest_team_metrics(players, ball_xy_interp, interp_valid, home_mask)
    d_A_interp, _, _ = _nearest_team_metrics(players, ball_xy_interp, interp_valid, away_mask)

    features = {
        "n_frames": n_frames,
        "fps": fps,
        "pitch": pitch,
        "players": players,
        "team_of_slot": team_of_slot,
        "team_names": atd_data["team_names"],
        "ball_xy": ball_xy,
        "ball_norm_xy": ball_norm_xy,
        "ball_visible": ball_visible,
        "ball_speed_raw": ball_speed_raw,
        "ball_xy_interp": ball_xy_interp,
        "interp_gap_len": interp_gap_len,
        "ball_vx": ball_vx,
        "ball_vy": ball_vy,
        "ball_speed": ball_speed,
        "ball_ax": ball_ax,
        "ball_ay": ball_ay,
        "ball_accel": ball_accel,
        "ball_dv": ball_dv,
        "d_H": d_H,
        "d_A": d_A,
        "d2_H": d2_H,
        "d2_A": d2_A,
        "d_H_interp": d_H_interp,
        "d_A_interp": d_A_interp,
        "nearest_H_speed": nearest_H_speed,
        "nearest_A_speed": nearest_A_speed,
        "n_tracked": n_tracked,
        "mean_speed_all": mean_speed_all,
        "p75_speed_all": p75_speed_all,
        "home_centroid_x": home_centroid_x,
        "away_centroid_x": away_centroid_x,
        "team_vx_H": team_vx_H,
        "team_vx_A": team_vx_A,
        "ball_oob": ball_oob,
        "ball_near_touchline": ball_near_touchline,
        "ball_near_center": ball_near_center,
    }
    features.update(st_data)
    return features


def load_and_extract_features(game=GAME, data_dir=None, use_cache=True, max_interp_gap=75):
    """Load ATD (via .npz cache) + Smart Tagging and compute all per-half features."""
    atd_data = load_or_build_atd_cache(game=game, data_dir=data_dir, use_cache=use_cache)
    n_frames = atd_data["players"].shape[0]
    st_data = load_smart_tagging_features(game=game, data_dir=data_dir, n_frames=n_frames)
    return compute_kinematic_features(atd_data, st_data, game=game, max_interp_gap=max_interp_gap)

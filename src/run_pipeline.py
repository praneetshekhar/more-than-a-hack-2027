"""Deterministic CLI entrypoint for Multi-Pass Anchor & Inflection-Point possession inference."""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks  # noqa: E402
from src import A, D, H, LABEL_STRINGS, X  # noqa: E402
from src.anchors import run_pass0_structural_and_kickoff, run_pass1_high_confidence_anchors  # noqa: E402
from src.features import load_and_extract_features  # noqa: E402
from src.inflection import run_pass2_inflection_resolution  # noqa: E402
from src.local_cv import extract_targeted_cv_cues  # noqa: E402
from src.postprocess import run_pass3_postprocess  # noqa: E402


def write_submission_csv(path, label_codes):
    """Write deterministic frame,label CSV matching SUBMISSION.md contract."""
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)

    chars = LABEL_STRINGS[label_codes]
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, lbl) for frame, lbl in enumerate(chars))
    return path


def count_possession_spells(label_codes, game=GAME, fps=None):
    """Count maximal H/A spells strictly inside half_video_blocks and compute mean duration."""
    fps = float(fps or GAMES[game]["fps"])
    n_spells = 0
    total_alive_frames = 0
    n_frames = len(label_codes)

    for _, first_vf, last_vf in half_video_blocks(game):
        s = max(0, int(first_vf))
        e = min(n_frames, int(last_vf) + 1)
        win = label_codes[s:e]
        if len(win) == 0:
            continue
        breaks = np.flatnonzero(win[1:] != win[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(win)]))
        for i in range(len(bounds) - 1):
            code = int(win[bounds[i]])
            if code in (H, A):
                n_spells += 1
                total_alive_frames += int(bounds[i + 1] - bounds[i])

    mean_dur_s = (total_alive_frames / fps / n_spells) if n_spells > 0 else 0.0
    return n_spells, mean_dur_s


def run_pipeline(
    out_path,
    game=GAME,
    data_dir=None,
    use_cache=True,
    enable_cv=False,
    min_spell_frames=30,
):
    """Execute Pass 0 -> Pass 1 -> Optional Local CV -> Pass 2 -> Pass 3 and save CSV."""
    np.random.seed(0)

    features = load_and_extract_features(game=game, data_dir=data_dir, use_cache=use_cache)
    labels_p0 = run_pass0_structural_and_kickoff(features, game=game)
    labels_p1 = run_pass1_high_confidence_anchors(features, labels_p0, game=game)
    cv_cues = extract_targeted_cv_cues(
        features, labels_p1, game=game, data_dir=data_dir, enabled=enable_cv
    )
    labels_p2 = run_pass2_inflection_resolution(
        features, labels_p1, game=game, cv_cues=cv_cues
    )
    labels_p3 = run_pass3_postprocess(
        features, labels_p2, game=game, min_spell_frames=min_spell_frames
    )

    write_submission_csv(out_path, labels_p3)
    return labels_p3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        help="Output submission CSV path (e.g., evaluation/results/<team_name>.csv)",
    )
    parser.add_argument("--game", default=GAME, help="Game key in games.GAMES (default: BAR-ATH)")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional override for data root (defaults to MTAH_DATA_DIR or ./data)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore and skip writing the .npz ATD cache",
    )
    parser.add_argument(
        "--enable-cv",
        action="store_true",
        help="Enable optional Targeted Local CV on tactical .mp4 (disabled by default for reproducibility)",
    )
    parser.add_argument(
        "--disable-cv",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--min-spell-frames",
        type=int,
        default=30,
        help="Minimum frames for an isolated opposing spell before deflection merge (default: 30)",
    )
    args = parser.parse_args()

    labels = run_pipeline(
        out_path=args.out,
        game=args.game,
        data_dir=args.data_dir,
        use_cache=not args.no_cache,
        enable_cv=bool(args.enable_cv and not args.disable_cv),
        min_spell_frames=args.min_spell_frames,
    )

    n_rows = len(labels)
    counts = {name: int(np.sum(labels == idx)) for idx, name in enumerate(LABEL_STRINGS)}
    n_spells, mean_dur_s = count_possession_spells(labels, game=args.game)
    print(
        "Wrote {} ({} rows): H={} A={} D={} X={}".format(
            args.out, n_rows, counts["H"], counts["A"], counts["D"], counts["X"]
        )
    )
    print("Predicted {} possessions, mean duration {:.1f} s".format(n_spells, mean_dur_s))


if __name__ == "__main__":
    main()


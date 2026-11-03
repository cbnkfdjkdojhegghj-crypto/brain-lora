"""Lazily download and prepare Algonauts data using DataLad.

Validation-only quick test (Friends season 6):
    python -m brain_lora.cli.prepare_data --preset validation --out data

Full training dataset:
    python -m brain_lora.cli.prepare_data --preset full --out data

Raw clips are downloaded, decoded, cached, then dropped one at a time.
Processed cache files and fmri.h5 are retained. Requires DataLad + git-annex.
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from brain_lora.data.algonauts import build_manifests, find_runs, media_key, write_fmri_store
from brain_lora.data.media import build_cache_entry

log = logging.getLogger("prepare_data")
DATASET_URL = "https://github.com/courtois-neuromod/algonauts_2025.competitors.git"


def _run(command, *, cwd=None):
    log.info("$ %s", " ".join(map(str, command)))
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def _ensure_dataset(root):
    if (root / ".datalad").is_dir() and (root / ".git").exists():
        return root
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"{root} exists but is not an installed DataLad dataset.")
    if shutil.which("datalad") is None:
        raise RuntimeError("Install DataLad and git-annex before running this command.")
    root.parent.mkdir(parents=True, exist_ok=True)
    _run(["datalad", "install", "-r", "-s", DATASET_URL, root])
    return root


def _get(root, *paths):
    _run(["datalad", "get", "-J", "4", *paths], cwd=root)


def _drop(root, path):
    _run(["datalad", "drop", str(path.relative_to(root))], cwd=root)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--algonauts-root", default="download/algonauts_2025.competitors")
    ap.add_argument("--out", default="data")
    ap.add_argument("--preset", choices=("validation", "full"), default="full",
                    help="'validation' downloads only Friends S6 for pretrained-model testing.")
    ap.add_argument("--max-clips", type=int, default=None,
                    help="Optional cap on unique clips (use 1-2 for a smoke test).")
    ap.add_argument("--keep-raw", action="store_true",
                    help="Keep downloaded raw videos and source fMRI HDF5 files.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    root = _ensure_dataset(Path(args.algonauts_root).expanduser().resolve())
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    if args.preset == "validation":
        tasks, seasons = ("friends",), (6,)
    else:
        tasks, seasons = ("friends", "movie10"), tuple(range(1, 7))

    # Transcript files are small. Fetch only the relevant fMRI containers:
    # validation uses the Friends HDF5 files, not Movie10.
    _get(root, "stimuli/transcripts")
    fmri_rel = []
    for subject in ("sub-01", "sub-02", "sub-03", "sub-05"):
        suffix = "_desc-s123456_bold.h5" if "friends" in tasks else "_desc-s123456_bold.h5"
        friends = Path("fmri") / subject / "func" / (
            f"{subject}_task-friends_space-MNI152NLin2009cAsym"
            f"_atlas-Schaefer18_parcel-1000Par7Net{suffix}")
        if "friends" in tasks:
            fmri_rel.append(str(friends))
        if "movie10" in tasks:
            movie10 = Path("fmri") / subject / "func" / (
                f"{subject}_task-movie10_space-MNI152NLin2009cAsym"
                "_atlas-Schaefer18_parcel-1000Par7Net_bold.h5")
            fmri_rel.append(str(movie10))
    _get(root, *fmri_rel)

    runs = find_runs(root, tasks=tasks, friends_seasons=seasons)
    if not runs:
        raise RuntimeError("No runs found. Check the DataLad dataset and transcript files.")
    log.info("Found %d subject/run records for preset=%s", len(runs), args.preset)

    # Restrict by unique stimulus, retaining every subject record for each
    # selected clip so all four subject targets remain available.
    if args.max_clips is not None:
        if args.max_clips < 1:
            raise ValueError("--max-clips must be >= 1")
        selected = set(list(dict.fromkeys(media_key(r) for r in runs))[:args.max_clips])
        runs = [r for r in runs if media_key(r) in selected]
        log.info("Limited smoke test to %d unique clips (%d subject/run records)",
                 len(selected), len(runs))

    # A clip may appear in four subject records. Decode each unique clip once.
    # Crucially, raw video files are never accumulated on disk.
    cache = out / "media_cache"
    clips = {}
    for run in runs:
        clips.setdefault(media_key(run), run)

    for i, (key, run) in enumerate(clips.items(), 1):
        video = Path(run["video"])
        transcript = Path(run["transcript"])
        marker = cache / f"{key}_meta.json"
        if not marker.exists():
            _get(root, str(video.relative_to(root)))
            log.info("[%d/%d] decoding %s", i, len(clips), key)
            build_cache_entry(video, transcript, cache / key)
        else:
            log.info("[%d/%d] cached: %s", i, len(clips), key)
        if not args.keep_raw and marker.exists() and video.exists():
            _drop(root, video)

    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train, val = build_manifests(runs, cache)
    train.to_csv(manifest_dir / "train.tsv", sep="\\t", index=False)
    val.to_csv(manifest_dir / "val.tsv", sep="\\t", index=False)

    # Validation preset writes only S6 targets; full preset writes all selected runs.
    fmri_out = out / "fmri.h5"
    write_fmri_store(runs, fmri_out)
    if not args.keep_raw:
        for source in sorted({Path(r["fmri_file"]) for r in runs}):
            if source.exists():
                _drop(root, source)

    log.info("Prepared %s: train=%d windows, val=%d windows",
             out, len(train), len(val))
    if args.preset == "validation":
        log.info("Validation-only dataset ready. No Friends S1-S5 or Movie10 videos downloaded.")


if __name__ == "__main__":
    main()

"""Lazily prepare selected Algonauts splits.

  python -m brain_lora.cli.prepare_data --splits validation
  python -m brain_lora.cli.prepare_data --splits id
  python -m brain_lora.cli.prepare_data --splits ood
  python -m brain_lora.cli.prepare_data --splits validation id ood
  python -m brain_lora.cli.prepare_data --splits train validation id ood

Train = Friends S1-S5 + Movie10; validation = Friends S6;
ID = Friends S7; OOD = six OOD movies. ID/OOD fMRI targets are withheld.
Requires DataLad and git-annex.
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from brain_lora.data.algonauts import (
    build_inference_manifest, build_manifests, find_inference_runs,
    find_runs, media_key, write_fmri_store,
)
from brain_lora.data.media import build_cache_entry

log = logging.getLogger("prepare_data")
DATASET_URL = "https://github.com/courtois-neuromod/algonauts_2025.competitors.git"
VALID_SPLITS = ("train", "validation", "id", "ood")
SUBJECTS = ("sub-01", "sub-02", "sub-03", "sub-05")


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
    if paths:
        _run(["datalad", "get", "-J", "4", *map(str, paths)], cwd=root)


def _drop(root, path):
    if path.exists():
        _run(["datalad", "drop", str(path.relative_to(root))], cwd=root)


def _fetch_fmri(root, tasks):
    paths = []
    for subject in SUBJECTS:
        for task in tasks:
            suffix = "_desc-s123456_bold.h5" if task == "friends" else "_bold.h5"
            paths.append(
                Path("fmri") / subject / "func" /
                (f"{subject}_task-{task}_space-MNI152NLin2009cAsym"
                 f"_atlas-Schaefer18_parcel-1000Par7Net{suffix}")
            )
    _get(root, *paths)


def _prepare_clips(root, runs, cache, *, keep_raw, max_clips=None):
    clips = {}
    for run in runs:
        clips.setdefault(media_key(run), run)
    if max_clips is not None:
        if max_clips < 1:
            raise ValueError("--max-clips must be >= 1")
        clips = dict(list(clips.items())[:max_clips])

    log.info("Preparing %d unique clips", len(clips))
    for i, (key, run) in enumerate(clips.items(), 1):
        video, transcript = Path(run["video"]), run.get("transcript")
        marker = cache / f"{key}_meta.json"
        if not marker.exists():
            _get(root, video.relative_to(root))
            log.info("[%d/%d] decoding %s", i, len(clips), key)
            build_cache_entry(video, transcript, cache / Path(key).parent)
        else:
            log.info("[%d/%d] cache exists: %s", i, len(clips), key)
        if not keep_raw and marker.exists():
            _drop(root, video)
    return set(clips)


def _write_manifest(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, sep="\\t")
        frame = pd.concat([old, frame], ignore_index=True)
        frame = frame.drop_duplicates("segment_id", keep="last")
    frame.to_csv(path, sep="\\t", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--algonauts-root", default="download/algonauts_2025.competitors")
    ap.add_argument("--out", default="data")
    ap.add_argument("--splits", nargs="+", choices=VALID_SPLITS,
                    default=["train", "validation"])
    ap.add_argument("--max-clips", type=int, default=None,
                    help="Limit each selected split to N unique clips (smoke tests).")
    ap.add_argument("--keep-raw", action="store_true",
                    help="Do not drop downloaded source videos/fMRI files.")
    args = ap.parse_args()

    if len(set(args.splits)) != len(args.splits):
        ap.error("--splits contains duplicates")
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    root = _ensure_dataset(Path(args.algonauts_root).expanduser().resolve())
    out = Path(args.out).expanduser().resolve()
    cache, manifest_dir = out / "media_cache", out / "manifests"
    cache.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    supervised = [s for s in args.splits if s in ("train", "validation")]
    if supervised:
        seasons = tuple(range(1, 6)) if "train" in supervised else ()
        if "validation" in supervised:
            seasons = (*seasons, 6)
        tasks = ["friends"]
        if "train" in supervised:
            tasks.append("movie10")

        # Fetch only transcript folders required by these splits.
        transcript_paths = [
            Path("stimuli/transcripts/friends") / f"s{season}"
            for season in seasons
        ]
        if "train" in supervised:
            transcript_paths.append(Path("stimuli/transcripts/movie10"))
        _get(root, *transcript_paths)
        _fetch_fmri(root, tasks)

        runs = find_runs(root, tasks=tuple(tasks), friends_seasons=seasons)
        if not runs:
            raise RuntimeError("No supervised runs discovered; check DataLad content.")
        selected = set()
        if "train" in supervised:
            train_runs = [r for r in runs
                          if r["task"] == "movie10" or r["season"] != 6]
            selected |= _prepare_clips(
                root, train_runs, cache, keep_raw=args.keep_raw,
                max_clips=args.max_clips)
        if "validation" in supervised:
            val_runs = [r for r in runs if r["season"] == 6]
            selected |= _prepare_clips(
                root, val_runs, cache, keep_raw=args.keep_raw,
                max_clips=args.max_clips)
        runs = [r for r in runs if media_key(r) in selected]

        train_df, val_df = build_manifests(runs, cache)
        if "train" in supervised:
            _write_manifest(manifest_dir / "train.tsv", train_df)
        if "validation" in supervised:
            _write_manifest(manifest_dir / "val.tsv", val_df)
            _write_manifest(manifest_dir / "validation.tsv", val_df)

        # Append targets so train and validation can be prepared independently.
        write_fmri_store(runs, out / "fmri.h5")
        if not args.keep_raw:
            for source in sorted({Path(r["fmri_file"]) for r in runs}):
                _drop(root, source)

    # Target-free inference splits never fetch fMRI response containers.
    for split in (s for s in args.splits if s in ("id", "ood")):
        if split == "id":
            _get(root, "stimuli/transcripts/friends/s7",
                 "stimuli/movies/friends/s7")
        else:
            _get(root, "stimuli/transcripts/ood", "stimuli/movies/ood")
        runs = find_inference_runs(root, split)
        selected = _prepare_clips(root, runs, cache, keep_raw=args.keep_raw,
                                  max_clips=args.max_clips)
        runs = [r for r in runs if media_key(r) in selected]
        frame = build_inference_manifest(runs, cache, split)
        _write_manifest(manifest_dir / f"{split}.tsv", frame)

    log.info("Preparation complete: %s", out)
    log.info("Available manifests: %s",
             ", ".join(p.name for p in sorted(manifest_dir.glob("*.tsv"))))


if __name__ == "__main__":
    main()

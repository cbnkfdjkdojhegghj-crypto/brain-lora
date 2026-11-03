"""Algonauts 2025: run discovery, window manifests and the fMRI target store.

Expected layout of the challenge download (``data_root``)::

    fmri/sub-0X/func/sub-0X_task-{friends,movie10}_..._bold.h5
    stimuli/movies/friends/s{N}/friends_s{NN}e{EE}{p}.mkv
    stimuli/movies/movie10/{movie}/{movie}{NN}.mkv
    stimuli/transcripts/friends/s{N}/friends_s{NN}e{EE}{p}.tsv
    stimuli/transcripts/movie10/{movie}/movie10_{movie}{NN}.tsv

Friends seasons 1-5 and Movie10 are the training set; season 6 is validation.
Each (subject, run) is cut into 149 s windows (see ``brain_lora.constants``);
one manifest row is one window of one subject's run.
"""

import ast
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from brain_lora.constants import SUBJECTS, TR, window_starts

log = logging.getLogger(__name__)

FRIENDS_SEASONS = range(1, 7)
VAL_SEASON = 6
FRIENDS_EPISODES = {1: 24, 2: 24, 3: 25, 4: 24, 5: 24, 6: 25}
FRIENDS_SKIP = {(5, 20, "a"), (4, 1, "a"), (6, 3, "a"), (4, 13, "b"), (4, 1, "b")}
MOVIE10_CHUNKS = {"bourne": 10, "wolf": 17, "life": 5, "figures": 12}
MOVIE10_RUNS = {"bourne": [0], "wolf": [0], "life": [1, 2], "figures": [1, 2]}

MANIFEST_COLUMNS = ["subject", "subject_idx", "task", "stimulus_id",
                    "fmri_item_id", "media", "split", "segment_idx",
                    "segment_id", "segment_start_s"]


def _fmri_file(root, subject, task):
    suffix = "_desc-s123456_bold.h5" if task == "friends" else "_bold.h5"
    return (root / "fmri" / subject / "func" /
            (f"{subject}_task-{task}_space-MNI152NLin2009cAsym"
             f"_atlas-Schaefer18_parcel-1000Par7Net{suffix}"))


def _find_key(keys, match):
    found = [k for k in keys if match in k]
    if len(found) != 1:
        raise KeyError(f"expected one fMRI run matching {match!r}, found {found}")
    return found[0]


def find_runs(data_root, *, tasks=("friends", "movie10"), friends_seasons=FRIENDS_SEASONS):
    """One record per selected run; supports validation-only discovery."""
    root = Path(data_root)
    runs = []
    for subject in SUBJECTS:
        keys = {}
        for task in tasks:
            with h5py.File(_fmri_file(root, subject, task), "r") as f:
                keys[task] = list(f.keys())

        for season in (friends_seasons if "friends" in tasks else ()):
            for episode in range(1, FRIENDS_EPISODES[season] + 1):
                for part in "abcd":
                    if (season, episode, part) in FRIENDS_SKIP:
                        continue
                    name = f"friends_s{season:02d}e{episode:02d}{part}"
                    transcript = (root / "stimuli/transcripts/friends" /
                                  f"s{season}" / f"{name}.tsv")
                    if not transcript.exists():
                        continue
                    runs.append(dict(
                        subject=subject, task="friends", season=season,
                        stimulus_id=f"friends/s{season:02d}e{episode:02d}{part}",
                        fmri_file=_fmri_file(root, subject, "friends"),
                        fmri_key=_find_key(keys["friends"],
                                           f"s{season:02d}e{episode:02d}{part}"),
                        video=root / "stimuli/movies/friends" / f"s{season}" / f"{name}.mkv",
                        transcript=transcript))

        for movie, n_chunks in (MOVIE10_CHUNKS.items() if "movie10" in tasks else ()):
            for chunk in range(1, n_chunks + 1):
                transcript = (root / "stimuli/transcripts/movie10" / movie /
                              f"movie10_{movie}{chunk:02d}.tsv")
                if not transcript.exists():
                    continue
                for run in MOVIE10_RUNS[movie]:
                    run_key = f"_run-{run}" if run else ""
                    runs.append(dict(
                        subject=subject, task="movie10", season=None,
                        stimulus_id=f"movie10/{movie}{chunk:02d}"
                                    + (f"_run{run}" if run else ""),
                        fmri_file=_fmri_file(root, subject, "movie10"),
                        fmri_key=_find_key(keys["movie10"],
                                           f"{movie}{chunk:02d}{run_key}"),
                        video=root / "stimuli/movies/movie10" / movie /
                        f"{movie}{chunk:02d}.mkv",
                        transcript=transcript))
    return runs


OOD_MOVIES = ("chaplin", "mononoke", "passepartout", "planetearth", "pulpfiction", "wot")


def find_inference_runs(data_root, split):
    """Discover stimulus-only runs for Friends S7 (ID) or the six OOD movies."""
    root = Path(data_root)
    runs = []
    if split == "id":
        transcript_files = sorted((root / "stimuli/transcripts/friends/s7").glob("*.tsv"))
        for transcript in transcript_files:
            stem = transcript.stem
            video = root / "stimuli/movies/friends/s7" / f"{stem}.mkv"
            if not video.exists():
                raise FileNotFoundError(f"Missing ID stimulus video: {video}")
            runs.append(dict(task="friends", season=7,
                             stimulus_id=f"friends/s7/{stem}",
                             video=video, transcript=transcript))
    elif split == "ood":
        for movie in OOD_MOVIES:
            for chunk in (1, 2):
                stem = f"task-{movie}{chunk}"
                video = root / "stimuli/movies/ood" / movie / f"{stem}_video.mkv"
                transcript = root / "stimuli/transcripts/ood" / movie / f"ood_{movie}{chunk}.tsv"
                if not video.exists():
                    raise FileNotFoundError(f"Missing OOD stimulus video: {video}")
                # Chaplin is silent and intentionally has no transcript.
                runs.append(dict(task="ood", season=None,
                                 stimulus_id=f"ood/{movie}{chunk}",
                                 video=video,
                                 transcript=transcript if transcript.exists() else None))
    else:
        raise ValueError(f"Unknown inference split: {split}")
    return runs


def build_inference_manifest(runs, media_cache, split):
    """Build target-free windows. Every window has all four subject indices."""
    from brain_lora.data.media import cache_meta
    rows = []
    for run in runs:
        media = f"{run['task']}/{Path(run['video']).stem}"
        duration = float(cache_meta(media_cache, media)["duration_s"])
        if run["transcript"] is not None:
            duration = max(duration, transcript_duration(run["transcript"]))
        for seg_idx, start in enumerate(window_starts(duration)):
            for subject_idx, subject in enumerate(SUBJECTS):
                item = f"{subject}/{run['stimulus_id']}"
                rows.append(dict(
                    subject=subject, subject_idx=subject_idx,
                    task=run["task"], stimulus_id=run["stimulus_id"],
                    fmri_item_id=item, media=media, split=split,
                    segment_idx=seg_idx,
                    segment_id=f"{item}@seg{seg_idx:03d}",
                    segment_start_s=float(start)))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def media_key(run):
    """Cache key of a run's stimulus: ``"<task>/<video stem>"``."""
    return f"{run['task']}/{Path(run['video']).stem}"


def transcript_duration(path):
    """Duration implied by a transcript: its TR count, or its last word end."""
    df = pd.read_csv(path, sep="\t")
    total = len(df) * TR
    for _, row in df.iterrows():
        try:
            onsets = ast.literal_eval(row.get("onsets_per_tr", "[]"))
            durs = ast.literal_eval(row.get("durations_per_tr", "[]"))
        except (ValueError, SyntaxError):
            continue
        for o, d in zip(onsets, durs):
            total = max(total, float(o) + max(float(d), 0.0))
    return float(total)


def build_manifests(runs, media_cache):
    """-> (train_df, val_df), one row per (subject, run, window).

    A run is windowed up to the longest of its transcript, BOLD and video
    durations; the video duration is read from the media cache.
    """
    from brain_lora.data.media import cache_meta

    durations = {}
    rows = []
    for run in runs:
        with h5py.File(run["fmri_file"], "r") as f:
            fmri_s = f[run["fmri_key"]].shape[0] * TR
        media = media_key(run)
        if media not in durations:
            durations[media] = (transcript_duration(run["transcript"]),
                                float(cache_meta(media_cache, media)["duration_s"]))
        stop = max(fmri_s, *durations[media])
        item = f"{run['subject']}/{run['stimulus_id']}"
        split = "val" if run["season"] == VAL_SEASON else "train"
        for k, start in enumerate(window_starts(stop)):
            rows.append(dict(
                subject=run["subject"],
                subject_idx=SUBJECTS.index(run["subject"]),
                task=run["task"], stimulus_id=run["stimulus_id"],
                fmri_item_id=item, media=media, split=split,
                segment_idx=k, segment_id=f"{item}@seg{k:03d}",
                segment_start_s=float(start)))
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    log.info("manifests: %d train rows, %d val rows", len(train), len(val))
    return train, val


def write_fmri_store(runs, path):
    """Z-score every BOLD run per parcel (sample std) and store it as float16
    under its ``<subject>/<stimulus_id>`` key."""
    with h5py.File(path, "w") as out:
        for run in runs:
            with h5py.File(run["fmri_file"], "r") as f:
                x = f[run["fmri_key"]][:].astype(np.float32)
            sd = x.std(axis=0, ddof=1, keepdims=True)
            x = (x - x.mean(axis=0, keepdims=True)) / np.where(sd < 1e-8, 1.0, sd)
            out.create_dataset(f"{run['subject']}/{run['stimulus_id']}",
                               data=x.astype(np.float16))


def load_manifests(manifest_dir):
    d = Path(manifest_dir)
    return (pd.read_csv(d / "train.tsv", sep="\t"),
            pd.read_csv(d / "val.tsv", sep="\t"))

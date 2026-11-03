import json

from brain_lora.constants import SUBJECTS
from brain_lora.data.algonauts import build_inference_manifest


def test_inference_manifest_has_all_subjects_and_no_targets(tmp_path):
    cache = tmp_path / "media_cache"
    clip_dir = cache / "ood"
    clip_dir.mkdir(parents=True)
    (clip_dir / "task-chaplin1_video_meta.json").write_text(
        json.dumps({"duration_s": 80.0})
    )
    runs = [{
        "task": "ood",
        "stimulus_id": "ood/chaplin1",
        "video": tmp_path / "task-chaplin1_video.mkv",
        "transcript": None,
    }]

    frame = build_inference_manifest(runs, cache, "ood")

    assert set(frame["subject"]) == set(SUBJECTS)
    assert frame["subject_idx"].tolist() == list(range(len(SUBJECTS)))
    assert frame["split"].unique().tolist() == ["ood"]
    assert frame["fmri_item_id"].notna().all()
    assert frame["media"].unique().tolist() == ["ood/task-chaplin1_video"]
    assert len(frame) == len(SUBJECTS)

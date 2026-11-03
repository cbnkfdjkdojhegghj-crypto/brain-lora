"""Generate predictions for target-withheld ID/OOD stimulus manifests.

Example:
  python -m brain_lora.cli.predict runs/lora_linear --split id
  python -m brain_lora.cli.predict runs/lora_linear --split ood --max-items 2

Writes predictions with shape (windows, subjects, 100 TRs, 1000 parcels).
No withheld fMRI targets are accessed or required.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from brain_lora.data.dataset import WindowDataset, make_loader
from brain_lora.model.readouts import READOUTS
from brain_lora.training import Config, build_trainer, data_paths


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--split", choices=("id", "ood"), required=True)
    ap.add_argument("--checkpoint", default="refit_last.pt")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")

    run_dir = Path(args.run_dir)
    cfg = Config.load(run_dir)
    paths = data_paths(cfg.data_dir)
    manifest = paths["manifests"] / f"{args.split}.tsv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found. Run prepare_data with --splits {args.split} first."
        )

    import pandas as pd
    from brain_lora.constants import SUBJECTS

    frame = pd.read_csv(manifest, sep="\\t")
    ds = WindowDataset(
        frame, paths["media_cache"], None,
        context_lead_s=cfg.context_lead_s, chunk_s=cfg.chunk_s,
        use_audio=cfg.use_audio, tokens=READOUTS[cfg.readout].NEEDS_TOKENS,
    )
    trainer = build_trainer(cfg, args.device)
    trainer.load(run_dir / args.checkpoint)
    trainer._set_mode(False)
    loader = make_loader(
        ds, shuffle=False, num_workers=cfg.num_workers if args.num_workers is None
        else args.num_workers,
    )

    predictions, segment_ids, stimulus_ids = [], [], []
    with torch.inference_mode():
        for i, item in enumerate(loader):
            if args.max_items is not None and i >= args.max_items:
                break
            pred = trainer.forward(item)
            predictions.append(pred.float().cpu().numpy())
            segment_ids.append(item["segment_id"])
            stimulus_ids.append(item["stimulus_id"])

    if not predictions:
        raise RuntimeError(f"No windows found in {manifest}")
    pred = np.stack(predictions).astype(np.float16)
    out = Path(args.out) if args.out else run_dir / f"predictions_{args.split}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, predictions=pred,
        segment_ids=np.asarray(segment_ids, dtype=str),
        stimulus_ids=np.asarray(stimulus_ids, dtype=str),
        subjects=np.asarray(SUBJECTS, dtype=str),
    )
    print(f"saved {len(pred)} windows x {len(SUBJECTS)} subjects to {out}")


if __name__ == "__main__":
    main()

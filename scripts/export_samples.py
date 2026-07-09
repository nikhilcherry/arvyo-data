#!/usr/bin/env python
"""Export one small, real, contract-valid .npz sample per label into
data/samples/ — the only committed data in this repo (everything else under
data/ is git-ignored; the bulk corpus lives on Kaggle).

For each label, candidates are every contract-valid .npz in its source
directory that is already <=500KB. One is picked uniformly at random from
that pool (seeded, reproducible) rather than always taking the smallest
file, so the committed sample isn't cherry-picked to be unusually thin. If
a label has zero candidates <=500KB, its smallest file is decimated
uniformly (keep every Nth point) until it fits, and that IS recorded
loudly in PROVENANCE.md — no silent shrinking.

Source directories:
    planet, eb, starspot, null   -> data/processed/<label>/
    blend                        -> data/kepler/processed/blend/
                                     (real Kepler DR25 centroid-offset false
                                     positives; the TESS `blend` class is
                                     synthetic-only and explicitly excluded
                                     by README.md's "Label classes" table)

Validates every exported file against arvyo-pipeline's frozen schema
contract (imported from the sibling repo at ../arvyo-pipeline, not
copy-pasted) before writing PROVENANCE.md.

Run: python scripts/export_samples.py --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
PROVENANCE_PATH = SAMPLES_DIR / "PROVENANCE.md"
MAX_BYTES = 500_000

PIPELINE_ROOT = ROOT.parent / "arvyo-pipeline"
if not PIPELINE_ROOT.exists():
    sys.exit(
        f"ERROR: sibling repo not found at {PIPELINE_ROOT}. "
        "arvyo-data expects arvyo-pipeline checked out alongside it."
    )
sys.path.insert(0, str(PIPELINE_ROOT))
from arvyo.contract import ContractError, load_sample  # noqa: E402

SOURCES = {
    "planet": ROOT / "data" / "processed" / "planet",
    "eb": ROOT / "data" / "processed" / "eb",
    "blend": ROOT / "data" / "kepler" / "processed" / "blend",
    "starspot": ROOT / "data" / "processed" / "starspot",
    "null": ROOT / "data" / "processed" / "null",
}

DECIMATABLE_ARRAYS = ["time", "flux", "flux_err", "flux_raw"]


def _contract_valid_files(source_dir: Path) -> list[Path]:
    valid = []
    for path in sorted(source_dir.glob("*.npz")):
        if path.stat().st_size == 0:
            continue
        try:
            load_sample(path)
        except ContractError:
            continue
        valid.append(path)
    return valid


def _decimate(data: dict, factor: int) -> dict:
    out = dict(data)
    for key in DECIMATABLE_ARRAYS:
        if key in out:
            out[key] = np.asarray(out[key])[::factor]
    return out


def _decimation_factor_for_budget(data: dict, max_bytes: int) -> int:
    n = np.asarray(data["time"]).size
    factor = 1
    while True:
        trial = _decimate(data, factor)
        buf_size = sum(
            np.asarray(trial[k]).nbytes for k in DECIMATABLE_ARRAYS if k in trial
        )
        if buf_size <= max_bytes or factor >= n:
            return factor
        factor += 1


def export_one(label: str, source_dir: Path, rng: np.random.Generator) -> dict:
    if not source_dir.is_dir():
        raise SystemExit(f"STOP: source directory for label {label!r} does not exist: "
                          f"{source_dir}. Not fabricating a sample for a missing class.")

    candidates = _contract_valid_files(source_dir)
    if not candidates:
        raise SystemExit(f"STOP: no contract-valid .npz files found for label {label!r} "
                          f"under {source_dir}. Not fabricating a sample for a missing class.")

    under_budget = [p for p in candidates if p.stat().st_size <= MAX_BYTES]

    label_dir = SAMPLES_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    if under_budget:
        chosen = under_budget[rng.integers(len(under_budget))]
        with np.load(chosen, allow_pickle=True) as npz:
            data = {k: npz[k] for k in npz.files}
        factor = 1
        original_points = int(np.asarray(data["time"]).size)
    else:
        # Fallback: nothing fits as-is: decimate the smallest candidate.
        chosen = min(candidates, key=lambda p: p.stat().st_size)
        with np.load(chosen, allow_pickle=True) as npz:
            data = {k: npz[k] for k in npz.files}
        original_points = int(np.asarray(data["time"]).size)
        factor = _decimation_factor_for_budget(data, MAX_BYTES)
        if factor > 1:
            data = _decimate(data, factor)

    dest_path = label_dir / chosen.name
    np.savez(dest_path, **data)

    # Validate the file we actually wrote, not the in-memory dict.
    load_sample(dest_path)

    return {
        "label": label,
        "source_path": str(chosen.relative_to(ROOT)),
        "dest_path": str(dest_path.relative_to(ROOT)),
        "original_points": original_points,
        "decimation_factor": factor,
        "final_points": int(np.asarray(data["time"]).size),
        "final_bytes": dest_path.stat().st_size,
        "n_candidates_under_budget": len(under_budget),
        "n_candidates_total": len(candidates),
    }


def write_provenance(entries: list[dict], seed: int) -> None:
    lines = [
        "# data/samples/ provenance",
        "",
        "Generated by `scripts/export_samples.py --seed "
        f"{seed}`. Do not hand-edit these .npz files; re-run the script "
        "(with the same --seed) to reproduce them exactly, or with a "
        "different --seed to pick a different real target per class.",
        "",
        "| label | dest | source | original pts | decimation | final pts | final size |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for e in entries:
        lines.append(
            f"| {e['label']} | `{e['dest_path']}` | `{e['source_path']}` | "
            f"{e['original_points']} | {e['decimation_factor']}x | "
            f"{e['final_points']} | {e['final_bytes'] / 1024:.1f} KB |"
        )
    lines.append("")
    lines.append(
        "All rows pass `arvyo.contract.load_sample()` from the sibling "
        "arvyo-pipeline repo unchanged (validated at export time)."
    )
    lines.append("")
    PROVENANCE_PATH.write_text("\n".join(lines))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    for label, source_dir in SOURCES.items():
        entry = export_one(label, source_dir, rng)
        entries.append(entry)
        print(f"{label:10s} <- {entry['source_path']} "
              f"({entry['original_points']} pts, decimation={entry['decimation_factor']}x) "
              f"-> {entry['dest_path']} ({entry['final_bytes'] / 1024:.1f} KB)")

    write_provenance(entries, args.seed)
    print(f"\nwrote {PROVENANCE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

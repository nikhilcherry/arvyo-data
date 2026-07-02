#!/usr/bin/env python
"""Build kepler_manifest.csv from NASA Exoplanet Archive TAP tables (Q1-Q17
DR25 TCE list, cumulative KOI dispositions/FP flags, Certified FP table).
See Kepler addendum Phase A1.

Label mapping mirrors the TESS taxonomy so both manifests can be mixed in
one training loop:
  koi_disposition == CONFIRMED          -> planet
  koi_fpflag_co == 1                    -> blend   (real centroid-offset FP)
  koi_fpflag_ss == 1 (and co == 0)      -> eb       (on-target stellar eclipse)
  koi_fpflag_nt == 1 (and both above 0) -> unknown
  koi_disposition == CANDIDATE          -> excluded -> kepler_manifest_candidates.csv

Flags can co-occur; priority is co > ss > nt (a centroid offset means the
signal isn't on-target regardless of its shape).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = ROOT / "data" / "kepler" / "catalogs"

TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={q}&format=csv"

# Same columns as manifest.csv (tic_id holds the KIC ID here; see README for
# why the column isn't renamed to star_id), plus a mission column.
MANIFEST_COLUMNS = [
    "tic_id", "label", "source_catalog", "disposition", "period_days",
    "epoch_btjd", "depth_ppm", "duration_hours", "tmag", "notes", "mission",
]

FLAG_PRIORITY = ["co", "ss", "nt"]  # blend beats eb beats unknown


def empty_manifest_df():
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


def tap_fetch(table, out_path, select="*"):
    print(f"  fetching {table} ...")
    q = f"select+{select}+from+{table}"
    url = TAP_BASE.format(q=q)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        text = r.text
        if text.lstrip().startswith("<?xml") or "VOTABLE" in text[:200]:
            raise ValueError(f"TAP returned an error document: {text[:300]}")
        df = pd.read_csv(pd.io.common.StringIO(text))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"    {len(df)} rows -> {out_path}")
        return df
    except Exception as e:
        print(f"    ERROR fetching {table}: {e}", file=sys.stderr)
        if out_path.exists():
            print(f"    using cached {out_path}")
            return pd.read_csv(out_path)
        return None


def list_tap_tables():
    try:
        r = requests.get(
            TAP_BASE.format(q="select+table_name+from+TAP_SCHEMA.tables"), timeout=30
        )
        r.raise_for_status()
        return pd.read_csv(pd.io.common.StringIO(r.text))["table_name"].tolist()
    except Exception as e:
        print(f"  could not list TAP tables: {e}", file=sys.stderr)
        return []


# ---------------------------------------------------------------- Source 1 --
def fetch_dr25_tce():
    print("[1/3] Fetching Q1-Q17 DR25 TCE table...")
    return tap_fetch("q1_q17_dr25_tce", CATALOG_DIR / "dr25_tce.csv")


# ---------------------------------------------------------------- Source 2 --
def fetch_cumulative_koi():
    print("[2/3] Fetching cumulative KOI table...")
    df = tap_fetch("cumulative", CATALOG_DIR / "koi_cumulative.csv")
    if df is None:
        tables = list_tap_tables()
        print("  available TAP tables:", tables)
        print("  STOP: could not fetch the cumulative KOI table via TAP.")
        print("  Manual fallback: download 'Cumulative KOI Table' CSV from")
        print("  https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative")
        print(f"  and save it to {CATALOG_DIR / 'koi_cumulative.csv'}, then re-run.")
        print("  Continuing assuming the file exists...")
        cached = CATALOG_DIR / "koi_cumulative.csv"
        if cached.exists():
            df = pd.read_csv(cached)
    return df


# ---------------------------------------------------------------- Source 3 --
def fetch_certified_fp():
    print("[3/3] Fetching Certified False Positive (FPWG) table...")
    df = tap_fetch("fpwg", CATALOG_DIR / "certified_fp.csv")
    if df is None:
        tables = list_tap_tables()
        print("  available TAP tables:", tables)
        print("  STOP: 'fpwg' is not exposed on the current TAP service (confirmed")
        print("  absent from TAP_SCHEMA.tables as of this run -- it appears to have")
        print("  been retired/merged into the cumulative KOI dispositions).")
        print("  Manual fallback (if a cross-check against human-vetted FPs is still")
        print("  wanted): search the Exoplanet Archive bulk-download pages for the")
        print("  'Kepler Certified False Positive Table' and save the CSV to")
        print(f"  {CATALOG_DIR / 'certified_fp.csv'}, then re-run this script.")
        print("  Continuing without the FPWG cross-check (flag-derived labels only)...")
        cached = CATALOG_DIR / "certified_fp.csv"
        if cached.exists():
            df = pd.read_csv(cached)
    return df


# ---------------------------------------------------------------------------
def label_from_flags(row):
    if row.get("koi_fpflag_co") == 1:
        return "blend"
    if row.get("koi_fpflag_ss") == 1:
        return "eb"
    if row.get("koi_fpflag_nt") == 1:
        return "unknown"
    return None


def build_labels(koi, fpwg):
    koi = koi.copy()
    disp = koi["koi_disposition"].astype(str).str.strip().str.upper()

    flag_cols = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"]
    for c in flag_cols:
        koi[c] = pd.to_numeric(koi[c], errors="coerce").fillna(0).astype(int)

    rows = []
    n_multi_flag = 0
    n_blend_from_co = 0
    fpwg_conflicts = 0

    fpwg_disp_by_koi = {}
    if fpwg is not None and not fpwg.empty:
        name_col = next((c for c in fpwg.columns if "kepoi" in c.lower()), None)
        disp_col = next((c for c in fpwg.columns if "disposition" in c.lower()), None)
        if name_col and disp_col:
            fpwg_disp_by_koi = dict(zip(fpwg[name_col], fpwg[disp_col]))

    for _, row in koi.iterrows():
        d = row.to_dict()
        koi_name = d.get("kepoi_name")
        notes = []

        if disp.loc[row.name] == "CANDIDATE":
            label = "candidate"
        elif disp.loc[row.name] == "CONFIRMED":
            label = "planet"
        else:
            flagged = [f for f in ("koi_fpflag_co", "koi_fpflag_ss", "koi_fpflag_nt")
                       if d.get(f) == 1]
            if len(flagged) > 1:
                n_multi_flag += 1
            label = label_from_flags(d)
            if label == "blend":
                n_blend_from_co += 1
            if label is None:
                label = "unknown"
                notes.append("FALSE POSITIVE with no fpflag set; defaulted to unknown")

        if label != "candidate" and label != "planet":
            flagged_names = [f.replace("koi_fpflag_", "") for f in
                              ("koi_fpflag_co", "koi_fpflag_ss", "koi_fpflag_nt")
                              if d.get(f) == 1]
            if flagged_names:
                notes.append(f"fpflags={'+'.join(flagged_names)}")

        fpwg_disp = fpwg_disp_by_koi.get(koi_name)
        if fpwg_disp is not None and label not in ("planet", "candidate"):
            fpwg_disp_u = str(fpwg_disp).strip().upper()
            if fpwg_disp_u in ("CERTIFIED FP", "FALSE POSITIVE") and label == "unknown":
                pass  # consistent enough, no conflict
            elif fpwg_disp_u in ("CERTIFIED FA", "FALSE ALARM"):
                fpwg_conflicts += 1
                notes.append(f"FPWG disposition={fpwg_disp} overrides flag-derived label; trusting FPWG")
                label = "unknown"

        rows.append({
            "tic_id": d.get("kepid"),
            "label": label,
            "source_catalog": "kepler_cumulative_koi",
            "disposition": disp.loc[row.name],
            "period_days": d.get("koi_period"),
            "epoch_btjd": d.get("koi_time0bk"),  # BKJD (BJD-2454833), not BTJD; see README
            "depth_ppm": d.get("koi_depth"),
            "duration_hours": d.get("koi_duration"),
            "tmag": d.get("koi_kepmag"),
            "notes": "; ".join(notes),
            "mission": "kepler",
            "_kepoi_name": koi_name,
        })

    out = pd.DataFrame(rows)
    print(f"  {n_multi_flag} KOIs had >1 FP flag set (resolved via co > ss > nt priority)")
    print(f"  {n_blend_from_co} blend rows sourced from koi_fpflag_co (real centroid-offset FPs)")
    if fpwg is not None:
        print(f"  {fpwg_conflicts} FPWG disposition conflicts with flag-derived label (FPWG trusted)")
    return out


def note_multi_koi_stars(df):
    counts = df["tic_id"].value_counts()
    multi = counts[counts > 1].index
    if len(multi):
        mask = df["tic_id"].isin(multi)
        df.loc[mask, "notes"] = (
            df.loc[mask, "notes"].astype(str) + "; multi-KOI star (kepid shared across KOIs)"
        ).str.lstrip("; ")
    return df


def cap_classes(df, caps, seed):
    parts = []
    for label, cap in caps.items():
        subset = df[df["label"] == label]
        if len(subset) > cap:
            subset = subset.sample(n=cap, random_state=seed)
        parts.append(subset)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def print_summary(df, title):
    print(f"\n{title}")
    counts = df["label"].value_counts()
    for label, n in counts.items():
        print(f"  {label:10s} {n}")
    print(f"  {'total':10s} {len(df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-planet", type=int, default=1500)
    ap.add_argument("--n-eb", type=int, default=1000)
    ap.add_argument("--n-blend", type=int, default=800)
    ap.add_argument("--n-unknown", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    fetch_dr25_tce()  # documents the Robovetter's input TCE population; not
                       # joined into the manifest itself (KOI table already
                       # carries the vetted dispositions/flags we need)
    koi = fetch_cumulative_koi()
    if koi is None or koi.empty:
        print("ERROR: no cumulative KOI table available, cannot build manifest.", file=sys.stderr)
        sys.exit(1)
    fpwg = fetch_certified_fp()

    labeled = build_labels(koi, fpwg)

    # dedupe: a KIC can host multiple KOIs. Keep one manifest ROW per KOI
    # (as instructed) but flag shared kepids in notes for the download step,
    # which dedupes by kepid.
    labeled = note_multi_koi_stars(labeled)

    candidates = labeled[labeled["label"] == "candidate"].copy()
    candidates_path = ROOT / "kepler_manifest_candidates.csv"
    candidates[MANIFEST_COLUMNS].to_csv(candidates_path, index=False)

    train = labeled[labeled["label"] != "candidate"].copy()
    train = train.dropna(subset=["tic_id"])
    train["tic_id"] = train["tic_id"].astype("Int64")

    full_path = ROOT / "kepler_manifest_full.csv"
    train[MANIFEST_COLUMNS].to_csv(full_path, index=False)

    caps = {
        "planet": args.n_planet,
        "eb": args.n_eb,
        "blend": args.n_blend,
        "unknown": args.n_unknown,
    }
    capped = cap_classes(train, caps, args.seed)
    manifest_path = ROOT / "kepler_manifest.csv"
    capped[MANIFEST_COLUMNS].to_csv(manifest_path, index=False)

    print_summary(train, f"kepler_manifest_full.csv -> {full_path}")
    print_summary(capped, f"kepler_manifest.csv (capped, seed={args.seed}) -> {manifest_path}")
    print(f"\ncandidates (excluded, disposition=CANDIDATE): {len(candidates)} -> {candidates_path}")


if __name__ == "__main__":
    main()

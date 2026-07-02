#!/usr/bin/env python
"""Build manifest.csv (and manifest_full.csv / manifest_candidates.csv) from
ExoFOP TOI, TESS-EB, a TESS rotation-period catalog, and a null-by-exclusion
sample. See CLAUDE_CODE_DATA_HANDOUT.md Phase 1 for the full spec.
"""
import argparse
import io
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = ROOT / "data" / "catalogs"

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=pipe"
TESS_EB_VILLANOVA_CANDIDATES = [
    "https://tessebs.villanova.edu/catalog.csv",
    "https://tessebs.villanova.edu/media/tesseb_catalog.csv",
]
TESS_EB_MAST_URL = (
    "https://archive.stsci.edu/hlsps/tess-ebs/"
    "hlsp_tess-ebs_tess_lcf-ffi_s0001-s0026_tess_v1.0_cat.csv"
)

MANIFEST_COLUMNS = [
    "tic_id", "label", "source_catalog", "disposition", "period_days",
    "epoch_btjd", "depth_ppm", "duration_hours", "tmag", "notes",
]

LABEL_PRIORITY = {"planet": 0, "eb": 1, "starspot": 2, "unknown": 3, "null": 4}


def normalize_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_").replace(".", "").replace("/", "_")
        for c in df.columns
    ]
    return df


def find_col(df, *keywords):
    """Return the first column whose name contains all keywords, else None."""
    for c in df.columns:
        if all(k in c for k in keywords):
            return c
    return None


def empty_manifest_row_df():
    return pd.DataFrame(columns=MANIFEST_COLUMNS)


# ---------------------------------------------------------------- Source 1 --
def fetch_exofop_toi():
    print("[1/4] Fetching ExoFOP TOI table...")
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = CATALOG_DIR / "exofop_toi.csv"
    try:
        r = requests.get(TOI_URL, timeout=30)
        r.raise_for_status()
        toi = pd.read_csv(io.StringIO(r.text), delimiter="|")
        toi.to_csv(raw_path, index=False)
    except Exception as e:
        if raw_path.exists():
            print(f"  download failed ({e}), using cached {raw_path}")
            toi = pd.read_csv(raw_path)
        else:
            print(f"  ERROR: could not fetch ExoFOP TOI table: {e}", file=sys.stderr)
            return empty_manifest_row_df(), empty_manifest_row_df()

    toi = normalize_columns(toi)
    col_tic = find_col(toi, "tic", "id") or find_col(toi, "tic")
    col_disp = find_col(toi, "tfopwg", "disposition")
    col_period = find_col(toi, "period", "days") or find_col(toi, "period")
    col_epoch = find_col(toi, "epoch")
    col_depth = find_col(toi, "depth", "ppm") or find_col(toi, "depth")
    col_duration = find_col(toi, "duration")
    col_tmag = find_col(toi, "tess", "mag") or find_col(toi, "tmag")
    col_comments = find_col(toi, "comment")

    if col_tic is None or col_disp is None:
        print("  ERROR: could not locate tic_id / disposition columns in TOI table",
              file=sys.stderr)
        return empty_manifest_row_df(), empty_manifest_row_df()

    out = pd.DataFrame()
    out["tic_id"] = toi[col_tic].astype("Int64")
    out["disposition"] = toi[col_disp]
    out["period_days"] = toi[col_period] if col_period else np.nan
    out["epoch_btjd"] = toi[col_epoch] if col_epoch else np.nan
    out["depth_ppm"] = toi[col_depth] if col_depth else np.nan
    out["duration_hours"] = toi[col_duration] if col_duration else np.nan
    out["tmag"] = toi[col_tmag] if col_tmag else np.nan
    out["notes"] = toi[col_comments].fillna("") if col_comments else ""
    out["source_catalog"] = "exofop_toi"

    disp = out["disposition"].astype(str).str.strip().str.upper()

    planet = out[disp.isin(["CP", "KP"])].copy()
    planet["label"] = "planet"

    fp = out[disp == "FP"].copy()
    fp["label"] = "unknown"
    fp["notes"] = ("FP (possible blend, disposition=" + disp[disp == "FP"] + "); "
                   + fp["notes"].astype(str))

    train = pd.concat([planet, fp], ignore_index=True)

    candidates = out[disp.isin(["PC", "", "NAN"]) | out["disposition"].isna()].copy()
    candidates["label"] = "candidate"
    candidates_path = ROOT / "manifest_candidates.csv"
    candidates[MANIFEST_COLUMNS].to_csv(candidates_path, index=False)
    print(f"  planet={len(planet)} unknown(FP)={len(fp)} "
          f"candidates(PC/blank)={len(candidates)} -> {candidates_path}")

    return train[MANIFEST_COLUMNS], candidates[MANIFEST_COLUMNS]


# ---------------------------------------------------------------- Source 2 --
def fetch_tess_eb():
    print("[2/4] Fetching TESS-EB catalog...")
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = CATALOG_DIR / "tess_ebs.csv"

    df = None
    for url in TESS_EB_VILLANOVA_CANDIDATES:
        try:
            r = requests.get(url, timeout=20, verify=True)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            print(f"  downloaded from Villanova: {url}")
            break
        except Exception as e:
            print(f"  Villanova candidate failed ({url}): {e}")

    if df is None:
        try:
            r = requests.get(TESS_EB_MAST_URL, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            print(f"  downloaded from MAST HLSP: {TESS_EB_MAST_URL}")
        except Exception as e:
            print(f"  MAST HLSP download failed: {e}")

    if df is None:
        if raw_path.exists():
            print(f"  using cached {raw_path}")
            df = pd.read_csv(raw_path)
        else:
            print("  STOP: could not fetch TESS-EB catalog programmatically.")
            print(f"  Manually download the catalog CSV from http://tessEBs.villanova.edu")
            print(f"  or https://archive.stsci.edu/hlsp/tess-ebs and save it to:")
            print(f"    {raw_path}")
            print("  Then re-run this script; continuing assuming the file exists...")
            if not raw_path.exists():
                return empty_manifest_row_df()
            df = pd.read_csv(raw_path)

    df.to_csv(raw_path, index=False)
    df = normalize_columns(df)

    col_tic = find_col(df, "tic") or find_col(df, "tess_id") or find_col(df, "tess", "id")
    col_period = find_col(df, "period")
    col_epoch = find_col(df, "epoch") or find_col(df, "bjd0") or find_col(df, "t0")
    col_morph = find_col(df, "morph")
    col_tmag = find_col(df, "tmag") or find_col(df, "tess", "mag")

    if col_tic is None:
        print("  ERROR: could not locate tic_id column in TESS-EB catalog", file=sys.stderr)
        return empty_manifest_row_df()

    out = pd.DataFrame()
    out["tic_id"] = df[col_tic].astype(str).str.extract(r"(\d+)")[0].astype("Int64")
    out["label"] = "eb"
    out["source_catalog"] = "tess_ebs_villanova"
    out["disposition"] = "EB"
    out["period_days"] = df[col_period] if col_period else np.nan
    out["epoch_btjd"] = df[col_epoch] if col_epoch else np.nan
    out["depth_ppm"] = np.nan
    out["duration_hours"] = np.nan
    out["tmag"] = df[col_tmag] if col_tmag else np.nan
    out["notes"] = ("morph=" + df[col_morph].astype(str)) if col_morph else ""

    print(f"  eb={len(out)}")
    return out[MANIFEST_COLUMNS]


# ---------------------------------------------------------------- Source 3 --
def fetch_rotation_catalog():
    print("[3/4] Fetching TESS rotation-period catalog via Vizier...")
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = CATALOG_DIR / "rotation_catalog.csv"
    try:
        from astroquery.vizier import Vizier
    except ImportError:
        print("  ERROR: astroquery not installed", file=sys.stderr)
        return empty_manifest_row_df()

    # Cap rows and set an explicit timeout: some matching catalogs (e.g.
    # Kounkel+2022) have 10^5-10^6 rows and an unbounded ROW_LIMIT/timeout
    # can hang for a very long time on a slow connection to CDS.
    Vizier.ROW_LIMIT = 50000
    Vizier.TIMEOUT = 60
    catalog_name = None
    df = None
    needs_gaia_crossmatch = False
    try:
        # "TESS rotation periods" alone doesn't surface Kounkel+2022 in
        # Vizier's fuzzy text search; searching the author name does.
        candidate_keys = []
        descriptions = {}
        for query in ("TESS rotation periods", "Kounkel"):
            found = Vizier.find_catalogs(query)
            for k in found.keys():
                if k not in candidate_keys:
                    candidate_keys.append(k)
                descriptions[k] = found[k].description

        preferred = [k for k in candidate_keys if "kounkel" in descriptions.get(k, "").lower()
                     and "tess" in descriptions.get(k, "").lower()]
        search_order = (preferred + [k for k in candidate_keys if k not in preferred])[:8]

        for key in search_order:
            try:
                tables = Vizier.get_catalogs(key)
            except Exception:
                continue
            for table in tables:
                cols = [c.lower() for c in table.colnames]
                has_tic = any("tic" in c for c in cols)
                has_gaia = any("gaia" in c for c in cols)
                has_prot = any("prot" in c or c == "per" or "period" in c for c in cols)
                if has_prot and (has_tic or has_gaia):
                    df = table.to_pandas()
                    catalog_name = key
                    needs_gaia_crossmatch = has_gaia and not has_tic
                    break
            if df is not None:
                break
    except Exception as e:
        print(f"  Vizier search failed: {e}")

    if df is None:
        if raw_path.exists():
            print(f"  using cached {raw_path}")
            df = pd.read_csv(raw_path)
            catalog_name = "cached"
        else:
            print("  STOP: could not fetch a TESS rotation-period catalog programmatically.")
            print("  Manually search Vizier (web search: 'TESS rotation period'), download a")
            print(f"  catalog with TIC IDs + Prot, and save it to: {raw_path}")
            print("  Then re-run this script; continuing assuming the file exists...")
            if not raw_path.exists():
                return empty_manifest_row_df()
            df = pd.read_csv(raw_path)
            catalog_name = "manual"

    df = normalize_columns(df)
    col_tic = find_col(df, "tic")
    col_gaia = find_col(df, "gaia")
    col_prot = find_col(df, "prot") or find_col(df, "period") or ("per" if "per" in df.columns else None)
    col_amp = find_col(df, "amp")
    col_ra = find_col(df, "raj2000") or find_col(df, "_ra") or ("ra" if "ra" in df.columns else None)
    col_dec = find_col(df, "dej2000") or find_col(df, "_dec") or ("dec" if "dec" in df.columns else None)

    if col_prot is None or (col_tic is None and col_gaia is None):
        print("  ERROR: could not locate tic/gaia/Prot columns in rotation catalog", file=sys.stderr)
        return empty_manifest_row_df()

    out = pd.DataFrame()
    out["period_days"] = pd.to_numeric(df[col_prot], errors="coerce")
    if col_tic is not None and not needs_gaia_crossmatch:
        out["tic_id"] = pd.to_numeric(df[col_tic], errors="coerce").astype("Int64")
    else:
        out["gaia_id"] = pd.to_numeric(df[col_gaia], errors="coerce").astype("Int64")
        if col_ra and col_dec:
            out["ra"] = pd.to_numeric(df[col_ra], errors="coerce")
            out["dec"] = pd.to_numeric(df[col_dec], errors="coerce")

    id_col = "tic_id" if "tic_id" in out.columns else "gaia_id"
    out = out.dropna(subset=[id_col, "period_days"])
    out = out[(out["period_days"] >= 0.5) & (out["period_days"] <= 15)]

    if col_amp:
        amp = pd.to_numeric(df.loc[out.index, col_amp], errors="coerce")
        out = out.assign(_amp=amp).sort_values("_amp", ascending=False).drop(columns="_amp")

    if id_col == "gaia_id":
        # This catalog keys on Gaia DR2 source_id, not TIC. A per-ID crossmatch
        # against MAST is far too slow (~1.7s/ID); instead bulk position-match
        # via the CDS X-Match service against the TIC v8.2 table on Vizier.
        if "ra" not in out.columns or "dec" not in out.columns:
            print("  ERROR: rotation catalog has no RA/Dec for X-Match crossmatch",
                  file=sys.stderr)
            return empty_manifest_row_df()

        print(f"  {len(out)} candidates with Prot in range, keyed by Gaia ID; "
              f"cross-matching a sample against TIC via CDS X-Match...")
        from astroquery.xmatch import XMatch
        import astropy.units as u
        from astropy.table import Table

        sample = out.sample(n=min(6000, len(out)), random_state=42).reset_index(drop=True)
        upload = Table.from_pandas(sample[["gaia_id", "ra", "dec", "period_days"]])

        try:
            xm = XMatch.query(cat1=upload, cat2="vizier:IV/39/tic82",
                               max_distance=2 * u.arcsec, colRA1="ra", colDec1="dec")
        except Exception as e:
            print(f"  ERROR: X-Match crossmatch failed: {e}", file=sys.stderr)
            return empty_manifest_row_df()

        xm_df = xm.to_pandas().sort_values("angDist")
        xm_df = xm_df.drop_duplicates(subset="gaia_id", keep="first")

        out = pd.DataFrame({
            "tic_id": xm_df["TIC"].astype("Int64"),
            "period_days": xm_df["period_days"],
            "tmag": xm_df.get("Tmag", np.nan),
        })
        print(f"  crossmatched {len(out)}/{len(sample)} to TIC IDs")
    else:
        out["tmag"] = np.nan

    out["label"] = "starspot"
    out["source_catalog"] = f"vizier:{catalog_name}"
    out["disposition"] = "rotator"
    out["epoch_btjd"] = np.nan
    out["depth_ppm"] = np.nan
    out["duration_hours"] = np.nan
    out["notes"] = ""

    df_for_cache = df.copy()
    df_for_cache.to_csv(raw_path, index=False)

    print(f"  starspot={len(out)} (catalog={catalog_name})")
    return out[MANIFEST_COLUMNS]


# ---------------------------------------------------------------- Source 4 --
def fetch_null_stars(exclude_tic_ids, n_needed, seed):
    print("[4/4] Building null (quiet-by-exclusion) sample...")
    try:
        from astroquery.vizier import Vizier
        import astropy.units as u
        from astropy.coordinates import SkyCoord
    except ImportError:
        print("  ERROR: astroquery not installed", file=sys.stderr)
        return empty_manifest_row_df()

    # Query the TIC mirror on Vizier with a server-side Tmag filter combined
    # with a region cone. MAST's query_region has no server-side magnitude
    # filter, so it must fetch+locally-filter every star in the cone; since
    # Tmag 8-11 stars are ~0.2% of TIC, that made it orders of magnitude
    # slower (single digits of matches per ~17s call) than doing the same
    # filter server-side via Vizier (hundreds of matches in a few seconds).
    v = Vizier(columns=["TIC", "Tmag"], column_filters={"Tmag": "8.0..11.0"})
    v.ROW_LIMIT = 5000
    v.TIMEOUT = 45

    rng = random.Random(seed)
    collected = []
    seen = set(exclude_tic_ids)
    attempts = 0
    max_attempts = 30

    while len(collected) < n_needed and attempts < max_attempts:
        attempts += 1
        ra = rng.uniform(0, 360)
        dec = rng.uniform(-70, 70)
        try:
            coord = SkyCoord(ra=ra, dec=dec, unit="deg")
            result = v.query_region(coord, radius=2 * u.deg, catalog="IV/39/tic82")
        except Exception as e:
            print(f"  query_region failed at ({ra:.1f},{dec:.1f}): {e}")
            continue

        if not result:
            continue
        df = result[0].to_pandas()
        if "Tmag" not in df.columns or "TIC" not in df.columns:
            continue

        for _, row in df.iterrows():
            tic_id = int(row["TIC"])
            if tic_id in seen:
                continue
            seen.add(tic_id)
            collected.append({"tic_id": tic_id, "tmag": row["Tmag"]})
            if len(collected) >= n_needed:
                break

    out = pd.DataFrame(collected)
    if out.empty:
        print("  WARNING: collected 0 null targets")
        return empty_manifest_row_df()

    out["label"] = "null"
    out["source_catalog"] = "tic_by_exclusion"
    out["disposition"] = ""
    out["period_days"] = np.nan
    out["epoch_btjd"] = np.nan
    out["depth_ppm"] = np.nan
    out["duration_hours"] = np.nan
    out["notes"] = "quiet by exclusion"

    print(f"  null={len(out)} (from {attempts} region queries)")
    return out[MANIFEST_COLUMNS]


# ---------------------------------------------------------------------------
def dedup_and_cap(frames, caps, seed):
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["_priority"] = all_rows["label"].map(LABEL_PRIORITY)

    conflict_notes = {}
    for tic_id, group in all_rows.groupby("tic_id"):
        if group["label"].nunique() > 1:
            labels = sorted(group["label"].unique())
            conflict_notes[tic_id] = f"label conflict across sources: {labels}"

    all_rows = all_rows.sort_values(["tic_id", "_priority"])
    deduped = all_rows.drop_duplicates(subset="tic_id", keep="first").copy()

    for tic_id, note in conflict_notes.items():
        mask = deduped["tic_id"] == tic_id
        deduped.loc[mask, "notes"] = (
            deduped.loc[mask, "notes"].astype(str) + f" [{note}]"
        )

    deduped = deduped.drop(columns="_priority")
    full = deduped[MANIFEST_COLUMNS].reset_index(drop=True)

    capped_parts = []
    for label, cap in caps.items():
        subset = full[full["label"] == label]
        if len(subset) > cap:
            subset = subset.sample(n=cap, random_state=seed)
        capped_parts.append(subset)
    capped = pd.concat(capped_parts, ignore_index=True) if capped_parts else full.copy()

    return full, capped


def print_summary(df, title):
    print(f"\n{title}")
    counts = df["label"].value_counts()
    for label, n in counts.items():
        print(f"  {label:10s} {n}")
    print(f"  {'total':10s} {len(df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-planet", type=int, default=1500)
    ap.add_argument("--n-eb", type=int, default=1500)
    ap.add_argument("--n-starspot", type=int, default=600)
    ap.add_argument("--n-null", type=int, default=600)
    ap.add_argument("--n-unknown", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_toi, _candidates = fetch_exofop_toi()
    eb = fetch_tess_eb()
    starspot = fetch_rotation_catalog()

    known_ids = set(
        pd.concat([train_toi["tic_id"], eb["tic_id"], starspot["tic_id"]])
        .dropna().astype(int).tolist()
    )
    null_needed = max(args.n_null * 2, args.n_null + 200)
    null_df = fetch_null_stars(known_ids, null_needed, args.seed)

    caps = {
        "planet": args.n_planet,
        "eb": args.n_eb,
        "starspot": args.n_starspot,
        "null": args.n_null,
        "unknown": args.n_unknown,
    }

    full, capped = dedup_and_cap([train_toi, eb, starspot, null_df], caps, args.seed)

    full_path = ROOT / "manifest_full.csv"
    manifest_path = ROOT / "manifest.csv"
    full.to_csv(full_path, index=False)
    capped.to_csv(manifest_path, index=False)

    print_summary(full, f"manifest_full.csv -> {full_path}")
    print_summary(capped, f"manifest.csv (capped, seed={args.seed}) -> {manifest_path}")


if __name__ == "__main__":
    main()

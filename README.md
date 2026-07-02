# arvyo-data

Dataset repository for team Arvyo's BAH2026 exoplanet transit-detection
pipeline. Builds a labeled set of TESS light curves for a 4-class classifier
(planet / eclipsing binary / blend / unknown), plus supporting `starspot` and
`null` classes used to validate the blend/novelty forward models against
real stellar variability.

## Label classes and sources

| label      | source                                                                 |
|------------|-------------------------------------------------------------------------|
| `planet`   | ExoFOP TOI table, `TFOPWG Disposition` in {`CP`, `KP`}                  |
| `unknown`  | ExoFOP TOI table, `TFOPWG Disposition` = `FP` (kept as a grab-bag; some FPs are blends and may be relabeled later) |
| `eb`       | TESS-EB catalog (Prša et al. 2022), villanova.edu / MAST HLSP           |
| `starspot` | A TESS rotation-period catalog (e.g. Kounkel et al. 2022) fetched via Vizier, `Prot` in [0.5, 15] days |
| `null`     | Quiet-by-exclusion: TIC stars (Tmag 8-11) not present in any of the above catalogs |
| `blend`    | Not populated here — generated synthetically downstream |

`manifest_candidates.csv` holds TOI rows with disposition `PC` or blank,
excluded from the training manifest, for later inference/demo use.

## Rebuilding from scratch

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/01_build_manifest.py          # -> manifest.csv, manifest_full.csv, manifest_candidates.csv
python scripts/02_download_lightcurves.py --limit 10   # verify a small batch first
python scripts/02_download_lightcurves.py               # full download (1-3 hours)
python scripts/03_preprocess.py
python scripts/verify_dataset.py
```

All three pipeline scripts are resumable and safe to re-run after a crash.
To scale the dataset up, re-run `01_build_manifest.py` with larger `--n-*`
flags, then `02` and `03` again — only new targets are downloaded/processed.

## Data hosting

`data/` is git-ignored (light curves total several GB). The bulk dataset is
hosted on Kaggle Datasets: `<placeholder — Kaggle link TBD>`.

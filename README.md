# Code for "Driver or Fueling Fault? Isolating the Cause of Fuel-Efficiency Anomalies from On-Board Diagnostic Signals Without Joint Cause Labels"

This repository contains the analysis code behind every number, table and figure in the manuscript. All datasets are
public and are downloaded from their original sources; none is redistributed here.

## Environment

Python 3.11 or later (tested with 3.14). Install the dependencies with

```
pip install -r requirements.txt
```

Everything runs on a CPU; the largest step (featurizing the Vehicle Energy Dataset) takes tens of minutes.

## Data

Place the datasets under `data/raw/` as follows (paths are relative to the repository root):

| Dataset | Source | Expected location |
|---|---|---|
| Vehicle Energy Dataset (VED) | https://github.com/gsoh/VED | `data/raw/VED_*_week.csv`, `data/raw/VED_Static_ICE_HEV.xlsx`, `data/raw/VED_Static_PHEV_EV.xlsx` |
| EngineFaultDB | https://github.com/leoxthomas/EngineFaultDB | `data/raw/EngineFaultDB_Final.csv` |
| UAH-DriveSet (1 Hz merged files) | UAH-DriveSet project page | `data/raw/uah/D*_merged.csv` |
| Driving Events | https://zenodo.org/records/6570972 | `data/raw/zenodo_driving/` |
| HCRL driving dataset (Kia Soul) | https://ocslab.hksecurity.net/Datasets/driving-dataset | `data/raw/hcrl/*.csv` |
| This Car is Mine! (Hyundai Sonata) | IEEE DataPort, doi:10.21227/qar8-sd42 | `data/raw/sonata/<driver>/All_*.csv` |
| LiU-ICE diagnosis benchmark | https://vehsys.gitlab-pages.liu.se/diagnostic_competition/ | `data/raw/liuice/*.csv` |
| OBD2_panel_opel_2012 | https://www.kaggle.com/datasets/pedro2025/obd2-panel-opel-2012 | `data/raw/opel_corsa/dataset.csv`, `refueling_history.csv` |

Processed artifacts are written to `data/processed/` (one JSON per script).

## Run order

Run from the repository root, for example `python scripts/p1_features.py`.

| Step | Scripts | Paper |
|---|---|---|
| Data audit and VED trip features | `p0_audit.py`, `p1_features.py` | Sections 3, 4.2 to 4.4 |
| Expected-fuel model and variance decomposition | `p2_*.py`, `p3_decomposition.py`, `p9_ved_stats.py`, `p10_consolidated.py` | Section 6, Table 4 |
| Behaviour-axis validation | `p4_*.py`, `p15_uah_e6.py` | Section 6.3 |
| Engine-fault bench by operating condition | `p45_bench_ci.py`, `diagnostics/p47_bench_conditions.py` | Section 7.1, Table 5, Figure 5 |
| Signature contrast and robustness | `p6_attribution.py`, `p7_baselines.py`, `p11_eaai_ved.py`, `p16_strengthen.py`, `p18_harden.py`, `p25_eq2_reconcile.py`, `p40_fuelcut_bias.py` | Sections 8.1, 8.2, 8.5 |
| Injected ground truth, operating points, comparisons, ablation | `p8_*.py`, `p13`-`p14`, `p17_attribution_fixed.py`, `p19_conformal.py`, `p20`-`p24`, `p33_attribution_baselines.py`, `p35_ablation_runtime.py`, `p36_revision_checks.py`, `p39_trip_effect_size.py` | Sections 4.6, 8.3, 8.4, Tables 7 and 8 |
| Fixed-car replications and false-fault rate | `p29_hcrl.py`, `p31_sonata.py`, `p32_trim_echo.py`, `p34_false_fault_rate.py` | Sections 4.5, 8.6 |
| Measured fueling faults (fault magnitude) | `p30_liuice_leak.py`, `p37_liuice_maf_dose.py` | Section 7.2 |
| Signature contrast on real vehicles | `p48_signature_contrast_real.py` | Section 8.1, Table 6, Figure 6 |
| Seasonal drift, rolling reference, cold start, road car | `p26_seasonal_zero.py`, `p49_rolling_onset.py`, `p28_coldstart.py`, `p38_bank_divergence.py`, `p41_opel_panel.py` | Sections 8.7, 10, Table 9 |
| Figures | `make_figures.py`, `fig_overview.py`, `fig_reanchor.py`, `fig_attribution_corrected.py`, `fig7.py`, `fig_*.py` | Figures 1 to 9 |

`attrib_setup.py` builds the shared frozen driving signature and the VED injection panel used by several scripts. The fault magnitude (3.49 percentage points of fuel per measured air, the smallest LiU-ICE fueling fault) is set where each script defines `S_fault`.
`diagnostics/` holds the diagnostic analyses cited in the text (fast-trim behaviour, baseline mechanism check).

Each script states its pre-registered checks in its header and writes them, with pass or fail, into its JSON output.

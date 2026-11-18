# Code for "Driver or Fueling Fault? A Fuel-Trim Fault Screen with a Driving-Style Indicator for Vehicle Fleets Without Joint Cause Labels"

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
| Data audit, VED trip features, panel flow | `p0_audit.py`, `p1_features.py`, `p62_panel_flow.py`, `p63_signature_features.py` | Sections 3, 4.2 to 4.4, Table 2 |
| Expected-fuel model and variance decomposition | `p2_*.py`, `p3_decomposition.py`, `p9_ved_stats.py`, `p10_consolidated.py` | Sections 6.1, 6.2, Table 4, Appendix L |
| Kinematic-signature validation | `p4_*.py`, `p15_uah_e6.py`, `p66_uah_within_driver.py`, `p68_fleet_outputs_and_roc.py` | Section 6.3, Figure 2, Appendix A |
| Measured fueling faults (fault magnitude) | `p30_liuice_leak.py`, `p37_liuice_maf_dose.py`, `p60_bench_and_target_checks.py` | Section 7 |
| Engine-bench emission signature | `p45_bench_ci.py`, `diagnostics/p47_bench_conditions.py` | Appendix G, Table 15 |
| Signature contrast on real vehicles and robustness | `p48_signature_contrast_real.py`, `p54_kinematic_link_checks.py`, `p64_axis_r2_panel.py`, `p25_eq2_reconcile.py`, `p40_fuelcut_bias.py` | Section 8.1, Table 5, Figure 3, Appendix C |
| Detection and feature-relevance baselines | `p7_baselines.py`, `p33_attribution_baselines.py` | Section 8.2, Appendix I, Table 16 |
| Injected ground truth, operating points, misspecified injections | `p17_attribution_fixed.py`, `p20`-`p24`, `p36_revision_checks.py`, `p39_trip_effect_size.py`, `p52_misspecified_injection.py`, `p57_real_shaped_injection.py`, `p59_physical_perturbation.py` | Sections 4.6, 8.3, Table 6, Figure 4, Appendix B |
| Four-condition test, simple rules, ablation, abstention audit | `p58_factorial_and_baselines.py`, `p35_ablation_runtime.py`, `p19_conformal.py` | Sections 8.3, 8.4, Tables 7 to 9, Appendix I, Table 17 |
| Exploratory vehicle-level check | `p65_vehicle_level_control.py`, `p67_vehicle_guarantee_robustness.py` | Appendix I, Table 18 |
| Fixed-car replications and outputs without an injected cause | `p29_hcrl.py`, `p31_sonata.py`, `p32_trim_echo.py`, `p34_false_fault_rate.py`, `p68_fleet_outputs_and_roc.py` | Section 8.5, Table 10 |
| Seasonal drift, rolling reference, change detectors | `p26_seasonal_zero.py`, `p49_rolling_onset.py`, `p55_change_detectors.py`, `p61_history_and_drivers.py` | Section 8.6, Appendices E and F, Tables 13 and 14 |
| Vehicle-level alarm policy (200 splits, matched false-alarm rate) | `p69_vehicle_alarm_policy.py` | Section 8.6, Table 11 |
| Road car, cold start, bank divergence | `p41_opel_panel.py`, `p28_coldstart.py`, `p38_bank_divergence.py` | Section 8.6, Appendices D and E |
| Figures | `make_figures.py`, `fig_overview.py`, `fig_reanchor.py`, `fig_attribution_corrected.py`, `fig7.py`, `fig_*.py` | Figures 1 to 9 |

`attrib_setup.py` builds the shared frozen kinematic signature and the VED injection panel used by several scripts. The fault magnitude (3.49 percentage points of fuel per measured air, the smallest LiU-ICE fueling fault) is set where each script defines `S_fault`.
`diagnostics/` holds the diagnostic analyses cited in the text (fast-trim behaviour, baseline mechanism check).

Each script states its pre-registered checks in its header and writes them, with pass or fail, into its JSON output.

## Regenerating everything

`bash run_all.sh` runs every analysis script in dependency order (logs in `data/processed/rerun_logs/`) and rebuilds the LaTeX versions. `CLAIM_AUDIT.md` maps each headline number in the paper to its dataset, unit, evidence level and output file.

# Claim audit: headline numbers to datasets, units, evidence level and outputs

Evidence level: **D** direct observation on real data; **M** mechanistic interpretation (closed-loop fuel control);
**S** synthetic consistency (injected effects on real trips; tests the decision rule, not field accuracy).
Regenerate everything with `bash run_all.sh`; each script writes the named JSON in `data/processed/`.

| Claim (paper location) | Value | Dataset, population | Unit | Level | Script -> output |
|---|---|---|---|---|---|
| Data flow (Sec. 3.1, Table 2) | 14,460/197 -> 14,170/194 -> 11,614/186 -> 11,304/136 | VED | trip, vehicle | D | `p62_panel_flow.py` -> `p62_panel_flow.json` |
| Expected-fuel model R^2 (Sec. 6.1) | 0.673 [0.628, 0.709]; MAPE 13.3% | VED 14,170 trips / 194 vehicles | trip, vehicle-grouped folds | D | `p9_ved_stats.py`, `p10_consolidated.py` |
| Predictive variance shares (Sec. 6.2, Table 4) | behaviour 0.080, vehicle-baseline 0.143 | VED 14,170 trips / 194 vehicles | trip; vehicle bootstrap | D (predictive, not causal) | `p10_consolidated.py` (one pass; p3 uses another depth); cross-fitted 0.136-0.138 `p11_eaai_ved.json` |
| Driving Events AUC (Sec. 6.3) | 0.958 [0.927, 0.981] | Driving Events, 3 drivers | event; event bootstrap | D | `p4_behaviour_validation.py` |
| UAH richer features AUC (Sec. 6.3) | 0.829 | UAH-DriveSet, 6 drivers | 30 s window, LODO | D | `p15_uah_e6.py` |
| Deployed speed signature AUC (Abstract, Sec. 6.3) | 0.694; per driver 0.51-0.77 | UAH-DriveSet, 6 drivers | 30 s window, LODO | D | `p61_history_and_drivers.py` -> `p61_history_and_drivers.json` |
| Driver effect size (Sec. 8.3) | d 0.65 [0.41, 0.84]; LODO 0.59 | UAH-DriveSet | window; driver bootstrap | D | `p17_attribution_fixed.py`, `diagnostics/p43_uah_lodo_d.py` |
| Signature contrast (Sec. 8.1, Table 6) | VED kin 0.272, trim -0.041 (equiv. p<0.001); Kia 0.426 / -0.023 (Holm 0.088); Sonata 0.444 / 0.100 | VED 11,304/136; Kia 688 windows/20 sessions; Sonata 651/28 | trip / one-minute window; vehicle or session bootstrap | D | `p48_signature_contrast_real.py` |
| Within-vehicle kinematic link (Sec. 8.1) | 0.33 [0.30, 0.36]; permutation placebo -0.01 | VED 11,304/136 | trip | D | `p54_kinematic_link_checks.py` |
| LiU-ICE leak shift (Sec. 7) | 4 mm 8.0% (grid range 7.4-8.9, CI [6.2, 10.2]); 6 mm 19.5% (18.2-22.2, [16.4, 23.9]); fuel -0.4% / +1.3% | LiU-ICE WLTP | matched operating bin; bin bootstrap | D (fuel per measured air is a proxy, no trims logged) | `p30_liuice_leak.py`, `p60_bench_and_target_checks.py` |
| Air-flow sensor dose response (Sec. 7) | slope 0.98 WLTP, 0.91 FTP-75; +5% gain -3.49 pp | LiU-ICE | matched bin | D + M | `p37_liuice_maf_dose.py` |
| Eq. 1 bias under +5% sensor (Sec. 4.2) | measured air +2.8%, injected fuel +0.4% | LiU-ICE WLTP | matched bin | D | `p60_bench_and_target_checks.py` |
| S_fault (Sec. 4.6) | 3.49 / 1.80 = 1.94 | LiU-ICE anchor / VED within-vehicle trim SD | trip | D (anchor) | `attrib_setup.py`, `p37_liuice_maf_dose.py` |
| Opel fill-up check (Sec. 4.2, 8.6) | Eq. 2 median 1.2% per tank; Eq. 1 -10.6% | Opel Corsa, 28 tanks | tank (not trip) | D | `p41_opel_panel.py` |
| Healthy-car false faults (Abstract, Sec. 8.5) | Kia 5/704 windows (0.7%, [0.23, 1.65]%); Sonata 0/665; no trip | Kia, Sonata | one-minute window, trip | D | `p34_false_fault_rate.py`, `p36_revision_checks.py`, `p53_risk_control_splits.py` |
| Headline injected accuracy (Abstract, Sec. 8.3) | fusion 0.846 window / 0.879 trip; argmax 0.801 / 0.873; closed form 0.847 / 0.796 | VED 11,304/136 | window / trip variants | S | `p33_attribution_baselines.py` -> `p33_baselines.json`, `p24_operating_points.py` |
| Causal rolling reference (Sec. 8.3) | 0.849 / 0.878 | VED 8,610/124 | trip | S | `p50_deployment_faithful.py` |
| Misspecified injections (Abstract, Table 7) | fusion 0.750-0.851 window; target-only 0.630-0.670 | VED 11,304/136 | window | S | `p52_misspecified_injection.py`, `p57_real_shaped_injection.py` |
| Physical perturbation (Sec. 8.3) | 0.856 vs feature-level 0.860; invalid 0% capped (17% uncapped, one-second steps) | VED first 3 weeks, 764 trips | trip, window operating point | S | `p59_physical_perturbation.py` |
| Four-condition factorial (Sec. 8.3, Table 8) | false fault on driver-only 12.6% / 8.6%; false driver on fault-only 9.0% / 7.5%; none->none 52.3% / 63.1% | VED 11,304 per condition | window / trip | S | `p58_factorial_and_baselines.py` |
| Simple rules, matched coverage (Sec. 8.4, Table 10) | fusion 0.844 / 0.874; trim-only 0.834 / 0.830; logistic 0.843 / 0.875 | VED, 68 held-out vehicles | window / trip | S | `p58_factorial_and_baselines.py` |
| Label-free comparison methods (Table 9) | target-only 0.659 / 0.727; pooled 0.729-0.780 / 0.783-0.809; SHAP 0.507 / 0.550 | VED 11,304/136 | window / trip | S | `p33_attribution_baselines.py` |
| Ablation (Table 11) | within-vehicle trim -16 / -11 points; equal weights 0.837 / 0.850 | VED 11,304/136 | window / trip | S | `p35_ablation_runtime.py` |
| Risk-controlled abstention (Sec. 8.4, Table 12) | alpha 0.05: coverage 0.553 / 0.721, error 0.041 [0.036, 0.046] / 0.048 [0.043, 0.053] | VED, 68 cal + 68 disjoint test vehicles | variant; vehicle-disjoint split | S | `p58_factorial_and_baselines.py`; 200 splits `p53_risk_control_splits.py` |
| Fuel magnitude alone (Sec. 8.3) | 0.523; three-channel 0.847 | VED 11,304/136 | window | S | `p22_magnitude_fusion.py` |
| S_fault sweep (Sec. 8.4) | argmax 0.66-0.84 over 0.5-3.5 | VED | window | S | `p23_sfault_sensitivity.py` |
| Rolling reference (Sec. 8.6, App. E, Table 14) | false alarms 0.093 (93.3 per 1,000 trips, 657 per 1,000 h); first-trip detection 0.62 at 3.49 pp | VED 8,610/124, season held out | trip | D (detection on added steps: S) | `p49_rolling_onset.py`, `p61_history_and_drivers.py` |
| Change detectors (App. F, Table 15) | first trip 0.44 vs EWMA 0.24, CUSUM 0.21 at ~1.9 alarms per 100 trips (onsets shared, 50 per vehicle) | VED 89 vehicles | trip | S | `p55_change_detectors.py` |
| Runtime (Sec. 8.4) | 0.71 ms per 33-minute trip, 15 parameters | one CPU thread | trip | measured | `p35_ablation_runtime.py` |

Signature features: the VED fleet's four signature features are recomputed with the source (UAH) definition by `p63_signature_features.py` -> `signature_features.parquet` and swapped in by every signature script.

Leakage assertions: `p19_conformal.py` and `p58_factorial_and_baselines.py` assert no vehicle and no base trip on both
sides of a calibration/test split; `p58` also asserts the fusion equals the linear likelihood-ratio rule;
`p62_panel_flow.py` asserts the 11,304 / 136 panel.

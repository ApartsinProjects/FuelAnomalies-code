#!/usr/bin/env bash
# Regenerate every analysis output behind the paper, in dependency order, then rebuild figures and the LaTeX/PDF
# versions. Each script writes one JSON in data/processed/ and states its pre-registered checks in its header.
# Raw data must be in data/raw/ (see code_release/README.md). Usage: bash run_all.sh [--skip-features]
set -euo pipefail
PY=${PY:-python}
run() { echo "== $1"; OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} $PY -P "scripts/$1.py" > "data/processed/rerun_logs/$1.log" 2>&1; }
mkdir -p data/processed/rerun_logs
[[ "${1:-}" == "--skip-features" ]] || run p1_features                      # VED trip features (slow)
# fleet model, decomposition, behaviour validation
for s in p62_panel_flow p2_gonogo p3_decomposition p4_behaviour_validation p4_permtest p15_uah_e6 p9_ved_stats p10_consolidated; do run $s; done
# signature contrast and robustness
for s in p6_attribution p25_eq2_reconcile p40_fuelcut_bias p48_signature_contrast_real p54_kinematic_link_checks; do run $s; done
# fixed-car replications and false faults (p26 sets the abstention thresholds used by p28/p34/p36)
for s in p29_hcrl p31_sonata p26_seasonal_zero p28_coldstart p34_false_fault_rate p36_revision_checks; do run $s; done
# measured fueling faults, bench, road car
for s in p30_liuice_leak p37_liuice_maf_dose p45_bench_ci p60_bench_and_target_checks p41_opel_panel p56_refuel_trim_steps; do run $s; done
$PY -P diagnostics/p47_bench_conditions.py > data/processed/rerun_logs/p47.log 2>&1
# injected ground truth, comparisons, ablation, abstention, stress tests
for s in p17_attribution_fixed p19_conformal p22_magnitude_fusion p23_sfault_sensitivity p24_operating_points p33_attribution_baselines \
         p35_ablation_runtime p39_trip_effect_size p50_deployment_faithful p52_misspecified_injection p53_risk_control_splits \
         p57_real_shaped_injection p58_factorial_and_baselines p59_physical_perturbation; do run $s; done
# rolling reference, change detectors, history length
for s in p49_rolling_onset p55_change_detectors p61_history_and_drivers; do run $s; done
# figures and paper builds
for s in make_figures fig_attribution_corrected fig_reanchor fig_overview fig7; do run $s || true; done
for t in ijphm twocol arxiv; do $PY -P scripts/make_eaai_latex.py --$t; done
echo "done; see data/processed/*.json and CLAIM_AUDIT.md for the claim-to-output map"

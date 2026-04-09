REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
postrun_pipeline.sh

Runs:
  1) scripts/variantdriver.py   (always)
  2) optional MATLAB PRCC -> .mat files
  3) optional parse .mat -> prcc_results_long.parquet
  4) optional PRCC postrun analysis -> results/prcc_analysis/
  5) optional trajectory plots -> results/trajectory_plots/

Flags control which outputs to produce:
  --prcc-long         Produce results/prcc_results_long.parquet (runs MATLAB PRCC first if needed)
  --prcc-analysis     Produce results/prcc_analysis/ (requires prcc_results_long.parquet; will build if missing)
  --trajectory-plots  Produce results/trajectory_plots/

Convenience:
  --all               Equivalent to: --prcc-long --prcc-analysis --trajectory-plots
  --baseline-policy NAME   Baseline policy for paired deltas + PRCC comparisons (optional)
  --run-dir DIR       If set, use this run directory instead of auto-detecting latest

PRCC analysis options:
  --method NAME       PRCC method passed to prcc_postrun_analysis.py (default: bhfdr)
  --sigonly           Only use significant PRCC entries in prcc_postrun_analysis.py
  --topk N            Top-k parameters (default: 15)

Notes:
- MATLAB scripts root is fixed to: ./scripts/PostRunProcessing/matlab_files
- Python runs via: conda run -n LHDsim python  (hard-coded)
- You should edit scripts/variantdriver.py manually before running this pipeline.

Examples:
  ./scripts/PostRunProcessing/postrun_pipeline.sh --all --baseline-policy observe_only
  ./scripts/PostRunProcessing/postrun_pipeline.sh --trajectory-plots --baseline-policy observe_only
EOF
}

# -------------------------
# Paths / defaults
# -------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"


MATLAB_SCRIPTS_ROOT="${REPO_ROOT}/scripts/PostRunProcessing/matlab_files"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
ALPHA="${ALPHA:-0.05}"

# Hard-coded conda env
CONDA_ENV="LHDsim"
PYTHON_BIN="python"

# flags
DO_PRCC_LONG=0
DO_PRCC_ANALYSIS=0
DO_TRAJECTORIES=0
DO_ALL=0

RUN_DIR=""
BASELINE_POLICY=""
METHOD="bhfdr"
SIGONLY=0
TOPK="15"

abspath() {
  python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"
}

run_python() {
  # Always run inside the fixed conda env
  if command -v conda >/dev/null 2>&1; then
    conda run -n "$CONDA_ENV" "$PYTHON_BIN" "$@"
  else
    echo "WARNING: conda not found; running python directly. Ensure correct env is active." >&2
    "$PYTHON_BIN" "$@"
  fi
}

detect_latest_run_dir() {
  run_python - <<PY
from pathlib import Path
root = Path("${REPO_ROOT}") / "model_runs"
cands = []
if root.exists():
    for p in root.iterdir():
        if p.is_dir() and (p / "StructuredPRCCs").exists():
            cands.append(p)
if not cands:
    raise SystemExit(2)
latest = max(cands, key=lambda p: p.stat().st_mtime)
print(str(latest))
PY
}

# -------------------------
# Arg parsing
# -------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prcc-long) DO_PRCC_LONG=1; shift 1 ;;
    --prcc-analysis) DO_PRCC_ANALYSIS=1; shift 1 ;;
    --trajectory-plots) DO_TRAJECTORIES=1; shift 1 ;;
    --all) DO_ALL=1; shift 1 ;;

    --baseline-policy) BASELINE_POLICY="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;

    --method) METHOD="$2"; shift 2 ;;
    --sigonly) SIGONLY=1; shift 1 ;;
    --topk) TOPK="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "$DO_ALL" == "1" ]]; then
  DO_PRCC_LONG=1
  DO_PRCC_ANALYSIS=1
  DO_TRAJECTORIES=1
fi

# If no output flags provided, default to --all behavior
if [[ "$DO_PRCC_LONG" == "0" && "$DO_PRCC_ANALYSIS" == "0" && "$DO_TRAJECTORIES" == "0" ]]; then
  DO_PRCC_LONG=1
  DO_PRCC_ANALYSIS=1
  DO_TRAJECTORIES=1
fi

# -------------------------
# Preflight checks
# -------------------------

cd "$REPO_ROOT"

if [[ ! -d "$MATLAB_SCRIPTS_ROOT" ]]; then
  echo "ERROR: MATLAB scripts root not found: $MATLAB_SCRIPTS_ROOT" >&2
  exit 2
fi

# Confirm lhsPrccFromCsv exists somewhere under matlab_files (could be .m or .p)
if ! find "$MATLAB_SCRIPTS_ROOT" -maxdepth 5 <math><mrow><mo>−</mo></mrow><mrow><mi>n</mi><mi>a</mi><mi>m</mi><mi>e</mi><mi>&quot;</mi><mi>l</mi><mi>h</mi><mi>s</mi><mi>P</mi><mi>r</mi><mi>c</mi><mi>c</mi><mi>F</mi><mi>r</mi><mi>o</mi><mi>m</mi><mi>C</mi><mi>s</mi><mi>v</mi><mi>.</mi><mi>m</mi><mi>&quot;</mi><mo>−</mo></mrow><mrow><mi>o</mi><mo>−</mo></mrow><mrow><mi>n</mi><mi>a</mi><mi>m</mi><mi>e</mi><mi>&quot;</mi><mi>l</mi><mi>h</mi><mi>s</mi><mi>P</mi><mi>r</mi><mi>c</mi><mi>c</mi><mi>F</mi><mi>r</mi><mi>o</mi><mi>m</mi><mi>C</mi><mi>s</mi><mi>v</mi><mi>.</mi><mi>p</mi><mi>&quot;</mi></mrow></math> | grep -q . ; then
  echo "ERROR: Could not find lhsPrccFromCsv.m or lhsPrccFromCsv.p under:" >&2
  echo "  $MATLAB_SCRIPTS_ROOT" >&2
  exit 2
fi

# -------------------------
# Step 0: run variantdriver
# -------------------------

echo "[pipeline] Step 0: running variantdriver.py"
run_python "${REPO_ROOT}/scripts/variantdriver.py"

# Determine run_dir
if [[ -n "$RUN_DIR" ]]; then
  RUN_DIR="$(abspath "$RUN_DIR")"
else
  echo "[pipeline] Detecting latest run_dir under ${REPO_ROOT}/model_runs/ with StructuredPRCCs/"
  RUN_DIR="$(detect_latest_run_dir)"
fi

if [[ ! -d "$RUN_DIR/StructuredPRCCs" ]]; then
  echo "ERROR: StructuredPRCCs not found under run_dir: $RUN_DIR" >&2
  echo "Check that scripts/variantdriver.py wrote PRCC input CSVs." >&2
  exit 2
fi

echo "[pipeline] Using run_dir: $RUN_DIR"

# -------------------------
# Step 1: MATLAB PRCC + parse to long parquet (prcc_results_long.parquet)
# -------------------------

PRCC_LONG="${RUN_DIR}/results/prcc_results_long.parquet"

if [[ "$DO_PRCC_LONG" == "1" || "$DO_PRCC_ANALYSIS" == "1" ]]; then
  echo "[pipeline] Step 1a: MATLAB PRCC -> .mat files"
  bash "${REPO_ROOT}/scripts/PostRunProcessing/run_prcc_matlab.sh" \
    --run-dir "$RUN_DIR" \
    --matlab-scripts-root "$MATLAB_SCRIPTS_ROOT" \
    --alpha "$ALPHA" \
    --matlab-bin "$MATLAB_BIN"

  echo "[pipeline] Step 1b: parse MATLAB .mat -> prcc_results_long.parquet"
  run_python "${REPO_ROOT}/scripts/PostRunProcessing/parse_matlabPRCC.py" "$RUN_DIR"
fi

# -------------------------
# Step 2: PRCC postrun analysis (heatmaps, lollipops, top tables)
# -------------------------

if [[ "$DO_PRCC_ANALYSIS" == "1" ]]; then
  if [[ ! -f "$PRCC_LONG" ]]; then
    echo "ERROR: Missing $PRCC_LONG (parse step did not produce it)." >&2
    exit 2
  fi

  echo "[pipeline] Step 2: PRCC postrun analysis -> results/prcc_analysis/"
  cmd=( run_python "${REPO_ROOT}/scripts/PostRunProcessing/prcc_postrun_analysis.py" "$RUN_DIR"
        --method "$METHOD" --topk "$TOPK" )

  if [[ "$SIGONLY" == "1" ]]; then
    cmd+=( --sigonly )
  fi
  if [[ -n "$BASELINE_POLICY" ]]; then
    cmd+=( --baseline-policy "$BASELINE_POLICY" )
  fi
  "${cmd[@]}"
fi

# -------------------------
# Step 3: Trajectory plots (incidence/prevalence/summary)
# -------------------------

if [[ "$DO_TRAJECTORIES" == "1" ]]; then
  echo "[pipeline] Step 3: trajectory plots -> results/trajectory_plots/"

  # Prefer trajectory_plots.py (this is what your repo currently has)
  TRAJ_SCRIPT="${REPO_ROOT}/scripts/visualization/trajectory_plots.py"
  if [[ ! -f "$TRAJ_SCRIPT" ]]; then
    # fallback if you later rename it
    TRAJ_SCRIPT="${REPO_ROOT}/scripts/visualization/run_trajectories.py"
  fi
  if [[ ! -f "$TRAJ_SCRIPT" ]]; then
    echo "ERROR: Could not find trajectory plotting script:" >&2
    echo "  expected ${REPO_ROOT}/scripts/visualization/trajectory_plots.py" >&2
    echo "  or      ${REPO_ROOT}/scripts/visualization/run_trajectories.py" >&2
    exit 2
  fi

  cmd=( run_python "$TRAJ_SCRIPT" "$RUN_DIR" )
  if [[ -n "$BASELINE_POLICY" ]]; then
    cmd+=( --baseline "$BASELINE_POLICY" )
  fi
  "${cmd[@]}"
fi

echo "[pipeline] Done."
echo "  run_dir:           $RUN_DIR"
echo "  prcc_long:         $PRCC_LONG"
echo "  prcc_analysis:     ${RUN_DIR}/results/prcc_analysis/"
echo "  trajectory_plots:  ${RUN_DIR}/results/trajectory_plots/"

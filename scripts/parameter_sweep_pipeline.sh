#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
parameter_sweep_pipeline.sh

Always runs:
  0) scripts/variantdriver.py  (you edit values inside it before running)

Then, depending on flags, runs:
  1) MATLAB PRCC -> .mat files
  2) parse .mat -> results/prcc_results_long.parquet
  3) PRCC postrun analysis -> results/prcc_analysis/
  4) trajectory plots -> results/trajectory_plots/

REQUIRED:
  --run-dir DIR
    The run directory variantdriver writes (must contain StructuredPRCCs/ and aggregated_*.parquet)

FLAGS (choose any; if none given, defaults to --all):
  --prcc-long         Produce results/prcc_results_long.parquet (runs MATLAB PRCC first)
  --prcc-analysis     Produce results/prcc_analysis/ (requires prcc_results_long.parquet)
  --trajectory-plots  Produce results/trajectory_plots/
  --all               Equivalent to: --prcc-long --prcc-analysis --trajectory-plots

OPTIONS:
  --baseline-policy NAME   Baseline policy for paired deltas + PRCC comparisons (optional)
  --method NAME            PRCC method for prcc_postrun_analysis.py (default: bhfdr)
  --sigonly                Only use significant entries in prcc_postrun_analysis.py
  --topk N                 Top-k parameters (default: 15)

ENV VARS (optional):
  MATLAB_BIN=/path/to/matlab   (default: matlab)
  ALPHA=0.05                   (default: 0.05)

Examples:
  ./scripts/parameter_sweep_pipeline.sh --run-dir model_runs/TESTMODELRUN --all --baseline-policy observe_only
  ./scripts/parameter_sweep_pipeline.sh --run-dir model_runs/InitialSensitivityAnalysis --trajectory-plots --baseline-policy observe_only
EOF
}

# -------------------------
# Paths / defaults
# -------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Repo root via git (fallback to parent of scripts/ if git fails)
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

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
  python -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"
}

run_python() {
  # unbuffered for immediate prints
  # Ensure the repo root is in PYTHONPATH so absolute imports (e.g. from scripts.x) work on all OS
  export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
  if command -v conda >/dev/null 2>&1; then
    conda run -n "$CONDA_ENV" "$PYTHON_BIN" -u "$@"
  else
    echo "WARNING: conda not found; running python directly. Ensure correct env is active." >&2
    "$PYTHON_BIN" -u "$@"
  fi
}

# -------------------------
# Arg parsing
# -------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;

    --prcc-long) DO_PRCC_LONG=1; shift 1 ;;
    --prcc-analysis) DO_PRCC_ANALYSIS=1; shift 1 ;;
    --trajectory-plots) DO_TRAJECTORIES=1; shift 1 ;;
    --all) DO_ALL=1; shift 1 ;;

    --baseline-policy) BASELINE_POLICY="$2"; shift 2 ;;
    --method) METHOD="$2"; shift 2 ;;
    --sigonly) SIGONLY=1; shift 1 ;;
    --topk) TOPK="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$RUN_DIR" ]]; then
  echo "ERROR: --run-dir is required." >&2
  usage
  exit 2
fi

if [[ "$DO_ALL" == "1" ]]; then
  DO_PRCC_LONG=1
  DO_PRCC_ANALYSIS=1
  DO_TRAJECTORIES=1
fi

# If no output flags provided, default to all outputs
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
if ! find "$MATLAB_SCRIPTS_ROOT" -maxdepth 5 -name "lhsPrccFromCsv.m" -print -quit | grep -q . \
   && ! find "$MATLAB_SCRIPTS_ROOT" -maxdepth 5 -name "lhsPrccFromCsv.p" -print -quit | grep -q . ; then
  echo "ERROR: Could not find lhsPrccFromCsv.m or lhsPrccFromCsv.p under:" >&2
  echo "  $MATLAB_SCRIPTS_ROOT" >&2
  exit 2
fi

# -------------------------
# Step 0: run variantdriver
# -------------------------

# Normalize and check against variantdriver default
RUN_DIR_ABS="$(abspath "$RUN_DIR")"
DEFAULT_DIR="model_runs/TESTMODELRUN"
if [[ "$RUN_DIR_ABS" != "$(abspath "$DEFAULT_DIR")" ]]; then
  echo "[pipeline] Overwriting variantdriver.py output_dir ($DEFAULT_DIR) with $RUN_DIR"
fi
RUN_DIR="$RUN_DIR_ABS"

echo "[pipeline] Step 0: running variantdriver.py"
run_python -m scripts.variantdriver --output-dir "$RUN_DIR"

# Wait briefly for filesystem writes if needed (helps on networked FS)
for i in {1..10}; do
  if [[ -d "$RUN_DIR/StructuredPRCCs" ]]; then
    break
  fi
  sleep 1
done

if [[ ! -d "$RUN_DIR/StructuredPRCCs" ]]; then
  echo "ERROR: StructuredPRCCs not found under run_dir: $RUN_DIR" >&2
  echo "Make sure scripts/variantdriver.py is configured to write to this exact output_dir." >&2
  echo "Tip: check: ls \"$RUN_DIR\"" >&2
  exit 2
fi

echo "[pipeline] Using run_dir: $RUN_DIR"

# -------------------------
# Step 1: MATLAB PRCC + parse to long parquet
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
  run_python -m scripts.PostRunProcessing.parse_matlabPRCC "$RUN_DIR"
fi

# -------------------------
# Step 2: PRCC postrun analysis
# -------------------------

if [[ "$DO_PRCC_ANALYSIS" == "1" ]]; then
  if [[ ! -f "$PRCC_LONG" ]]; then
    echo "ERROR: Missing $PRCC_LONG (parse step did not produce it)." >&2
    exit 2
  fi

  echo "[pipeline] Step 2: PRCC postrun analysis -> results/prcc_analysis/"
  cmd=( run_python -m scripts.PostRunProcessing.prcc_postrun_analysis "$RUN_DIR"
        --method "$METHOD" --topk "$TOPK" )
  [[ "$SIGONLY" == "1" ]] && cmd+=( --sigonly )
  [[ -n "$BASELINE_POLICY" ]] && cmd+=( --baseline-policy "$BASELINE_POLICY" )
  "${cmd[@]}"
fi

# -------------------------
# Step 3: Trajectory plots
# -------------------------

if [[ "$DO_TRAJECTORIES" == "1" ]]; then
  echo "[pipeline] Step 3: trajectory plots -> results/trajectory_plots/"

  TRAJ_SCRIPT="${REPO_ROOT}/scripts/visualization/trajectory_plots.py"
  [[ ! -f "$TRAJ_SCRIPT" ]] && TRAJ_SCRIPT="${REPO_ROOT}/scripts/visualization/run_trajectories.py"
  [[ ! -f "$TRAJ_SCRIPT" ]] && { echo "ERROR: trajectory plot script not found" >&2; exit 2; }

  cmd=( run_python "$TRAJ_SCRIPT" "$RUN_DIR" )
  [[ -n "$BASELINE_POLICY" ]] && cmd+=( --baseline "$BASELINE_POLICY" )
  "${cmd[@]}"
fi

echo "[pipeline] Done."
echo "  run_dir:           $RUN_DIR"
echo "  prcc_long:         $PRCC_LONG"
echo "  prcc_analysis:     ${RUN_DIR}/results/prcc_analysis/"
echo "  trajectory_plots:  ${RUN_DIR}/results/trajectory_plots/"
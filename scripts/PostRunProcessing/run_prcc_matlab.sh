#!/usr/bin/env bash
set -euo pipefail

usage() {
cat <<'EOF'
Run lhsPrccFromCsv.m on every PRCC-input CSV under a run directory.

Required:
  --run-dir DIR
  --matlab-scripts-root DIR

Optional:
  --alpha FLOAT      (default: 0.05)
  --func NAME        (default: lhsPrccFromCsv)
  --pattern GLOB     (default: *stat-cols-*.csv)
  --matlab-bin PATH  (default: matlab)
EOF
}

MATLAB_BIN="matlab"
FUNC="lhsPrccFromCsv"
ALPHA="0.05"
PATTERN="*stat-cols-*.csv"
RUN_DIR=""
SCRIPTS_ROOT=""

abspath() {
  python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --matlab-scripts-root) SCRIPTS_ROOT="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --func) FUNC="$2"; shift 2 ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    --matlab-bin) MATLAB_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$RUN_DIR" || -z "$SCRIPTS_ROOT" ]]; then
  echo "ERROR: --run-dir and --matlab-scripts-root are required." >&2
  usage
  exit 2
fi

RUN_DIR="$(abspath "$RUN_DIR")"
SCRIPTS_ROOT="$(abspath "$SCRIPTS_ROOT")"

STRUCT_DIR="${RUN_DIR}/StructuredPRCCs"
if [[ ! -d "$STRUCT_DIR" ]]; then
  echo "ERROR: StructuredPRCCs directory not found: $STRUCT_DIR" >&2
  exit 2
fi

# Find matching CSVs (NUL-delimited for safety with spaces)
FILES=()
while IFS= read -r -d '' f; do
  FILES+=("$f")
done < <(find "$STRUCT_DIR" -type f -name "$PATTERN" -print0)

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "ERROR: No CSVs matched pattern '$PATTERN' under $STRUCT_DIR" >&2
  exit 2
fi

echo "[run_prcc_matlab] Found ${#FILES[@]} PRCC input CSVs under $STRUCT_DIR"

# macOS-friendly mktemp usage
TMPBASE="${TMPDIR:-/tmp}"
FILELIST="$(mktemp "${TMPBASE}/prcc_filelist_XXXXXX")"
MATLAB_DRIVER="$(mktemp "${TMPBASE}/prcc_driver_XXXXXX.m")"

cleanup() { rm -f "$FILELIST" "$MATLAB_DRIVER"; }
trap cleanup EXIT

printf "%s\n" "${FILES[@]}" > "$FILELIST"

cat > "$MATLAB_DRIVER" <<'MATLAB'
try
    scriptsRoot = getenv('PRCC_SCRIPTS_ROOT');
    fileListPath = getenv('PRCC_FILELIST');
    funcName = getenv('PRCC_FUNC');
    alpha = str2double(getenv('PRCC_ALPHA'));

    if isempty(scriptsRoot) || isempty(fileListPath) || isempty(funcName) || isnan(alpha)
        error('Missing env vars. Need PRCC_SCRIPTS_ROOT/PRCC_FILELIST/PRCC_FUNC/PRCC_ALPHA.');
    end

    addpath(genpath(scriptsRoot));

    fid = fopen(fileListPath, 'r');
    if fid < 0
        error('Could not open file list: %s', fileListPath);
    end
    C = textscan(fid, '%s', 'Delimiter', '\n');
    fclose(fid);
    files = C{1};

    if isempty(files)
        error('File list is empty.');
    end

    files = sort(files); % do sorting here (portable)

    f = str2func(funcName);

    for i = 1:numel(files)
        inCsv = strtrim(files{i});
        if isempty(inCsv); continue; end

        [folder, name, ~] = fileparts(inCsv);
        outMat = fullfile(folder, [name '.mat']);

        fprintf('[MATLAB] %s(%s, %g, %s)\n', funcName, inCsv, alpha, outMat);
        feval(f, inCsv, alpha, outMat);
    end

catch ME
    disp(getReport(ME, 'extended'));
    exit(1);
end
exit(0);
MATLAB

export PRCC_SCRIPTS_ROOT="$SCRIPTS_ROOT"
export PRCC_FILELIST="$FILELIST"
export PRCC_FUNC="$FUNC"
export PRCC_ALPHA="$ALPHA"

"$MATLAB_BIN" -batch "run('$MATLAB_DRIVER')"

echo "[run_prcc_matlab] Done. .mat files written next to each input CSV."
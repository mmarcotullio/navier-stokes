#!/bin/bash

# --- 0. Global Variables ---
# When piped via  sed ... | bash  BASH_SOURCE[0] is empty; fall back to pwd.
# In that case the caller must cd to the project root before piping.
_SRC="${BASH_SOURCE[0]:-}"
if [[ -n "$_SRC" && "$_SRC" != "bash" ]]; then
    # Script is run directly: forward/scripts/ -> ../.. = project root
    ROOT_DIR="$(cd "$(dirname "$_SRC")/../.." && pwd)"
else
    # Piped via  cd <project_root> && sed ... | bash  → pwd IS the project root
    ROOT_DIR="$(pwd)"
fi
CASES_DIR="$ROOT_DIR/cfd_training_data/cases"
TEMPLATE="$ROOT_DIR/cfd_training_data/template_case"
MESHBASE="$ROOT_DIR/cfd_training_data/meshBase"

# Physics Ranges (Laminar-focused for FNO)
Re_min=100
Re_max=1000
U_min=0.1
U_max=1.0

# Geometry
R_PIPE=0.5
D_PIPE=$(awk "BEGIN{print 2.0 * $R_PIPE}")

# Helper function with high-res seeding
rand_uniform() {
    awk -v min="$1" -v max="$2" -v seed="$(date +%N)$RANDOM" 'BEGIN{srand(seed); print min+rand()*(max-min)}'
}

# --- 1. The run_case Function ---
run_case() {
    local i=$1
    local CASE_NAME=$(printf "case_%04d" "$i")
    local CASE_PATH="$CASES_DIR/$CASE_NAME"

    # 1. Clean & Create Directory (with NFS busy-check)
    if [ -d "$CASE_PATH" ]; then 
        rm -rf "$CASE_PATH" 2>/dev/null || { 
            echo "  [INFO] $CASE_NAME busy, waiting for NFS..."
            sleep 2
            rm -rf "$CASE_PATH"
        }
    fi
    mkdir -p "$CASE_PATH"

    # 2. Copy Template & Mesh
    cp -r "$TEMPLATE/." "$CASE_PATH/"
    mkdir -p "$CASE_PATH/constant"
    if [ -d "$MESHBASE/constant/polyMesh" ]; then
        cp -r "$MESHBASE/constant/polyMesh" "$CASE_PATH/constant/"
    else
        echo "  [ERROR] $CASE_NAME: Mesh not found."
        return 1
    fi

    # 3. Calculate Physics Parameters
    local Re=$(rand_uniform "$Re_min" "$Re_max")
    local U_in=$(rand_uniform "$U_min" "$U_max")
    
    # Use ${D_PIPE} to ensure variable is passed to Python
    local nu=$(python3 -c "print($U_in * $D_PIPE / $Re)")

    # 4. Modify OpenFOAM Files via Python
    export CASE_PATH nu U_in
    
    # Update nu
    python3 - <<'EOF'
import os, re, pathlib
path = pathlib.Path(f"{os.environ['CASE_PATH']}/constant/transportProperties")
if path.exists():
    txt = path.read_text()
    txt = re.sub(r"nu\s*\[[^]]*\]\s*[0-9eE.+-]+", f"nu               [0 2 -1 0 0 0 0] {os.environ['nu']}", txt)
    path.write_text(txt)
EOF

    # Update Umean
    python3 - <<'EOF'
import os, re, pathlib
path = pathlib.Path(f"{os.environ['CASE_PATH']}/0/U")
if path.exists():
    txt = path.read_text()
    if re.search(r"\bUmean\b", txt):
        txt = re.sub(r"(\bUmean\s+)([0-9eE.+-]+)(\s*;)", fr"\g<1>{os.environ['U_in']}\3", txt)
    else:
        m = re.search(r"(dimensions\s*\[[^\]]*\]\s*;\s*)", txt)
        if m:
            txt = txt[:m.end()] + f"\nUmean           {os.environ['U_in']};\n" + txt[m.end():]
    path.write_text(txt)
EOF

    # 5. Metadata & Permissions
    echo "$CASE_NAME,$Re,$U_in,$nu,$R_PIPE,$D_PIPE" > "$CASE_PATH/metadata.line"
    mkdir -p "$CASE_PATH/dynamicCode"
    chmod -R 777 "$CASE_PATH/dynamicCode" 2>/dev/null

    # 6. Batch Info Printout
    echo "------------------------------------------------"
    echo "RUNNING: $CASE_NAME"
    echo "  Reynolds (Re): $Re"
    echo "  Inlet Vel (U): $U_in"
    echo "  Diameter (D):  $D_PIPE"
    echo "  Viscosity (nu): $nu"
    echo "------------------------------------------------"

    # 7. Run Solver
    (
        cd "$CASE_PATH"
        simpleFoam > log.simpleFoam 2>&1
    )

    if [ $? -eq 0 ]; then
        echo "  [SUCCESS] $CASE_NAME"
    else
        echo "  [FAIL]    $CASE_NAME (Check log.simpleFoam)"
    fi
}

# --- 2. EXECUTE ---
echo "Starting Batch Simulation..."
for i in {1..1000}; do
    run_case "$i"
done
echo "Batch finished."
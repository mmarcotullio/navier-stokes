#!/usr/bin/env python3
import os
import csv
import glob
import numpy as np

# -------------------------------------------------------------------
# Paths / constants
# -------------------------------------------------------------------

# Repo root: one level above this scripts/ directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CASES_DIR = os.path.join(ROOT_DIR, "..", "cfd_training_data", "cases")
OUT_BASE_DIR = os.path.join(ROOT_DIR, "..", "cfd_training_data", "fno_grid")

# Cylindrical grid file (OpenFOAM-style points file with parentheses)
CYL_GRID_PATH = os.path.join(ROOT_DIR, "..", "cfd_training_data", "cyl_grid.xyz")

# Name of the sampling set in sampleDict
SET_NAME = "pipeCylGrid"

# Name of the functionObject / sampleDict in controlDict
SAMPLE_FUNC_NAME = "sampleDict"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def load_cyl_grid_points(cyl_grid_path: str) -> np.ndarray:
    """
    Load OpenFOAM-style points file used for the cylindrical grid:

        (x y z)
        (x y z)
        ...

    or plain "x y z" without parentheses.

    Returns:
        coords_grid_flat: (N, 3) float64

    This is the FULL Cartesian grid the FNO will operate on.
    """
    pts = []
    with open(cyl_grid_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip obvious headers / dictionary stuff
            if (
                line.startswith("FoamFile")
                or line.startswith("{")
                or line.startswith("}")
                or line.startswith("//")
                or line.startswith("#")
            ):
                continue

            # Handle lines like "(0.0 0.1 0.2)" or "( 0.0 0.1 0.2 )"
            if line[0] == "(" and line[-1] == ")":
                line = line[1:-1].strip()

            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                x, y, z = map(float, parts[:3])
            except ValueError:
                continue

            pts.append((x, y, z))

    if not pts:
        raise ValueError(f"No points parsed from cylindrical grid file: {cyl_grid_path}")

    coords = np.array(pts, dtype=float)
    print(f"Loaded cylindrical grid: {coords.shape[0]} points from {cyl_grid_path}")
    return coords


def make_coord_key(x: float, y: float, z: float, ndigits: int = 5):
    """
    Create a hashable key for a coordinate triple, with rounding to reduce
    floating-point issues.

    We use 5 decimal digits so that minor differences between the grid file
    and sampled coordinates map to the same key, while grid points remain distinct.
    """
    return (
        round(float(x), ndigits),
        round(float(y), ndigits),
        round(float(z), ndigits),
    )


def read_combined_raw(path: str):
    """
    Read a combined 'raw' sampled file for the set SET_NAME containing both
    p and U, e.g. pipeCylGrid_p_U.xy, with columns:

        x  y  z  p  Ux  Uy  Uz

    Returns:
        coords: (N,3)
        U:      (N,3)
        p:      (N,)
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#") or line.startswith("//"):
                continue

            # Remove parentheses if present
            line_clean = line.replace("(", " ").replace(")", " ")
            parts = line_clean.split()

            if len(parts) < 7:
                continue

            try:
                vals = list(map(float, parts[:7]))
            except ValueError:
                continue

            rows.append(vals)

    if not rows:
        raise ValueError(f"No valid data lines parsed from combined raw file: {path}")

    arr = np.array(rows, dtype=float)  # (N, 7)
    x = arr[:, 0]
    y = arr[:, 1]
    z = arr[:, 2]
    p = arr[:, 3]
    Ux = arr[:, 4]
    Uy = arr[:, 5]
    Uz = arr[:, 6]

    coords = np.stack([x, y, z], axis=1)
    U = np.stack([Ux, Uy, Uz], axis=1)
    return coords, U, p


def process_case(case_dir: str, coords_grid_flat: np.ndarray, grid_index_map: dict):
    """
    For a single case_XXX directory (steady-state simpleFoam run):

      1. Find the last pseudo-time directory under postProcessing/sampleDict —
         that is the converged equilibrium state.
      2. Read the combined p/U sampled on the set SET_NAME from that directory.
      3. Scatter the field onto the FULL cylindrical grid coords_grid_flat.
      4. Save to OUT_BASE_DIR/case_XXX/{coords,u,p,sampled_mask}.npy

    Saved files (no time dimension — steady-state has one state per case):

        coords.npy      : (N_grid, 3)   FULL Cartesian grid (for FNO)
        u.npy           : (N_grid, 3)   equilibrium velocity, 0 at unsampled points
        p.npy           : (N_grid,)     equilibrium pressure, 0 at unsampled points
        sampled_mask.npy: (N_grid,)     1 if that grid point was sampled, else 0
    """
    case_name = os.path.basename(case_dir.rstrip("/"))
    print(f"\n=== Processing {case_name} ===")

    out_dir = os.path.join(OUT_BASE_DIR, case_name)
    os.makedirs(out_dir, exist_ok=True)

    # Locate sampleDict output (produced by simpleFoam at runtime)
    sample_root = os.path.join(case_dir, "postProcessing", SAMPLE_FUNC_NAME)
    if not os.path.isdir(sample_root):
        print(f"  No {SAMPLE_FUNC_NAME} directory for {case_name}, skipping.")
        return

    # Collect all valid pseudo-time directories and take the last (most converged)
    valid_time_dirs = []
    for d in glob.glob(os.path.join(sample_root, "*")):
        if not os.path.isdir(d):
            continue
        try:
            float(os.path.basename(d))
        except ValueError:
            continue
        valid_time_dirs.append(d)

    if not valid_time_dirs:
        print(f"  No time directories under {sample_root}, skipping.")
        return

    # Sort numerically and take the last (highest pseudo-time = most converged)
    valid_time_dirs.sort(key=lambda d: float(os.path.basename(d)))
    last_dir = valid_time_dirs[-1]
    t_str = os.path.basename(last_dir)
    print(f"  Using converged state from pseudo-time directory: {t_str}")

    pattern = os.path.join(last_dir, f"{SET_NAME}*.xy")
    matches = sorted(glob.glob(pattern))
    if not matches:
        print(f"  Missing combined raw file in {last_dir}, skipping.")
        return

    coords_ref, U_vals, p_vals = read_combined_raw(matches[0])
    N_sample = coords_ref.shape[0]
    N_grid = coords_grid_flat.shape[0]

    print(f"  N_sample={N_sample}, N_grid={N_grid}")

    # Build mapping from sampled coords to full grid indices
    grid_idx = np.empty(N_sample, dtype=int)
    for i in range(N_sample):
        x, y, z = coords_ref[i]
        key = make_coord_key(x, y, z)
        if key not in grid_index_map:
            raise KeyError(
                f"Sampled coordinate {coords_ref[i]} (rounded key {key}) "
                f"not found in cylindrical grid. "
                f"Check that cyl_grid.xyz is the same points file used by sampleDict."
            )
        grid_idx[i] = grid_index_map[key]

    if len(np.unique(grid_idx)) != N_sample:
        print("  WARNING: sampled coordinates map to non-unique grid indices.")

    # Scatter sampled values into full grid arrays (no time dimension)
    u_full = np.zeros((N_grid, 3), dtype=float)
    p_full = np.zeros((N_grid,), dtype=float)

    u_full[grid_idx, :] = U_vals
    p_full[grid_idx] = p_vals

    # sampled_mask: which grid points were successfully sampled
    sampled_mask = np.zeros((N_grid,), dtype=np.float32)
    sampled_mask[grid_idx] = 1.0

    # Save outputs
    np.save(os.path.join(out_dir, "coords.npy"), coords_grid_flat)
    np.save(os.path.join(out_dir, "u.npy"), u_full)
    np.save(os.path.join(out_dir, "p.npy"), p_full)
    np.save(os.path.join(out_dir, "sampled_mask.npy"), sampled_mask)

    missing = int(N_grid - np.count_nonzero(sampled_mask))
    print(f"  sampled_mask: {int(np.count_nonzero(sampled_mask))} sampled, {missing} missing")
    print(
        f"  Saved coords {coords_grid_flat.shape}, "
        f"u {u_full.shape}, p {p_full.shape} to {out_dir}"
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    os.makedirs(OUT_BASE_DIR, exist_ok=True)

    coords_grid_flat = load_cyl_grid_points(CYL_GRID_PATH)
    N_grid = coords_grid_flat.shape[0]

    print("Building coordinate -> index map for full grid...")
    grid_index_map = {}
    for idx, (x, y, z) in enumerate(coords_grid_flat):
        key = make_coord_key(x, y, z)
        if key in grid_index_map:
            raise ValueError(f"Duplicate coordinate key {key} in cylindrical grid.")
        grid_index_map[key] = idx
    print(f"  Built grid_index_map for {N_grid} points.")

    cases = sorted(
        d for d in glob.glob(os.path.join(CASES_DIR, "case_*"))
        if os.path.isdir(d)
    )
    if not cases:
        print(f"No cases found under {CASES_DIR}")
        return

    for case_dir in cases:
        process_case(case_dir, coords_grid_flat, grid_index_map)

    # Generate metadata.csv from per-case metadata.line files
    generate_metadata_csv(cases)


def generate_metadata_csv(case_dirs):
    """
    Read metadata.line from each case directory and write a consolidated
    metadata.csv into OUT_BASE_DIR (fno_grid/).

    metadata.line format (written by run_batch_openfoam_parallel.sh):
        case_name,Re,U_in,nu,R_PIPE,D_PIPE
    """
    meta_path = os.path.join(OUT_BASE_DIR, "metadata.csv")
    rows = []

    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir.rstrip("/"))
        meta_line_path = os.path.join(case_dir, "metadata.line")

        if not os.path.isfile(meta_line_path):
            print(f"  WARNING: No metadata.line for {case_name}, skipping metadata entry.")
            continue

        with open(meta_line_path, "r") as f:
            line = f.readline().strip()

        if not line:
            print(f"  WARNING: Empty metadata.line for {case_name}, skipping.")
            continue

        parts = line.split(",")
        if len(parts) < 3:
            print(f"  WARNING: Malformed metadata.line for {case_name}: {line}")
            continue

        rows.append({
            "case_id": parts[0].strip(),
            "Re": parts[1].strip(),
            "U_in": parts[2].strip(),
        })

    if not rows:
        print("  WARNING: No metadata rows collected. metadata.csv will not be written.")
        return

    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "Re", "U_in"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote metadata.csv with {len(rows)} entries to {meta_path}")


if __name__ == "__main__":
    main()

# FNO CFD Explorer — Visualisation App Report

**Project:** `fourier_net_steady_state`
**App location:** `three/`
**Date:** 2026-02-22

---

## Table of Contents

1. [Overview](#1-overview)
2. [Technology Stack](#2-technology-stack)
3. [System Architecture](#3-system-architecture)
4. [Machine Learning Model](#4-machine-learning-model)
5. [Backend](#5-backend)
6. [Frontend](#6-frontend)
7. [Design Decisions](#7-design-decisions)
8. [Physics Indicators](#8-physics-indicators)
9. [Known Limitations](#9-known-limitations)
10. [Running the App](#10-running-the-app)

---

## 1. Overview

The FNO CFD Explorer is an interactive browser-based visualisation of steady-state 3D pipe flow predicted by a trained Fourier Neural Operator (FNO). The app allows a user to adjust physical parameters (Reynolds number, inlet velocity) and grid resolution, then instantly see the resulting velocity and pressure fields rendered as a 3D point cloud and a 2D centreline heatmap — without running a CFD solver. Inference takes place server-side in milliseconds using the trained model.

The physical domain is a circular pipe (length 10 m, radius 0.5 m) containing a spherical obstacle (radius 0.25 m centred at x = 3 m). The flow is steady, incompressible, and laminar across the full controllable range.

---

## 2. Technology Stack

### Backend

| Component | Version / Detail |
|-----------|-----------------|
| Python | 3.x (project venv) |
| PyTorch | GPU/CPU autodetect via `torch.cuda.is_available()` |
| Flask | Latest in venv |
| flask-cors | CORS enabled for all origins |
| NumPy | Array construction, masking, downsampling |

### Frontend

| Component | Version |
|-----------|---------|
| React | 18.3.1 |
| Three.js | 0.172.0 |
| D3 | 7.9.0 |
| Vite | 6.1.0 |
| @vitejs/plugin-react | 4.3.4 |

### Three.js add-ons (imported from `three/addons/`)

- `OrbitControls` — camera orbit, zoom, pan
- `CSS2DRenderer` + `CSS2DObject` — DOM text labels overlaid on the WebGL canvas

### Infrastructure

- `start.sh` — bash launcher that kills stale processes on ports 5050/5173 with `fuser -k`, activates the Python venv, starts Flask and Vite as background processes, and traps `EXIT`/`INT`/`TERM` for clean shutdown
- Vite dev server proxies `/api/*` → `http://localhost:5050` via `vite.config.js`, so the React app makes same-origin requests with no CORS handling needed in the browser

---

## 3. System Architecture

```
┌─────────────────────────────────────┐
│  Browser (localhost:5173)           │
│                                     │
│  ┌──────────┐   ┌─────────────────┐ │
│  │ Controls │   │  Three.js       │ │
│  │  panel   │   │  WebGL canvas   │ │
│  │ (React)  │   │  + CSS2D labels │ │
│  └────┬─────┘   └────────┬────────┘ │
│       │                  │          │
│  ┌────▼──────────────────▼────────┐ │
│  │       App.jsx (React state)    │ │
│  │  Re, U_in, res, colorMode,     │ │
│  │  clipR, scaleMode, loading…    │ │
│  └────────────────┬───────────────┘ │
│                   │ fetch /api/predict│
│  ┌────────────────▼───────────────┐ │
│  │    D3 Heatmap (Heatmap.jsx)    │ │
│  │    y = 0 slice, SVG            │ │
│  └────────────────────────────────┘ │
└──────────────┬──────────────────────┘
               │ HTTP JSON (Vite proxy)
               ▼
┌─────────────────────────────────────┐
│  Flask server (localhost:5050)      │
│                                     │
│  GET /api/predict?Re=&U_in=&Ny=&Nz= │
│                                     │
│  1. Build 7-channel Cartesian grid  │
│  2. pad_to_efficient_grid()         │
│  3. FNO3d forward pass (no_grad)    │
│  4. Crop padding, apply U_in scale  │
│  5. Mask fluid domain               │
│  6. Downsample 3D → max 120k pts    │
│  7. Extract y=0 slice (full, no DS) │
│  8. Return JSON                     │
│                                     │
│  GET /api/health                    │
└───────────────┬─────────────────────┘
                │ torch.load
                ▼
        fno3d_best.pt (weights)
```

---

## 4. Machine Learning Model

### 4.1 Architecture — FNO3d

The model is a 3D Fourier Neural Operator (Li et al., 2020, "Fourier Neural Operator for Parametric Partial Differential Equations"). It operates directly on structured 3D Cartesian grids.

**Instantiation:**
```python
FNO3d(modes_x=16, modes_y=10, modes_z=10,
      width=32, in_channels=7, out_channels=4, n_layers=4)
```

**Forward pass:**
1. **Lifting layer** — `Conv3d(7 → 32, kernel=1)` maps input channels to the latent width
2. **N = 4 FNO blocks**, each:
   - `SpectralConv3d`: 3D real FFT → truncate to (modes_x, modes_y, modes_z) low-frequency modes across all 4 half-space octants → learnable complex weight multiplication → inverse FFT
   - `Conv3d(32 → 32, kernel=1)`: residual local branch (bypass the spectral path)
   - Sum of spectral + local branch, then GELU activation
3. **Projection** — `Conv3d(32 → 32, kernel=1)` + GELU + `Conv3d(32 → 4, kernel=1)`

The spectral convolution retains modes_x=16, modes_y=10, modes_z=10 Fourier modes. For an (Nx × Ny × Nz) grid, the real FFT produces (Nx × Ny × Nz/2+1) complex coefficients; only the low-frequency block is multiplied by learned weights (four weight tensors, one per quadrant in the xy-plane). This gives the model global receptive field at every layer while remaining resolution-invariant.

**Approximate parameter count:** ≈ 2–5 M (dominated by the 4 × 4 spectral weight tensors of shape `(32, 32, 16, 10, 10, 2)` each)

### 4.2 Inputs (7 channels)

| Channel | Description |
|---------|-------------|
| 0 | `x_norm` — x coordinate normalised to [0, 1] |
| 1 | `y_norm` — y coordinate normalised to [0, 1] |
| 2 | `z_norm` — z coordinate normalised to [0, 1] |
| 3 | `fluid_mask` — 1 inside circular pipe cross-section, 0 outside |
| 4 | `cyl_mask` — 1 inside spherical obstacle (dist ≤ 0.25 m from centre), 0 outside |
| 5 | `Re_norm` — Reynolds number normalised: `(Re − 100) / 900` |
| 6 | `U_in_norm` — inlet velocity normalised: `(U_in − 0.1) / 0.9` |

Channels 5 and 6 are broadcast as constant fields (same value at every grid point), encoding the operating condition into the spatial input.

### 4.3 Outputs (4 channels)

| Channel | Description |
|---------|-------------|
| 0 | `ux / U_in` — streamwise velocity, normalised by inlet velocity |
| 1 | `uy / U_in` — radial velocity (y), normalised |
| 2 | `uz / U_in` — radial velocity (z), normalised |
| 3 | `p / U_in²` — kinematic pressure (mean-centred over fluid domain) |

Physical velocities are recovered by multiplying by `U_in`; physical pressure by multiplying by `U_in²`.

### 4.4 Training

| Hyperparameter | Value |
|---------------|-------|
| Max epochs | 2000 (early stopping: patience 60) |
| Batch size | 8 |
| Optimizer | AdamW (lr = 1 × 10⁻³, weight_decay = 1 × 10⁻⁴) |
| LR schedule | Linear warmup (20 epochs, factor 0.01→1) then cosine annealing to 1 × 10⁻⁶ (T_max = 380) |
| Loss | Masked relative L2: `‖pred − target‖_F / ‖target‖_F`, restricted to active fluid cells |
| Grad clipping | max_norm = 1.0 |
| Data split | 90% train / 10% val, case-level (no data leakage) |
| Augmentation | Random 0°/90°/180°/270° rotation around pipe axis; uy/uz velocity components rotated consistently |
| Padding | Spatial dims padded to multiples of 4 for FFT efficiency (`pad_to_efficient_grid`) |

Training data was generated with OpenFOAM's `simpleFoam` solver across Re ∈ [100, 1000] and U_in ∈ [0.1, 1.0] m/s. The best checkpoint (lowest validation loss) is saved as `three/fno3d_best.pt`.

---

## 5. Backend

### 5.1 Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | GET | Run inference, return JSON point cloud + slice |
| `/api/health` | GET | Returns `{"status": "ok", "device": "cuda"/"cpu"}` |

**Query parameters for `/api/predict`:**

| Param | Default | Clip range | Description |
|-------|---------|------------|-------------|
| `Re` | 500 | [100, 1000] | Reynolds number |
| `U_in` | 0.5 | [0.1, 1.0] | Inlet velocity (m/s) |
| `Ny` | 20 | [9, 29] | Cross-section grid points (y) |
| `Nz` | 20 | [9, 29] | Cross-section grid points (z) |

Both `Ny` and `Nz` are snapped to odd integers server-side (`if Ny % 2 == 0: Ny += 1`). This guarantees the grid always contains y = 0 and z = 0, which ensures the centreline lies on a grid node and that the outermost Cartesian ring is not fully masked by the circular pipe cross-section.

### 5.2 Inference Pipeline

```
1. Derive Nx from isotropic spacing: dy = 1/(Ny−1), Nx = floor(10/dy) + 1
2. Build Cartesian meshgrid: x∈[0,10], y/z∈[−0.5, 0.5]
3. Compute masks: fluid_mask (r ≤ 0.5), cyl_mask (dist to (3,0,0) ≤ 0.25)
4. Stack 7-channel input tensor (float32)
5. pad_to_efficient_grid() — pad to multiples of 4 in each spatial dim
6. FNO3d forward pass under torch.no_grad() → (1, 4, Nx', Ny', Nz')
7. Crop padding: pred[0, :, :Nx, :Ny, :Nz]
8. Recover physical fields: ux/uy/uz × U_in, p × U_in²
9. Extract y ≈ 0 slice (iy = Ny//2) — full resolution, no downsampling
10. Filter 3D: fluid_only = (fluid_mask > 0.5) & (cyl_mask < 0.5)
11. Downsample if n_points > 120,000 (uniform random, replace=False)
12. Serialise as float32 lists → JSON
```

### 5.3 JSON Response Schema

```json
{
  "x": [...],      // 3D point positions (m)
  "y": [...],
  "z": [...],
  "ux": [...],     // velocity components (m/s)
  "uy": [...],
  "uz": [...],
  "p": [...],      // kinematic pressure (m²/s²)
  "vmag": [...],   // |u| (m/s)
  "slice": {       // full y≈0 cross-section (no downsampling)
    "x": [...], "z": [...], "vmag": [...], "p": [...],
    "y_val": 0.0, "Nx": 101, "Nz": 19
  },
  "meta": {
    "Nx": 101, "Ny": 19, "Nz": 19,
    "Re": 500.0, "U_in": 0.5,
    "n_points": 42381
  }
}
```

The `slice` payload is never downsampled; the frontend D3 heatmap always receives the complete xz cross-section. The 3D payload is capped at 120,000 points to keep transfer below ~2 MB.

---

## 6. Frontend

### 6.1 Component Structure

```
App.jsx
├── Slider             (reusable range input with label/value)
├── Colorbar           (gradient legend with min/mid/max ticks)
├── Heatmap.jsx        (D3 SVG, y=0 cross-section)
└── Three.js scene     (managed by useEffect, stored in sceneRef)
    ├── WebGLRenderer
    ├── CSS2DRenderer  (overlaid, pointer-events:none)
    ├── OrbitControls
    ├── PerspectiveCamera
    ├── Scene
    │   ├── Cylinder wireframe (pipe outline)
    │   ├── Torus (obstacle ring)
    │   ├── Line + CSS2DObject × 3 (x/y/z axes with labels)
    │   ├── CSS2DObject × 2 (Inlet / Outlet labels)
    │   └── Points (fluid point cloud — rebuilt on each fetch/clip/colourMode)
    └── Animation loop (requestAnimationFrame)
```

### 6.2 State Management

All UI state lives in a single `App` component with React `useState`:

| State | Type | Description |
|-------|------|-------------|
| `Re` | number | Reynolds number (100–1000, step 10) |
| `uIn` | number | Inlet velocity (0.1–1.0, step 0.01) |
| `res` | number | Grid resolution Ny=Nz (9–29, step 2, always odd) |
| `colorMode` | string | `"vmag"` or `"pressure"` |
| `clipR` | number | Wall-clip fraction (0.2–1.0, step 0.05) |
| `scaleMode` | string | `"auto"` or `"fixed"` |
| `fixedRange` | object\|null | `{vmin, vmax}` locked when scaleMode = "fixed" |
| `loading` | bool | True during active fetch |
| `error` | string\|null | Last fetch error message |
| `meta` | object\|null | Grid/parameter info from last response |
| `vrange` | object | `{vmin, vmax}` currently displayed |
| `sliceData` | object\|null | y=0 slice payload for Heatmap |

**Refs** (mutable values readable from callbacks without stale closure):

| Ref | Mirrors state | Purpose |
|-----|--------------|---------|
| `rawDataRef` | — | Full JSON response from last successful fetch |
| `clipRRef` | `clipR` | Read inside `fetchAndRender` (defined with `[]` deps) |
| `colorModeRef` | `colorMode` | Read inside `applyPointCloud` |
| `fixedRangeRef` | `fixedRange` | Read inside `applyPointCloud`; also written immediately in handler before state batch |

### 6.3 Data Flow and Effects

```
[Re, uIn, res, colorMode change]
        │
        ├─ useEffect([Re,uIn,res,colorMode])
        │    └─ setLoading(true)  ← immediate blur before debounce fires
        │
        └─ useEffect([Re,uIn,res,colorMode,fetchAndRender])
             └─ setTimeout(350ms debounce)
                  └─ fetchAndRender()
                       ├─ AbortController: cancel previous in-flight request
                       ├─ fetch /api/predict
                       ├─ applyPointCloud(data, colorMode, clipRRef.current)
                       ├─ setMeta / setSliceData / setLoading(false)
                       └─ abort → no state changes (loading stays true)

[clipR change]
        └─ useEffect([clipR])
             └─ applyPointCloud(rawDataRef.current, colorModeRef.current, clipR)
                  (no fetch — client-side geometry filter only)

[scaleMode / fixedRange change]
        └─ useEffect([scaleMode, fixedRange])
             ├─ applyPointCloud(rawDataRef, colorModeRef, clipRRef)
             └─ setSliceData(prev => ({...prev}))  ← force D3 re-render
```

### 6.4 Point Cloud Construction (`buildPointCloud`)

```javascript
function buildPointCloud(data, colorMode, clipR, fixedVmin, fixedVmax)
```

1. Compute `rMax = R_PIPE × clipR`
2. Filter indices: keep only points where `√(y² + z²) ≤ rMax + ε`
3. Determine colour range:
   - **Auto:** scan kept points for `[vmin, vmax]`
   - **Fixed:** use `fixedVmin`, `fixedVmax` directly
4. Build `Float32Array` position buffer (x offset by `-L_PIPE/2` to centre scene at origin)
5. Build `Float32Array` colour buffer using `colorFn(t)` where `t = (v − vmin) / (vmax − vmin)`, clamped to [0, 1]
6. Return `THREE.BufferGeometry` with `position` and `color` attributes

Point size is 0.035 (world units), size-attenuating, 85% opacity.

### 6.5 Colormap

**Moreland Cool-to-Warm** (ParaView default), implemented as piecewise linear interpolation in sRGB over 5 control points:

| t | R | G | B | Appearance |
|---|---|---|---|------------|
| 0.00 | 0.230 | 0.299 | 0.754 | Moreland blue |
| 0.25 | 0.553 | 0.690 | 0.896 | Light blue |
| 0.50 | 0.865 | 0.865 | 0.865 | Neutral grey |
| 0.75 | 0.797 | 0.424 | 0.367 | Salmon |
| 1.00 | 0.706 | 0.016 | 0.150 | Moreland crimson |

Source: Kenneth Moreland, "Diverging Color Maps for Scientific Visualization" (2009). This diverging map is perceptually balanced around the midpoint grey and avoids the green channel that makes blue-to-red ramps appear to include a distracting intermediate hue.

The same `colorFn` is exported from `App.jsx` and imported by `Heatmap.jsx` so both views always use an identical mapping.

### 6.6 D3 Heatmap (`Heatmap.jsx`)

A fixed-size SVG renders the y = 0 (pipe centreline) xz cross-section. Dimensions are chosen for a true 10:1 physical aspect ratio:

| Constant | Value | Meaning |
|----------|-------|---------|
| `PX_PER_M` | 108 | Pixels per metre on both axes |
| `IW` | 1080 px | Inner width (x: 0–10 m) |
| `IH` | 108 px | Inner height (z: −0.5–0.5 m) |
| `W` | 1162 px | Total SVG width (including margins) |
| `H` | 200 px | Total SVG height |

The container has `maxWidth: calc(100% - 28px)` and `overflowX: auto`, so the SVG is horizontally scrollable on narrow viewports.

**D3 render is intentionally triggered only when `sliceData` changes**, not on every `colorMode`/`vmin`/`vmax` change. This prevents the heatmap from momentarily rendering with stale range data while a new fetch is in flight. Instead, `colorMode`, `vmin`, and `vmax` are kept in refs (`colorModeRef`, `vminRef`, `vmaxRef`) that are read synchronously at D3 render time.

**Loading state** is communicated by a CSS filter: `grayscale(1) blur(4px)` applied immediately (transition: none) when loading starts, and faded off with a 0.4 s ease transition once new data renders.

**Physical overlays:**
- Pipe boundary: two horizontal `<line>` elements at z = ±0.5 m
- Cylinder obstacle: an `<ellipse>` with physically correct semi-axes `rx = xScale(CYL_R) − xScale(0)`, `ry = |zScale(0) − zScale(CYL_R)|`, drawn with a dashed white stroke

**Axes:** D3 `axisBottom` and `axisLeft` with styled tick labels; axis labels `x (m)` and `z (m)` with units restored (coordinates are in physical metres).

**Title format:** `Top-down slice  y = 0.00 m  —  Velocity magnitude`

### 6.7 CSS2D Labels

Three.js `CSS2DRenderer` is overlaid on the WebGL canvas (`position: absolute; top: 0; pointer-events: none`). It projects 3D world positions to screen coordinates and positions HTML `<div>` elements accordingly.

**Axes** originate from the bottom-front-left corner of the pipe bounding box `O = (−5, −0.5, −0.5)` with a 0.18 m outset pad:

| Axis | Direction | Colour |
|------|-----------|--------|
| x | Along pipe | `#ff6b6b` (red) |
| y | Vertical | `#69db7c` (green) |
| z | Depth | `#74c0fc` (blue) |

**Inlet / Outlet** labels sit at `(±5, 0, 0.72)` — above the pipe end-faces at z = R + 0.22. Both axes and labels use `text-shadow` for readability over the dark background.

---

## 7. Design Decisions

### 7.1 Resolution — Odd Snapping

The grid cross-section uses Cartesian coordinates. At even `N`, the linspace `np.linspace(−0.5, 0.5, N)` does not include 0, so the outermost ring of grid points sits just outside the circular pipe mask when projected, producing a visually sparse result ("only 6 dots" at resolution 8). Snapping to odd integers guarantees `0 ∈ linspace` and the grid always includes both centreline and wall ring. Enforced both in the UI (slider `step=2`, `min=9`) and server-side.

### 7.2 Abort Controller Pattern

Each call to `fetchAndRender` immediately aborts any previous in-flight fetch via `AbortController`. On abort, the catch block checks `err.name !== "AbortError"` and skips all state updates — critically including `setLoading(false)`. This prevents a superseded response from briefly clearing the loading overlay before the new response arrives.

### 7.3 Immediate Blur on Parameter Change

A separate `useEffect([Re, uIn, res, colorMode])` calls `setLoading(true)` synchronously before the 350 ms debounce fires. This ensures the heatmap greys and blurs at the moment the user touches a slider, not 350 ms later, preventing a window where the old field is visible with new parameter values applied.

### 7.4 Ref Pattern for D3

D3's render is in a `useEffect([sliceData])` closure. Adding `colorMode`/`vmin`/`vmax` to the dependency array would cause D3 to immediately re-render with the new colour mode but the previous response's data — before the new fetch completes — producing a briefly distorted heatmap. The ref pattern (`colorModeRef.current = colorMode` on every render) lets D3 read the latest values at render time without being in the effect's dependency array.

### 7.5 Wall Clip

A "Wall clip" slider (0.2–1.0) filters the 3D point cloud client-side by radial distance `r = √(y² + z²) ≤ R_PIPE × clipR`. This peels back the outer annulus to reveal interior velocity structure (e.g. the near-wall boundary layer versus the core jet). No re-fetch is performed; the raw response is cached in `rawDataRef` and `buildPointCloud` refilters it instantly.

### 7.6 Auto / Fixed Color Range

By default (`scaleMode = "auto"`) the colormap rescales to `[vmin, vmax]` of the current response, maximising contrast at any operating point but making cross-run comparisons misleading (low and high U_in appear equally coloured). In `"fixed"` mode, the range is snapped to the current run's `[vmin, vmax]` at the moment the button is clicked, and all subsequent runs are mapped onto the same scale. This allows direct visual comparison of, e.g., U_in = 0.1 m/s vs 1.0 m/s. The locked range is displayed numerically in the panel.

When the mode changes, `applyPointCloud` is called immediately (no fetch) and `setSliceData(prev => ({...prev}))` creates a new object reference to force Heatmap's `useEffect([sliceData])` to re-run D3 with the updated vmin/vmax refs.

### 7.7 Downsampling

The server caps the 3D point payload at 120,000 points via `np.random.choice(n, 120_000, replace=False)`. The y=0 slice is never downsampled. The cap was chosen to keep JSON transfer below ~2 MB (6 float32 arrays × 120k × 4 bytes ≈ 2.9 MB before JSON overhead) while maintaining visual density sufficient to see the flow structure.

### 7.8 Grid Padding for FFT Efficiency

Before inference, the input tensor is padded to spatial dimensions that are multiples of 4 (`pad_to_efficient_grid`). PyTorch's FFT is significantly faster on sizes with small prime factors. The padding is zero-filled and cropped after the forward pass so it has no effect on the output.

---

## 8. Physics Indicators

### 8.1 Kinematic Viscosity from Re

The Reynolds number for pipe flow is:

```
Re = U_bulk · D / ν
```

Where `U_bulk` is the bulk mean (cross-sectional average) velocity, `D = 2 · R_PIPE = 1 m` is the pipe diameter, and `ν` is the kinematic viscosity. Rearranging:

```
ν = U_in · D / Re = U_in / Re    [m²/s]   (since D = 1 m)
```

`U_in` is a valid substitute for `U_bulk` because for incompressible flow in a constant cross-section pipe, conservation of mass requires the bulk mean velocity to be constant along the entire pipe length, regardless of how the velocity profile develops. The inlet condition is specified as a uniform profile, so `U_in = U_bulk` everywhere.

The app displays the full substituted calculation step-by-step:
```
ν = U · D / Re
  = 0.50 × 1 / 500
  = 1.00 × 10⁻³ m²/s
≈ 1,000 centistokes (cSt) — glycerin
laminar flow
```

### 8.2 Fluid Reference Table

Nearest real-world fluid is identified by minimum log-distance in viscosity space (i.e., closest by multiplicative ratio, appropriate for a quantity spanning many orders of magnitude). Reference values at approximately 20 °C:

| Fluid | ν (m²/s) |
|-------|----------|
| Mercury | 1.14 × 10⁻⁷ |
| Water | 1.00 × 10⁻⁶ |
| Ethanol | 1.52 × 10⁻⁶ |
| Milk | 2.00 × 10⁻⁶ |
| Blood | 3.50 × 10⁻⁶ |
| Ethylene glycol | 1.70 × 10⁻⁵ |
| Olive oil | 8.40 × 10⁻⁵ |
| SAE 20 oil | 1.50 × 10⁻⁴ |
| SAE 40 oil | 3.00 × 10⁻⁴ |
| Glycerin | 1.18 × 10⁻³ |
| Honey | 7.00 × 10⁻³ |

**Sources:**
- White, F.M. *Fluid Mechanics*, 8th ed., McGraw-Hill (2011), Appendix A, Table A.1
- Engineering Toolbox, engineeringtoolbox.com/kinematic-viscosity-d_397.html

In the controllable range (Re 100–1000, U_in 0.1–1.0 m/s, D = 1 m), `ν` spans 100–10,000 cSt, corresponding to heavy oils through glycerin and honey. This reflects the 1 m pipe diameter — the same Re in a 10 mm laboratory tube would imply a much less viscous fluid.

### 8.3 Flow Regime

All Re values in [100, 1000] are firmly laminar. Pipe flow transition to turbulence begins around Re ≈ 2,300 (Hagen, 1839; Reynolds, 1883). The indicator shows "laminar / transitional / turbulent" dynamically but will always read "laminar" within the model's trained domain.

### 8.4 Entry Length and Apparent Velocity Near the Inlet

The colourmap always auto-scales (unless fixed) to the current field's `[vmin, vmax]`. Near the inlet, the velocity profile is flat (plug flow). As the flow develops downstream, the no-slip boundary layer thickens and — by mass conservation — the centreline must accelerate to compensate, peaking near `U_max ≈ 2 × U_bulk` for a fully developed Poiseuille profile. Consequently the colour scale's maximum is pulled toward the downstream, fully-developed region, making the inlet appear relatively blue (slow) even though the bulk velocity is the same everywhere. This is a physically real entry-length effect, not a visualisation artefact.

---

## 9. Known Limitations

| Limitation | Detail |
|-----------|--------|
| Steady-state only | The FNO predicts the converged equilibrium; it cannot model transients or unsteady shedding |
| Laminar range | Training covers Re 100–1000; extrapolation beyond is undefined |
| Cartesian grid on circular domain | Grid points at the pipe wall are partially masked. At low resolution, the outer Cartesian ring falls outside the circular cross-section, reducing apparent point count |
| Obstacle geometry | The cylinder mask uses a spherical approximation `dist ≤ 0.25 m` rather than a true cylinder; this matches the training data geometry but differs from a 2D-extruded cylinder |
| 3D downsampling | The 120k-point cap means the displayed cloud is a random subsample; point density is not uniform across the domain |
| Pressure is mean-centred | Absolute pressure is not meaningful; only pressure differences within a single run are physically interpretable |
| D = 1 m pipe | The viscosity-to-fluid mapping assumes a 1 m diameter pipe, which is very large. For laboratory-scale pipes the equivalent fluid would be far less viscous |

---

## 10. Running the App

**Prerequisites:** Python venv with PyTorch, Flask, flask-cors, NumPy; Node.js with npm.

```bash
cd fourier_net_steady_state/three
./start.sh
# Opens: http://localhost:5173
```

The script kills any stale processes on ports 5050 and 5173 first, then starts both services. Press Ctrl+C to shut down both.

**Manual start (separate terminals):**
```bash
# Terminal 1 — backend
source ../venv/bin/activate
python server.py

# Terminal 2 — frontend
cd app
npm run dev
```

**Health check:**
```bash
curl http://localhost:5050/api/health
# {"device": "cpu", "status": "ok"}
```

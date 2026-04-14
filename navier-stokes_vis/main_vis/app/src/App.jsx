import { useEffect, useRef, useState, useCallback } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import Heatmap from "./Heatmap.jsx";
import InverseHeatmap from "./InverseHeatmap.jsx";
import InverseToolPanel from "./InverseToolPanel.jsx";

// ── Colormap: Moreland Cool-to-Warm (ParaView default) ───────────────────────
// Control points from Kenneth Moreland, "Diverging Color Maps for Scientific
// Visualization" (2009).  Interpolated in sRGB as a close approximation.
const COLORMAP_STOPS = [
  [0.000, 0.230, 0.299, 0.754],  // Moreland blue
  [0.250, 0.553, 0.690, 0.896],  // light blue
  [0.500, 0.865, 0.865, 0.865],  // neutral gray (midpoint)
  [0.750, 0.797, 0.424, 0.367],  // salmon
  [1.000, 0.706, 0.016, 0.150],  // Moreland crimson red
];

export function colorFn(t) {
  t = Math.max(0, Math.min(1, t));
  let i = 0;
  while (i < COLORMAP_STOPS.length - 2 && t >= COLORMAP_STOPS[i + 1][0]) i++;
  const [t0, r0, g0, b0] = COLORMAP_STOPS[i];
  const [t1, r1, g1, b1] = COLORMAP_STOPS[i + 1];
  const a = (t - t0) / (t1 - t0);
  return [r0 + a * (r1 - r0), g0 + a * (g1 - g0), b0 + a * (b1 - b0)];
}

// ── Constants matching the physics domain ────────────────────────────────────
const L_PIPE = 10.0;
const R_PIPE = 0.5;
const CYL_X  = 3.0;
const CYL_R  = 0.25;

// ── Fluid reference: kinematic viscosity ν at ~20 °C ─────────────────────────
// Source: White, F.M. "Fluid Mechanics" 8th ed. (2011), App. A, Table A.1;
//         Engineering Toolbox, engineeringtoolbox.com/kinematic-viscosity-d_397.html
const FLUID_REF = [
  { name: "mercury",         nu: 1.14e-7 },
  { name: "water",           nu: 1.00e-6 },
  { name: "ethanol",         nu: 1.52e-6 },
  { name: "milk",            nu: 2.00e-6 },
  { name: "blood",           nu: 3.50e-6 },
  { name: "ethylene glycol", nu: 1.70e-5 },
  { name: "olive oil",       nu: 8.40e-5 },
  { name: "SAE 20 oil",      nu: 1.50e-4 },
  { name: "SAE 40 oil",      nu: 3.00e-4 },
  { name: "glycerin",        nu: 1.18e-3 },
  { name: "honey",           nu: 7.00e-3 },
];

// Nearest fluid in log-viscosity space (multiplicative distance)
function closestFluid(nu) {
  return FLUID_REF.reduce((best, f) =>
    Math.abs(Math.log(f.nu / nu)) < Math.abs(Math.log(best.nu / nu)) ? f : best
  );
}

// Scientific notation with Unicode superscripts, e.g. "1.00 × 10⁻³"
const _sup = ['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹'];
function sciStr(v) {
  const exp = Math.floor(Math.log10(Math.abs(v)));
  const m   = v / Math.pow(10, exp);
  const e   = String(Math.abs(exp)).split('').map(d => _sup[+d]).join('');
  return `${m.toFixed(2)} × 10${exp < 0 ? '⁻' : ''}${e}`;
}

// ── Three.js helpers ──────────────────────────────────────────────────────────
// clipR: fraction of pipe radius to show (1.0 = full pipe, 0.2 = core only)
function buildPointCloud(data, colorMode, clipR = 1.0, fixedVmin = null, fixedVmax = null) {
  const rMax = R_PIPE * clipR;
  const vals = colorMode === "pressure" ? data.p : data.vmag;

  // Filter to points within the clipped radius
  const keep = [];
  for (let i = 0; i < data.x.length; i++) {
    if (Math.sqrt(data.y[i] ** 2 + data.z[i] ** 2) <= rMax + 1e-6) keep.push(i);
  }

  const n = keep.length;
  const positions = new Float32Array(n * 3);
  const colors    = new Float32Array(n * 3);

  let vmin, vmax;
  if (fixedVmin !== null && fixedVmax !== null) {
    vmin = fixedVmin; vmax = fixedVmax;
  } else {
    vmin = Infinity; vmax = -Infinity;
    for (const i of keep) {
      if (vals[i] < vmin) vmin = vals[i];
      if (vals[i] > vmax) vmax = vals[i];
    }
  }
  const vrange = vmax - vmin || 1;

  for (let j = 0; j < n; j++) {
    const i = keep[j];
    positions[j * 3]     = data.x[i] - L_PIPE / 2;
    positions[j * 3 + 1] = data.y[i];
    positions[j * 3 + 2] = data.z[i];

    const t = (vals[i] - vmin) / vrange;
    const [r, g, b] = colorFn(t);
    colors[j * 3]     = r;
    colors[j * 3 + 1] = g;
    colors[j * 3 + 2] = b;
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setAttribute("color",    new THREE.BufferAttribute(colors, 3));
  geom.computeBoundingSphere();
  return { geom, vmin, vmax };
}

// ── Slider component ──────────────────────────────────────────────────────────
function Slider({ label, value, min, max, step, onChange, format }) {
  return (
    <div style={styles.sliderRow}>
      <div style={styles.sliderLabel}>
        <span>{label}</span>
        <span style={styles.sliderValue}>{format ? format(value) : value}</span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={styles.range}
      />
    </div>
  );
}

// ── Colorbar legend ───────────────────────────────────────────────────────────
function Colorbar({ vmin, vmax, label }) {
  const stops = Array.from({ length: 101 }, (_, i) => {
    const t = i / 100;
    const [r, g, b] = colorFn(t);
    return `rgb(${(r*255)|0},${(g*255)|0},${(b*255)|0})`;
  }).join(", ");

  return (
    <div style={styles.colorbar}>
      <div style={styles.colorbarLabel}>{label}</div>
      <div style={styles.colorbarTrack}>
        <div style={{
          ...styles.colorbarGradient,
          background: `linear-gradient(to right, ${stops})`,
        }} />
        <div style={styles.colorbarTicks}>
          <span>{vmin.toFixed(3)}</span>
          <span>{((vmin + vmax) / 2).toFixed(3)}</span>
          <span>{vmax.toFixed(3)}</span>
        </div>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const mountRef  = useRef(null);
  const sceneRef  = useRef(null);
  const fetchRef  = useRef(null);

  const [Re,        setRe]        = useState(500);
  const [uIn,       setUIn]       = useState(0.5);
  const [res,       setRes]       = useState(19);   // always odd
  const [colorMode, setColorMode] = useState("vmag");

  const [clipR,     setClipR]     = useState(1.0);
  const [scaleMode, setScaleMode] = useState("auto");
  const [fixedRange, setFixedRange] = useState(null);  // { vmin, vmax } | null

  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState(null);
  const [meta,      setMeta]      = useState(null);
  const [vrange,    setVrange]    = useState({ vmin: 0, vmax: 1 });
  const [sliceData, setSliceData] = useState(null);

  const [invSliceData, setInvSliceData] = useState(null);
  const [invRePred,    setInvRePred]    = useState(null);

  const handleVisualizeInverse = useCallback((sliceData, rePred) => {
    setInvSliceData(sliceData);
    setInvRePred(rePred);
  }, []);

  const handleResetInverse = useCallback(() => {
    setInvSliceData(null);
    setInvRePred(null);
  }, []);

  // ── Left-panel drag-resize ─────────────────────────────────────────────────
  const [panelWidth, setPanelWidth] = useState(390);
  const _panelW    = useRef(390);
  _panelW.current  = panelWidth;
  const _panelDrag = useRef({ on: false, x0: 0, w0: 0 });

  // Track canvas area width so Heatmap can clamp itself
  const [canvasW, setCanvasW] = useState(() => window.innerWidth - 390 - 5);
  useEffect(() => {
    setCanvasW(window.innerWidth - panelWidth - 5);
  }, [panelWidth]);
  useEffect(() => {
    const onResize = () => setCanvasW(window.innerWidth - _panelW.current - 5);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Refs so callbacks can read latest values without stale closures
  const rawDataRef    = useRef(null);
  const clipRRef      = useRef(1.0);
  clipRRef.current    = clipR;
  const colorModeRef  = useRef(colorMode);
  colorModeRef.current = colorMode;
  const fixedRangeRef = useRef(null);
  fixedRangeRef.current = fixedRange;

  // ── Initialise Three.js once ────────────────────────────────────────────
  useEffect(() => {
    const el = mountRef.current;
    const w = el.clientWidth, h = el.clientHeight;

    // WebGL renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h);
    renderer.setClearColor(0x0a0a0f);
    el.appendChild(renderer.domElement);

    // CSS2D renderer for text labels (overlaid on top)
    const labelRenderer = new CSS2DRenderer();
    labelRenderer.setSize(w, h);
    labelRenderer.domElement.style.position = "absolute";
    labelRenderer.domElement.style.top = "0";
    labelRenderer.domElement.style.pointerEvents = "none";
    el.appendChild(labelRenderer.domElement);

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 200);
    camera.position.set(0, 2, 10);
    camera.lookAt(0, 0, 0);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance   = 0.5;
    controls.maxDistance   = 50;

    // Pipe outline
    const pipeBox = new THREE.Mesh(
      new THREE.CylinderGeometry(R_PIPE, R_PIPE, L_PIPE, 48, 1, true),
      new THREE.MeshBasicMaterial({ color: 0x334455, wireframe: true, transparent: true, opacity: 0.15 })
    );
    pipeBox.rotation.z = Math.PI / 2;
    scene.add(pipeBox);

    // Sphere obstacle rings — two perpendicular great circles
    const cylGeom = new THREE.TorusGeometry(CYL_R, 0.008, 8, 32);
    const cylMat  = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 });
    // Ring 1: y-z plane (cross-sectional, perpendicular to pipe axis)
    const cylMesh = new THREE.Mesh(cylGeom, cylMat);
    cylMesh.position.set(CYL_X - L_PIPE / 2, 0, 0);
    cylMesh.rotation.y = Math.PI / 2;
    scene.add(cylMesh);
    // Ring 2: x-z plane (axial, perpendicular to ring 1)
    const cylMesh2 = new THREE.Mesh(cylGeom, cylMat);
    cylMesh2.position.set(CYL_X - L_PIPE / 2, 0, 0);
    cylMesh2.rotation.x = Math.PI / 2;
    scene.add(cylMesh2);

    // ── Labeled axes along the pipe bounding box ─────────────────────────────
    // Corner origin: bottom-front-left of the pipe bounding box
    const O  = [-L_PIPE / 2, -R_PIPE, -R_PIPE];  // [-5, -0.5, -0.5]
    const pad = 0.18;  // how far outside the box to draw the axis lines

    function axisLine(from, to, color) {
      const geom = new THREE.BufferGeometry().setFromPoints(
        [new THREE.Vector3(...from), new THREE.Vector3(...to)]
      );
      return new THREE.Line(geom,
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }));
    }

    function axisLabel(text, position, color) {
      const div = document.createElement("div");
      div.textContent = text;
      div.style.cssText = [
        `color:${color}`,
        "font-size:12px",
        "font-weight:700",
        "font-family:system-ui,sans-serif",
        "pointer-events:none",
        "text-shadow:0 1px 4px rgba(0,0,0,0.9)",
        "white-space:nowrap",
      ].join(";");
      const obj = new CSS2DObject(div);
      obj.position.set(...position);
      return obj;
    }

    // X axis (along pipe) — red
    scene.add(axisLine(
      [O[0] - pad, O[1] - pad, O[2] - pad],
      [L_PIPE / 2 + pad, O[1] - pad, O[2] - pad],
      0xff6b6b
    ));
    scene.add(axisLabel("x", [L_PIPE / 2 + pad + 0.2, O[1] - pad, O[2] - pad], "#ff6b6b"));

    // Y axis (vertical) — green
    scene.add(axisLine(
      [O[0] - pad, O[1] - pad, O[2] - pad],
      [O[0] - pad, R_PIPE + pad, O[2] - pad],
      0x69db7c
    ));
    scene.add(axisLabel("y", [O[0] - pad, R_PIPE + pad + 0.15, O[2] - pad], "#69db7c"));

    // Z axis (depth) — blue
    scene.add(axisLine(
      [O[0] - pad, O[1] - pad, O[2] - pad],
      [O[0] - pad, O[1] - pad, R_PIPE + pad],
      0x74c0fc
    ));
    scene.add(axisLabel("z", [O[0] - pad, O[1] - pad, R_PIPE + pad + 0.15], "#74c0fc"));

    // ── Inlet / Outlet labels ────────────────────────────────────────────────
    function endLabel(text, xPos) {
      const div = document.createElement("div");
      div.textContent = text;
      div.style.cssText = [
        "color:#94a3b8",
        "font-size:11px",
        "font-weight:600",
        "font-family:system-ui,sans-serif",
        "letter-spacing:0.08em",
        "text-transform:uppercase",
        "pointer-events:none",
        "text-shadow:0 1px 4px rgba(0,0,0,0.9)",
      ].join(";");
      const obj = new CSS2DObject(div);
      obj.position.set(xPos, 0, R_PIPE + 0.22);
      return obj;
    }
    scene.add(endLabel("Inlet",  -L_PIPE / 2));
    scene.add(endLabel("Outlet",  L_PIPE / 2));

    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    };
    animate();

    const onResize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      labelRenderer.setSize(w, h);
    };
    window.addEventListener("resize", onResize);

    sceneRef.current = { renderer, scene, camera, controls, points: null };

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("resize", onResize);
      controls.dispose();
      renderer.dispose();
      if (el.contains(renderer.domElement))    el.removeChild(renderer.domElement);
      if (el.contains(labelRenderer.domElement)) el.removeChild(labelRenderer.domElement);
    };
  }, []);

  // ── Shared helper: swap Three.js point cloud from data ──────────────────
  const applyPointCloud = useCallback((data, cm, cr) => {
    const fr = fixedRangeRef.current;
    const { geom, vmin, vmax } = buildPointCloud(
      data, cm, cr,
      fr ? fr.vmin : null,
      fr ? fr.vmax : null,
    );
    const mat = new THREE.PointsMaterial({
      size: 0.035, vertexColors: true, sizeAttenuation: true,
      transparent: true, opacity: 0.85,
    });
    const pts = new THREE.Points(geom, mat);
    const sc = sceneRef.current;
    if (!sc) return;
    if (sc.points) {
      sc.scene.remove(sc.points);
      sc.points.geometry.dispose();
      sc.points.material.dispose();
    }
    sc.scene.add(pts);
    sc.points = pts;
    setVrange({ vmin, vmax });
  }, []);

  // ── Blur immediately when any param changes ──────────────────────────────
  useEffect(() => { setLoading(true); }, [Re, uIn, res, colorMode]);

  // ── Rebuild cloud instantly when clip radius changes (no fetch needed) ───
  useEffect(() => {
    if (!rawDataRef.current) return;
    applyPointCloud(rawDataRef.current, colorModeRef.current, clipR);
  }, [clipR, applyPointCloud]);

  // ── Rebuild cloud + heatmap when scale mode toggles (no fetch needed) ────
  useEffect(() => {
    if (!rawDataRef.current) return;
    applyPointCloud(rawDataRef.current, colorModeRef.current, clipRRef.current);
    setSliceData(prev => prev ? { ...prev } : prev);  // new ref → triggers heatmap D3
  }, [scaleMode, fixedRange, applyPointCloud]);

  const handleScaleMode = (mode) => {
    if (mode === "fixed") {
      const fr = { vmin: vrange.vmin, vmax: vrange.vmax };
      fixedRangeRef.current = fr;   // update ref immediately (before state batches)
      setFixedRange(fr);
    } else {
      fixedRangeRef.current = null;
      setFixedRange(null);
    }
    setScaleMode(mode);
  };

  // ── Fetch + update point cloud ───────────────────────────────────────────
  const fetchAndRender = useCallback(async (Re, uIn, res, colorMode) => {
    if (fetchRef.current) fetchRef.current.abort();
    const ctrl = new AbortController();
    fetchRef.current = ctrl;

    setLoading(true);
    setError(null);

    try {
      const url = `/api/predict?Re=${Re}&U_in=${uIn}&Ny=${res}&Nz=${res}`;
      const resp = await fetch(url, { signal: ctrl.signal });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok || data.error) throw new Error(data.error || `HTTP ${resp.status}`);

      rawDataRef.current = data;
      applyPointCloud(data, colorMode, clipRRef.current);

      setMeta(data.meta);
      setSliceData(data.slice);
      setLoading(false);
    } catch (err) {
      if (err.name !== "AbortError") {
        setError(err.message);
        setLoading(false);
      }
    }
  }, [applyPointCloud]);

  // Debounced trigger on parameter changes
  useEffect(() => {
    const t = setTimeout(() => fetchAndRender(Re, uIn, res, colorMode), 350);
    return () => clearTimeout(t);
  }, [Re, uIn, res, colorMode, fetchAndRender]);

  // ── Left-panel resize handlers ─────────────────────────────────────────────
  const handlePanelDragStart = useCallback((e) => {
    _panelDrag.current = { on: true, x0: e.clientX, w0: _panelW.current };
    document.body.style.cursor     = "ew-resize";
    document.body.style.userSelect = "none";
    e.preventDefault();
  }, []);

  useEffect(() => {
    const onMove = (e) => {
      if (!_panelDrag.current.on) return;
      const w = Math.max(200, Math.min(620, _panelDrag.current.w0 + e.clientX - _panelDrag.current.x0));
      setPanelWidth(w);
    };
    const onUp = () => {
      if (!_panelDrag.current.on) return;
      _panelDrag.current.on      = false;
      document.body.style.cursor     = "";
      document.body.style.userSelect = "";
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup",   onUp);
    return () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup",   onUp);
    };
  }, []);

  const colorLabel = colorMode === "vmag"
    ? "Velocity magnitude (m/s)"
    : "Pressure / U_in² (m²/s²)";

  // Re = U_in · D / ν  →  ν = U_in · D / Re,  D = 2·R_PIPE = 1 m
  const nu      = uIn / Re;                    // m²/s
  const nuCst   = nu / 1e-6;                   // centistokes (1 cSt = 1 mm²/s)
  const nuFluid = closestFluid(nu);
  const regime  = Re < 2300 ? "laminar" : Re < 4000 ? "transitional" : "turbulent";

  return (
    <div style={styles.root}>
      {/* ── Controls panel ── */}
      <div style={{ ...styles.panel, width: panelWidth, minWidth: panelWidth }}>
        <div style={styles.title}>CFD Explorer</div>
        <div style={styles.subtitle}>Fourier Neural Operator · 3D Steady-State Incompressible Flow</div>

        <div style={styles.section}>Physics</div>
        <Slider
          label="Reynolds number  Re"
          value={Re} min={100} max={1000} step={10}
          onChange={setRe}
          format={v => v.toFixed(0)}
        />
        <Slider
          label="Inlet velocity  U_in"
          value={uIn} min={0.1} max={1.0} step={0.01}
          onChange={setUIn}
          format={v => `${v.toFixed(2)} m/s`}
        />
        <div style={styles.nuCalcRow}>ν = U · D / Re</div>
        <div style={{...styles.nuCalcRow, paddingLeft: 12}}>= {uIn.toFixed(2)} × 1 / {Re}</div>
        <div style={{...styles.nuCalcRow, paddingLeft: 12}}>= {sciStr(nu)} m²/s</div>
        <div style={styles.nuInfo}>≈ {Math.round(nuCst).toLocaleString()} centistokes (cSt) — {nuFluid.name}</div>
        <div style={styles.nuSub}>{regime} flow</div>

        <div style={styles.section}>Grid</div>
        <Slider
          label="Resolution  Ny = Nz"
          value={res} min={9} max={51} step={2}
          onChange={setRes}
          format={v => `${v} pts`}
        />
        {meta && (
          <div style={styles.gridInfo}>
            Grid {meta.Nx} × {meta.Ny} × {meta.Nz} &nbsp;|&nbsp; {meta.n_points.toLocaleString()} fluid pts
          </div>
        )}

        <div style={styles.section}>View</div>
        <Slider
          label="Wall clip"
          value={clipR} min={0.1} max={1.0} step={0.05}
          onChange={setClipR}
          format={v => v === 1.0 ? "off" : `inner ${(v * 100).toFixed(0)}% of radius`}
        />

        <div style={styles.section}>Encode by</div>
        <div style={styles.toggleRow}>
          {["vmag", "pressure"].map(m => (
            <button
              key={m}
              onClick={() => setColorMode(m)}
              style={{ ...styles.toggleBtn, ...(colorMode === m ? styles.toggleBtnActive : {}) }}
            >
              {m === "vmag" ? "Velocity" : "Pressure"}
            </button>
          ))}
        </div>

        <div style={styles.section}>Color range</div>
        <div style={styles.toggleRow}>
          {[["auto", "Auto"], ["fixed", "Fixed"]].map(([m, label]) => (
            <button
              key={m}
              onClick={() => handleScaleMode(m)}
              style={{ ...styles.toggleBtn, ...(scaleMode === m ? styles.toggleBtnActive : {}) }}
            >
              {label}
            </button>
          ))}
        </div>
        {scaleMode === "fixed" && fixedRange && (
          <div style={styles.gridInfo}>
            Locked [{fixedRange.vmin.toFixed(3)}, {fixedRange.vmax.toFixed(3)}]
          </div>
        )}

        {meta && (
          <Colorbar
            vmin={vrange.vmin}
            vmax={vrange.vmax}
            label={colorLabel}
          />
        )}

        <div style={styles.hint}>
          Drag to orbit · Scroll to zoom · Ctrl+drag to pan
        </div>
      </div>

      {/* ── Panel resize handle ── */}
      <div
        className="panel-drag-handle"
        style={styles.panelDragHandle}
        onMouseDown={handlePanelDragStart}
      />

      {/* ── Canvas area (Three.js + heatmap overlay) ── */}
      <div style={styles.canvasArea}>
        <div ref={mountRef} style={{ position: "absolute", inset: 0 }} />
        <InverseToolPanel
          Re={Re} uIn={uIn} res={res}
          onVisualizeInverse={handleVisualizeInverse}
          onResetInverse={handleResetInverse}
        />

        {invSliceData && (
          <InverseHeatmap
            sliceData={invSliceData}
            rePred={invRePred}
            colorMode={colorMode}
            colorFn={colorFn}
            vmin={vrange.vmin}
            vmax={vrange.vmax}
            onClose={handleResetInverse}
          />
        )}

        {sliceData && (
          <Heatmap
            sliceData={sliceData}
            colorMode={colorMode}
            colorFn={colorFn}
            vmin={vrange.vmin}
            vmax={vrange.vmax}
            loading={loading}
            maxWidth={Math.max(480, canvasW - 28)}
          />
        )}

        {loading && (
          <div style={styles.overlay}>
            <div style={styles.spinner} />
            <div style={styles.overlayText}>Running inference…</div>
          </div>
        )}

        {error && (
          <div style={styles.errorBanner}>Error: {error}</div>
        )}
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = {
  root: {
    display: "flex",
    width: "100%",
    height: "100%",
    fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
    color: "#e2e8f0",
  },
  panelDragHandle: {
    width: 5,
    flexShrink: 0,
    cursor: "ew-resize",
    background: "rgba(255,255,255,0.04)",
    transition: "background 0.15s",
    zIndex: 20,
  },
  panel: {
    width: 390,
    minWidth: 390,
    background: "rgba(10,12,20,0.92)",
    backdropFilter: "blur(8px)",
    borderRight: "1px solid rgba(255,255,255,0.08)",
    padding: "20px 16px",
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: 4,
    zIndex: 10,
  },
  canvasArea: {
    flex: 1,
    position: "relative",
    overflow: "hidden",
  },
  title: {
    fontSize: 17,
    fontWeight: 700,
    letterSpacing: "0.02em",
    color: "#7dd3fc",
    marginBottom: 2,
  },
  subtitle: {
    fontSize: 11,
    color: "#94a3b8",
    marginBottom: 14,
  },
  section: {
    fontSize: 10,
    fontWeight: 600,
    letterSpacing: "0.12em",
    textTransform: "uppercase",
    color: "#64748b",
    marginTop: 14,
    marginBottom: 4,
  },
  sliderRow: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    marginBottom: 8,
  },
  sliderLabel: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: 12,
    color: "#94a3b8",
  },
  sliderValue: {
    color: "#e2e8f0",
    fontVariantNumeric: "tabular-nums",
    fontWeight: 600,
  },
  range: {
    width: "100%",
    accentColor: "#38bdf8",
    cursor: "pointer",
  },
  gridInfo: {
    fontSize: 11,
    color: "#64748b",
    marginTop: -4,
    marginBottom: 4,
  },
  nuCalcRow: {
    fontFamily: "monospace",
    fontSize: 11,
    color: "#94a3b8",
    lineHeight: 1.65,
  },
  nuInfo: {
    fontSize: 12,
    color: "#94a3b8",
    marginTop: 2,
    marginBottom: 1,
    fontVariantNumeric: "tabular-nums",
  },
  nuSub: {
    fontSize: 11,
    color: "#64748b",
    marginBottom: 6,
  },
  toggleRow: {
    display: "flex",
    gap: 6,
    marginBottom: 8,
  },
  toggleBtn: {
    flex: 1,
    padding: "5px 0",
    borderRadius: 6,
    border: "1px solid rgba(255,255,255,0.1)",
    background: "rgba(255,255,255,0.04)",
    color: "#64748b",
    fontSize: 12,
    cursor: "pointer",
    transition: "all 0.15s",
  },
  toggleBtnActive: {
    background: "rgba(56,189,248,0.15)",
    borderColor: "#38bdf8",
    color: "#38bdf8",
    fontWeight: 600,
  },
  colorbar: {
    marginTop: 6,
    marginBottom: 6,
  },
  colorbarLabel: {
    fontSize: 11,
    color: "#94a3b8",
    marginBottom: 4,
  },
  colorbarTrack: {
    display: "flex",
    flexDirection: "column",
    gap: 2,
  },
  colorbarGradient: {
    height: 12,
    borderRadius: 4,
  },
  colorbarTicks: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: 10,
    color: "#64748b",
    fontVariantNumeric: "tabular-nums",
  },
  hint: {
    marginTop: "auto",
    paddingTop: 16,
    fontSize: 13,
    color: "#64748b",
    lineHeight: 1.6,
  },
  overlay: {
    position: "absolute",
    inset: 0,
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    background: "rgba(10,10,15,0.6)",
    backdropFilter: "blur(2px)",
    zIndex: 5,
    gap: 12,
    pointerEvents: "none",
  },
  overlayText: {
    fontSize: 14,
    color: "#94a3b8",
    letterSpacing: "0.05em",
  },
  spinner: {
    width: 36,
    height: 36,
    border: "3px solid rgba(56,189,248,0.2)",
    borderTop: "3px solid #38bdf8",
    borderRadius: "50%",
    animation: "spin 0.9s linear infinite",
  },
  errorBanner: {
    position: "absolute",
    top: 12,
    left: "50%",
    transform: "translateX(-50%)",
    background: "rgba(220,38,38,0.9)",
    color: "#fff",
    padding: "8px 20px",
    borderRadius: 8,
    fontSize: 13,
    zIndex: 20,
  },
};

const styleTag = document.createElement("style");
styleTag.textContent = `
  @keyframes spin { to { transform: rotate(360deg); } }
  .panel-drag-handle:hover { background: rgba(56,189,248,0.35) !important; }
  .inv-drag-handle:hover   { background: rgba(56,189,248,0.35) !important; }
`;
document.head.appendChild(styleTag);

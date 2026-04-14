import { useCallback, useEffect, useRef, useState } from "react";
import * as d3 from "d3";

// ── Constants ──────────────────────────────────────────────────────────────────
const PIPE_H   = 80;   // canvas height (px) — 1 unit pipe diameter
const PIPE_W   = 800;  // canvas width (px)  — 10 unit pipe length (10:1 physical scale)
const PROF_MARGIN = { top: 20, right: 20, bottom: 48, left: 56 };
const PROF_H   = 360;  // svg total height
const CFD_COLOR  = "#2166ac";
const FNO_COLOR  = "#d73027";

const CASE_LABELS = {
  case_0500: "Re 371, U_in 0.91",
  case_0501: "Re 521, U_in 0.86",
  case_0502: "Re 516, U_in 0.26",
  case_0503: "Re 363, U_in 0.24",
  case_0504: "Re 819, U_in 0.79",
  case_0505: "Re 190, U_in 0.11",
};

// Sphere x-index range (from data analysis)
const SPHERE_XI_MIN = 53;
const SPHERE_XI_MAX = 61;

// ── Viridis LUT ───────────────────────────────────────────────────────────────
const VIRIDIS_STOPS = [
  [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
  [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
  [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
  [0.741, 0.873, 0.150], [0.993, 0.906, 0.144],
];
const VIRIDIS_LUT = (() => {
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t  = (i / 255) * (VIRIDIS_STOPS.length - 1);
    const lo = Math.floor(t), hi = Math.min(lo + 1, VIRIDIS_STOPS.length - 1);
    const f  = t - lo;
    lut[i * 3]     = Math.round((VIRIDIS_STOPS[lo][0] * (1 - f) + VIRIDIS_STOPS[hi][0] * f) * 255);
    lut[i * 3 + 1] = Math.round((VIRIDIS_STOPS[lo][1] * (1 - f) + VIRIDIS_STOPS[hi][1] * f) * 255);
    lut[i * 3 + 2] = Math.round((VIRIDIS_STOPS[lo][2] * (1 - f) + VIRIDIS_STOPS[hi][2] * f) * 255);
  }
  return lut;
})();

// ── Organise raw CSV rows into typed arrays ───────────────────────────────────
function organiseData(raw) {
  const xSet = new Set();
  const zSet = new Set();
  raw.forEach(r => { xSet.add(+r.x); zSet.add(+r.z); });

  const xVals = [...xSet].sort((a, b) => a - b);
  const zVals = [...zSet].sort((a, b) => a - b);
  const NX = xVals.length, NZ = zVals.length;
  const xIdx = new Map(xVals.map((v, i) => [v, i]));
  const zIdx = new Map(zVals.map((v, i) => [v, i]));

  const mask   = new Uint8Array(NX * NZ);
  const caseMap = {};

  // First pass: build per-case buffers
  raw.forEach(r => {
    const cid = r.case_id;
    if (!caseMap[cid]) {
      caseMap[cid] = {
        meta:   { re: +r.Re, uin: +r.U_in },
        actual: new Float32Array(NX * NZ),
        pred:   new Float32Array(NX * NZ),
      };
    }
    const xi  = xIdx.get(+r.x);
    const zi  = zIdx.get(+r.z);
    if (xi === undefined || zi === undefined) return;
    const idx = xi * NZ + zi;
    caseMap[cid].actual[idx] = +r.ux_actual;
    caseMap[cid].pred[idx]   = +r.ux_pred;
    mask[idx] = +r.mask;
  });

  return { xVals, zVals, NX, NZ, mask, cases: caseMap };
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function VelocityProfilePanel() {
  const containerRef = useRef(null);
  const canvasRef    = useRef(null);
  const cbCanvasRef  = useRef(null);  // colorbar
  const svgRef       = useRef(null);
  const dataRef      = useRef(null);  // organised data (no re-render on write)

  const [ready, setReady]           = useState(false);
  const [error, setError]           = useState(null);
  const [currentCase, setCurrentCase] = useState("case_0500");
  const [currentXi, setCurrentXi]   = useState(0);
  const [infoX, setInfoX]           = useState(0);
  const [infoRegion, setInfoRegion] = useState("Upstream");

  // ── Load CSV once ────────────────────────────────────────────────────────────
  useEffect(() => {
    d3.csv("/data/fno_vs_cfd_all_cases.csv").then(raw => {
      dataRef.current = organiseData(raw);
      setReady(true);
    }).catch(err => setError(err.message));
  }, []);

  // ── Pipe canvas draw ─────────────────────────────────────────────────────────
  const drawPipe = useCallback(() => {
    const d = dataRef.current;
    if (!d || !canvasRef.current || !containerRef.current) return;
    const { NX, NZ, mask, cases } = d;
    const caseData = cases[currentCase];

    const W = PIPE_W;
    canvasRef.current.width  = W;
    canvasRef.current.height = PIPE_H;
    const ctx = canvasRef.current.getContext("2d");
    const cellW = W / NX;
    const cellH = PIPE_H / NZ;

    const imgData = ctx.createImageData(W, PIPE_H);
    const buf = imgData.data;

    for (let xi = 0; xi < NX; xi++) {
      for (let zi = 0; zi < NZ; zi++) {
        const idx  = xi * NZ + zi;
        const m    = mask[idx];
        const isWall = zi === 0 || zi === NZ - 1;

        let r, g, b;
        if (m === 0) {
          // wall or sphere
          [r, g, b] = isWall ? [15, 23, 42] : [55, 65, 81];
        } else {
          const t  = Math.max(0, Math.min(1, caseData.actual[idx] / 2.0));
          const li = Math.round(t * 255) * 3;
          r = VIRIDIS_LUT[li]; g = VIRIDIS_LUT[li + 1]; b = VIRIDIS_LUT[li + 2];
        }

        // Fill pixel rectangle for this cell
        const px0 = Math.round(xi * cellW);
        const px1 = Math.round((xi + 1) * cellW);
        const py0 = Math.round(zi * cellH);
        const py1 = Math.round((zi + 1) * cellH);
        for (let py = py0; py < py1; py++) {
          for (let px = px0; px < px1; px++) {
            const pi = (py * W + px) * 4;
            buf[pi] = r; buf[pi+1] = g; buf[pi+2] = b; buf[pi+3] = 255;
          }
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Sphere outline
    const sx0 = Math.round(SPHERE_XI_MIN * cellW);
    const sx1 = Math.round((SPHERE_XI_MAX + 1) * cellW);
    // Find z extent of sphere at center xi
    const cxi = Math.round((SPHERE_XI_MIN + SPHERE_XI_MAX) / 2);
    let szMin = NZ, szMax = 0;
    for (let zi = 1; zi < NZ - 1; zi++) {
      if (mask[cxi * NZ + zi] === 0) { szMin = Math.min(szMin, zi); szMax = Math.max(szMax, zi); }
    }
    if (szMin <= szMax) {
      ctx.strokeStyle = "#f59e0b";
      ctx.lineWidth   = 1.5;
      ctx.strokeRect(sx0, Math.round(szMin * cellH), sx1 - sx0, Math.round((szMax - szMin + 1) * cellH));
    }

    // Current x marker
    const lx = Math.round((currentXi + 0.5) * cellW);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth   = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(lx, 0); ctx.lineTo(lx, PIPE_H); ctx.stroke();
    ctx.setLineDash([]);
  }, [currentCase, currentXi]);

  // ── Colorbar draw ────────────────────────────────────────────────────────────
  const drawColorbar = useCallback(() => {
    if (!cbCanvasRef.current || !containerRef.current) return;
    const W = Math.max(1, containerRef.current.clientWidth - 60);
    cbCanvasRef.current.width  = W;
    cbCanvasRef.current.height = 10;
    const ctx = cbCanvasRef.current.getContext("2d");
    const img = ctx.createImageData(W, 10);
    for (let i = 0; i < W; i++) {
      const li = Math.round((i / W) * 255) * 3;
      const r = VIRIDIS_LUT[li], g = VIRIDIS_LUT[li+1], b = VIRIDIS_LUT[li+2];
      for (let row = 0; row < 10; row++) {
        const pi = (row * W + i) * 4;
        img.data[pi] = r; img.data[pi+1] = g; img.data[pi+2] = b; img.data[pi+3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, []);

  // ── Profile SVG draw ─────────────────────────────────────────────────────────
  const drawProfile = useCallback(() => {
    const d = dataRef.current;
    if (!d || !svgRef.current || !containerRef.current) return;
    const { zVals, NZ, mask, cases } = d;
    const caseData = cases[currentCase];
    const xi = currentXi;

    const totalW = containerRef.current.clientWidth;
    const iW = totalW - PROF_MARGIN.left - PROF_MARGIN.right;
    const iH = PROF_H - PROF_MARGIN.top  - PROF_MARGIN.bottom;

    d3.select(svgRef.current).selectAll("*").remove();
    const svg = d3.select(svgRef.current).attr("width", totalW).attr("height", PROF_H);
    const g   = svg.append("g").attr("transform", `translate(${PROF_MARGIN.left},${PROF_MARGIN.top})`);

    const xSc = d3.scaleLinear().domain([-0.5, 0.5]).range([0, iW]);
    const ySc = d3.scaleLinear().domain([-0.1, 2.2]).range([iH, 0]);

    // Grid
    g.append("g")
      .call(d3.axisLeft(ySc).ticks(5).tickSize(-iW).tickFormat(""))
      .call(ax => { ax.select(".domain").remove(); ax.selectAll("line").attr("stroke", "rgba(255,255,255,0.07)"); });
    g.append("g").attr("transform", `translate(0,${iH})`)
      .call(d3.axisBottom(xSc).ticks(6).tickSize(-iH).tickFormat(""))
      .call(ax => { ax.select(".domain").remove(); ax.selectAll("line").attr("stroke", "rgba(255,255,255,0.07)"); });

    // Build slice
    const slice = zVals.map((z, zi) => ({
      z,
      a: caseData.actual[xi * NZ + zi],
      p: caseData.pred[xi * NZ + zi],
      m: mask[xi * NZ + zi],
    }));

    // Sphere/wall shade bands
    let inBand = false, bandStart = null;
    slice.forEach((pt, i) => {
      const blocked = pt.m === 0 && i !== 0 && i !== NZ - 1;
      if (blocked && !inBand) { inBand = true; bandStart = pt.z; }
      if (!blocked && inBand) {
        g.append("rect")
          .attr("x", xSc(bandStart)).attr("y", 0)
          .attr("width", Math.abs(xSc(slice[i - 1].z) - xSc(bandStart)))
          .attr("height", iH)
          .attr("fill", "#334155").attr("opacity", 0.65);
        inBand = false; bandStart = null;
      }
    });

    // Split fluid into contiguous segments (splits at sphere)
    const lineActual = d3.line().x(pt => xSc(pt.z)).y(pt => ySc(pt.a)).curve(d3.curveMonotoneX);
    const linePred   = d3.line().x(pt => xSc(pt.z)).y(pt => ySc(pt.p)).curve(d3.curveMonotoneX);

    let seg = [];
    const flush = () => {
      if (seg.length > 1) {
        g.append("path").datum(seg).attr("fill","none").attr("stroke", CFD_COLOR).attr("stroke-width", 2.5).attr("d", lineActual);
        g.append("path").datum(seg).attr("fill","none").attr("stroke", FNO_COLOR).attr("stroke-width", 2.5).attr("stroke-dasharray","6,4").attr("d", linePred);
      }
      seg = [];
    };
    slice.forEach(pt => { if (pt.m === 1) seg.push(pt); else flush(); });
    flush();

    // Axes
    g.append("g").attr("transform", `translate(0,${iH})`)
      .call(d3.axisBottom(xSc).ticks(6))
      .call(ax => { ax.select(".domain").attr("stroke","#475569"); ax.selectAll("text").attr("fill","#94a3b8").attr("font-size","13px"); ax.selectAll("line").attr("stroke","#475569"); });
    g.append("g")
      .call(d3.axisLeft(ySc).ticks(5))
      .call(ax => { ax.select(".domain").attr("stroke","#475569"); ax.selectAll("text").attr("fill","#94a3b8").attr("font-size","13px"); ax.selectAll("line").attr("stroke","#475569"); });

    // Axis labels
    svg.append("text")
      .attr("x", PROF_MARGIN.left + iW / 2).attr("y", PROF_H - 8)
      .attr("text-anchor","middle").attr("fill","#94a3b8").attr("font-size","14px")
      .text("z");
    svg.append("text")
      .attr("transform", `translate(14,${PROF_MARGIN.top + iH / 2}) rotate(-90)`)
      .attr("text-anchor","middle").attr("fill","#94a3b8").attr("font-size","14px")
      .text("ũₓ");

    // Wall labels
    g.append("text").attr("x", xSc(-0.5) + 3).attr("y", iH - 4).attr("fill","#475569").attr("font-size","10px").text("wall");
    g.append("text").attr("x", xSc(0.5)  - 3).attr("y", iH - 4).attr("text-anchor","end").attr("fill","#475569").attr("font-size","10px").text("wall");

  }, [currentCase, currentXi]);

  // ── Update info strip ────────────────────────────────────────────────────────
  useEffect(() => {
    const d = dataRef.current;
    if (!d) return;
    const x = d.xVals[currentXi];
    setInfoX(x?.toFixed(3) ?? "—");
    const SPHERE_X_MIN = d.xVals[SPHERE_XI_MIN];
    const SPHERE_X_MAX = d.xVals[SPHERE_XI_MAX];
    let region = "Upstream";
    if (x >= SPHERE_X_MIN - 0.5 && x < SPHERE_X_MIN) region = "Approaching sphere";
    else if (x >= SPHERE_X_MIN && x <= SPHERE_X_MAX)  region = "🔴 Inside sphere obstacle";
    else if (x > SPHERE_X_MAX && x <= SPHERE_X_MAX + 1.5) region = "Wake / recovery";
    else if (x > SPHERE_X_MAX + 1.5) region = "Downstream";
    setInfoRegion(region);
  }, [currentXi]);

  // ── Full redraw when case/xi/ready changes ───────────────────────────────────
  useEffect(() => {
    if (!ready) return;
    drawPipe();
    drawColorbar();
    drawProfile();
  }, [ready, drawPipe, drawColorbar, drawProfile]);

  // ── Resize observer ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || !ready) return;
    const ro = new ResizeObserver(() => {
      drawPipe(); drawColorbar(); drawProfile();
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [ready, drawPipe, drawColorbar, drawProfile]);

  // ── Pipe drag interaction ────────────────────────────────────────────────────
  const handlePipeEvent = useCallback(e => {
    if (!dataRef.current || !canvasRef.current) return;
    if (e.type === "mousemove" && e.buttons !== 1) return;
    const rect   = canvasRef.current.getBoundingClientRect();
    const relX   = (e.clientX ?? e.touches?.[0]?.clientX ?? 0) - rect.left;
    const xi     = Math.max(0, Math.min(dataRef.current.NX - 1, Math.round(relX / rect.width * dataRef.current.NX)));
    setCurrentXi(xi);
  }, []);

  if (error) return (
    <div className="glass" style={{ padding: "1.5rem", color: "#f87171" }}>
      Failed to load data: {error}
    </div>
  );

  if (!ready) return (
    <div className="glass" style={{ padding: "2rem", textAlign: "center", color: "#94a3b8" }}>
      Loading velocity data…
    </div>
  );

  const meta = dataRef.current?.cases[currentCase]?.meta;

  return (
    <div className="glass" ref={containerRef} style={{ padding: "1rem 1.25rem", display: "flex", flexDirection: "column", gap: "0.75rem", overflow: "hidden" }}>

      {/* Case selector */}
      <div style={{ display: "flex", gap: "0.4rem", flexWrap: "wrap", alignItems: "center" }}>
        {Object.entries(CASE_LABELS).map(([cid, lbl]) => (
          <button
            key={cid}
            onClick={() => setCurrentCase(cid)}
            style={{
              background: currentCase === cid ? "rgba(56,189,248,0.1)" : "#111827",
              border: `1px solid ${currentCase === cid ? "#38bdf8" : "#334155"}`,
              color: currentCase === cid ? "#38bdf8" : "#94a3b8",
              fontWeight: currentCase === cid ? 600 : 400,
              fontSize: "0.8rem", padding: "3px 10px", borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            {lbl}
          </button>
        ))}
      </div>

      {/* Info strip */}
      <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", fontSize: "0.82rem", borderTop: "1px solid rgba(255,255,255,0.07)", borderBottom: "1px solid rgba(255,255,255,0.07)", padding: "0.4rem 0" }}>
        {[
          ["Re",      meta?.re],
          ["U_in",    meta ? `${meta.uin} m/s` : "—"],
          ["x",       infoX],
          ["Region",  infoRegion],
        ].map(([label, val]) => (
          <div key={label}>
            <div style={{ color: "#64748b", fontSize: "0.7rem", marginBottom: 1 }}>{label}</div>
            <div style={{ color: label === "x" ? "#38bdf8" : "#e2e8f0", fontWeight: 600 }}>{val}</div>
          </div>
        ))}
      </div>

      {/* Pipe canvas — fixed physical scale, centered */}
      <div style={{ overflowX: "auto" }}>
        <div style={{ width: PIPE_W, margin: "0 auto" }}>
          <canvas
            ref={canvasRef}
            style={{ display: "block", width: PIPE_W, height: PIPE_H, borderRadius: 4, border: "1px solid rgba(255,255,255,0.07)", cursor: "crosshair" }}
            onClick={handlePipeEvent}
            onMouseMove={handlePipeEvent}
          />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", color: "#475569", marginTop: 2 }}>
            <span>x = 0 (inlet)</span>
            <span>x = 5</span>
            <span>x = 10 (outlet)</span>
          </div>
        </div>
      </div>

      {/* Colorbar */}
      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <span style={{ fontSize: "0.72rem", color: "#64748b", whiteSpace: "nowrap" }}>ũₓ</span>
        <div style={{ flex: 1 }}>
          <canvas ref={cbCanvasRef} height="10" style={{ display: "block", width: "100%", borderRadius: 2 }} />
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.68rem", color: "#64748b", marginTop: 2 }}>
            <span>0</span><span>0.5</span><span>1.0</span><span>1.5</span><span>≥2.0</span>
          </div>
        </div>
      </div>

      {/* Profile SVG */}
      <svg ref={svgRef} style={{ display: "block", width: "100%" }} />

      {/* Legend */}
      <div style={{ display: "flex", gap: "1.5rem", justifyContent: "center", flexWrap: "wrap", fontSize: "0.82rem", color: "#94a3b8" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <svg width="28" height="10"><line x1="0" y1="5" x2="28" y2="5" stroke={CFD_COLOR} strokeWidth="2.5"/></svg>
          CFD actual
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <svg width="28" height="10"><line x1="0" y1="5" x2="28" y2="5" stroke={FNO_COLOR} strokeWidth="2.5" strokeDasharray="6,4"/></svg>
          FNO predicted
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <svg width="12" height="10"><rect width="12" height="10" fill="#334155" opacity="0.8" rx="1"/></svg>
          Sphere / wall
        </div>
      </div>
    </div>
  );
}

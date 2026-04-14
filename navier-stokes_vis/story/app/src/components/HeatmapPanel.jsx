import { useEffect, useRef, useState, useCallback } from "react";
import * as d3 from "d3";

// ── Physics constants ─────────────────────────────────────────────────────────
const CYL_X  = 3.0;
const CYL_R  = 0.25;
const R_PIPE = 0.5;

const FIELD_LABELS = ["ux [m/s]", "uy [m/s]", "uz [m/s]", "p/\u03c1 [m\u00b2/s\u00b2]"];
const ROW_LABELS   = ["Actual", "Predicted", "|Error|"];
const FIELD_KEYS   = ["ux", "uy", "uz", "p"];

// ── Colormaps ──────────────────────────────────────
const CMAPS = {
  viridis: t => d3.interpolateViridis(t),
  RdBu_r:  t => d3.interpolateRdBu(1 - t),
  hot_r: t => {
    const s = 1 - t;
    let r, g, b;
    if      (s < 1/3) { r = Math.round(255 * s * 3);          g = 0;                         b = 0; }
    else if (s < 2/3) { r = 255;                               g = Math.round(255*(s-1/3)*3); b = 0; }
    else              { r = 255;                               g = 255;                        b = Math.round(255*(s-2/3)*3); }
    return `rgb(${r},${g},${b})`;
  },
};

function makeDataURL(values, Nx, Nz, vmin, vmax, cmapName) {
  const canvas = document.createElement("canvas");
  canvas.width  = Nx;
  canvas.height = Nz;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(Nx, Nz);
  const fn  = CMAPS[cmapName];
  const rng = Math.max(vmax - vmin, 1e-12);

  for (let ix = 0; ix < Nx; ix++) {
    for (let iz = 0; iz < Nz; iz++) {
      const v   = values[ix * Nz + iz];
      const izc = Nz - 1 - iz;          // flip z for display
      const p   = (izc * Nx + ix) * 4;
      if (v == null || isNaN(v)) {
        img.data[p] = img.data[p+1] = img.data[p+2] = 32; img.data[p+3] = 255;
      } else {
        const t = Math.max(0, Math.min(1, (v - vmin) / rng));
        const c = d3.color(fn(t));
        img.data[p] = c.r; img.data[p+1] = c.g; img.data[p+2] = c.b; img.data[p+3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL();
}

// ── Parse CSV into per-case 2D arrays ─────────────────────────────────────────
function parseCases(raw) {
  const caseMap = new Map();

  for (const row of raw) {
    const cid = row.case_id;
    if (!caseMap.has(cid)) {
      caseMap.set(cid, {
        Re:    +row.Re,
        U_in:  +row.U_in,
        iy:    +row.iy,
        y_val: +row.y_val,
        pts:   [],
      });
    }
    caseMap.get(cid).pts.push({
      x:    +row.x,    z:    +row.z,
      uxa:  +row.ux_actual, uya: +row.uy_actual,
      uza:  +row.uz_actual, pa:  +row.p_actual,
      uxp:  +row.ux_pred,   uyp: +row.uy_pred,
      uzp:  +row.uz_pred,   pp:  +row.p_pred,
      mask: +row.mask,
    });
  }

  const result = new Map();
  for (const [cid, data] of caseMap) {
    const pts  = data.pts;
    const xSet = [...new Set(pts.map(p => p.x))].sort((a,b) => a-b);
    const zSet = [...new Set(pts.map(p => p.z))].sort((a,b) => a-b);
    const Nx   = xSet.length;
    const Nz   = zSet.length;
    const xi   = new Map(xSet.map((v, i) => [v, i]));
    const zi   = new Map(zSet.map((v, i) => [v, i]));

    const fields = { ux: {}, uy: {}, uz: {}, p: {} };
    for (const fn of FIELD_KEYS) {
      fields[fn].actual = new Float32Array(Nx * Nz).fill(NaN);
      fields[fn].pred   = new Float32Array(Nx * Nz).fill(NaN);
      fields[fn].err    = new Float32Array(Nx * Nz).fill(NaN);
    }

    for (const pt of pts) {
      const ix = xi.get(pt.x);
      const iz = zi.get(pt.z);
      if (pt.mask === 0) continue; // out-of-fluid: keep NaN
      const i = ix * Nz + iz;
      fields.ux.actual[i] = pt.uxa; fields.ux.pred[i] = pt.uxp;
      fields.uy.actual[i] = pt.uya; fields.uy.pred[i] = pt.uyp;
      fields.uz.actual[i] = pt.uza; fields.uz.pred[i] = pt.uzp;
      fields.p.actual[i]  = pt.pa;  fields.p.pred[i]  = pt.pp;
      fields.ux.err[i] = Math.abs(pt.uxp - pt.uxa);
      fields.uy.err[i] = Math.abs(pt.uyp - pt.uya);
      fields.uz.err[i] = Math.abs(pt.uzp - pt.uza);
      fields.p.err[i]  = Math.abs(pt.pp  - pt.pa);
    }

    result.set(cid, {
      Re: data.Re, U_in: data.U_in,
      iy: data.iy, y_val: data.y_val,
      xVals: xSet, zVals: zSet, Nx, Nz,
      fields,
    });
  }
  return result;
}

// ── Main component ────────────────────────────────────────────────────────────
export default function HeatmapPanel() {
  const svgRef      = useRef(null);
  const containerRef = useRef(null);
  const [cases, setCases]         = useState(null);
  const [activeCase, setActiveCase] = useState(null);
  const [error, setError]           = useState(null);
  const [loading, setLoading]       = useState(true);

  useEffect(() => {
    setLoading(true);
    d3.csv("/data/fno_vs_cfd_all_cases.csv").then(raw => {
      const parsed = parseCases(raw);
      setCases(parsed);
      setActiveCase([...parsed.keys()][0]);
      setLoading(false);
    }).catch(err => { setError(err.message); setLoading(false); });
  }, []);

  const draw = useCallback(() => {
    if (!cases || !activeCase || !svgRef.current || !containerRef.current) return;

    const caseData = cases.get(activeCase);
    if (!caseData) return;

    const { xVals, zVals, Nx, Nz, fields, Re, U_in, y_val, iy } = caseData;
    const containerW = containerRef.current.clientWidth || 900;

    // Layout config cols 1-3 use minimal margin
    const cfg = {
      outerPadL: 8, outerPadR: 16, outerPadT: 8, outerPadB: 16,
      titleH: 52,
      rowLabelW: 80,
      marginL0: 50, marginL: 6, marginR: 4, marginT: 4, marginB: 26,
      colLabelH: 22,
      cbW: 10, cbGap: 4, cbTickW: 40,
      colGap: 10, rowGap: 8,
      nRows: 3, nCols: 4,
    };

    // Compute plot width from available space (col 0 uses marginL0; cols 1-3 use marginL)
    const usableW = containerW - cfg.outerPadL - cfg.outerPadR - cfg.rowLabelW
                  - cfg.marginL0 - (cfg.nCols - 1) * cfg.marginL
                  - cfg.nCols * (cfg.marginR + cfg.cbGap + cfg.cbW + cfg.cbTickW)
                  - (cfg.nCols - 1) * cfg.colGap;
    const plotW = Math.max(Math.floor(usableW / cfg.nCols), 60);
    const physAsp = (xVals[xVals.length-1] - xVals[0]) / Math.max(zVals[zVals.length-1] - zVals[0], 1e-6);
    const plotH = Math.max(Math.round(plotW / physAsp), 20);

    const panelW0 = cfg.marginL0 + plotW + cfg.marginR + cfg.cbGap + cfg.cbW + cfg.cbTickW;
    const panelW  = cfg.marginL  + plotW + cfg.marginR + cfg.cbGap + cfg.cbW + cfg.cbTickW;
    const panelH  = cfg.colLabelH + cfg.marginT + plotH + cfg.marginB;
    const svgW    = cfg.outerPadL + cfg.rowLabelW + panelW0 + (cfg.nCols-1)*panelW + (cfg.nCols-1)*cfg.colGap + cfg.outerPadR;
    const svgH   = cfg.outerPadT + cfg.titleH + cfg.nRows * panelH + (cfg.nRows-1)*cfg.rowGap + cfg.outerPadB;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("width", svgW).attr("height", svgH)
      .style("background", "transparent");

    // Title
    svg.append("text")
      .attr("x", svgW/2).attr("y", cfg.outerPadT + 22)
      .attr("text-anchor", "middle").attr("fill", "#f1f5f9")
      .attr("font-size", "15px").attr("font-weight", "600")
      .text(`CFD vs FNO  |  Re = ${Re.toFixed(1)},  U_in = ${U_in.toFixed(3)} m/s`);

    let gradCount = 0;

    // Build panels
    for (let row = 0; row < cfg.nRows; row++) {
      for (let col = 0; col < cfg.nCols; col++) {
        const fname = FIELD_KEYS[col];
        const fData = fields[fname];

        // Determine values, vmin/vmax, colormap
        let arr, vmin, vmax, cmapName;
        if (row === 0) {
          arr      = fData.actual;
          const a  = [...fData.actual].filter(v => !isNaN(v));
          const b  = [...fData.pred  ].filter(v => !isNaN(v));
          vmin     = Math.min(...a, ...b);
          vmax     = Math.max(...a, ...b);
          if (fname !== "ux") { const m = Math.max(Math.abs(vmin), Math.abs(vmax)); vmin = -m; vmax = m; }
          cmapName = fname === "ux" ? "viridis" : "RdBu_r";
        } else if (row === 1) {
          arr      = fData.pred;
          const a  = [...fData.actual].filter(v => !isNaN(v));
          const b  = [...fData.pred  ].filter(v => !isNaN(v));
          vmin     = Math.min(...a, ...b);
          vmax     = Math.max(...a, ...b);
          if (fname !== "ux") { const m = Math.max(Math.abs(vmin), Math.abs(vmax)); vmin = -m; vmax = m; }
          cmapName = fname === "ux" ? "viridis" : "RdBu_r";
        } else {
          arr      = fData.err;
          const e  = [...fData.err].filter(v => !isNaN(v));
          vmin     = 0;
          vmax     = e.length ? Math.max(...e) : 1e-12;
          cmapName = "hot_r";
        }

        const url   = makeDataURL(arr, Nx, Nz, vmin, vmax, cmapName);
        const ox    = cfg.outerPadL + cfg.rowLabelW +
                      (col === 0 ? 0 : panelW0 + (col - 1) * (panelW + cfg.colGap) + cfg.colGap);
        const oy    = cfg.outerPadT + cfg.titleH    + row * (panelH + cfg.rowGap);
        const plotX = ox + (col === 0 ? cfg.marginL0 : cfg.marginL);
        const plotY = oy + cfg.colLabelH + cfg.marginT;

        const xSc = d3.scaleLinear().domain([xVals[0], xVals[xVals.length-1]]).range([plotX, plotX + plotW]);
        const zSc = d3.scaleLinear().domain([zVals[0], zVals[zVals.length-1]]).range([plotY + plotH, plotY]);

        // Heatmap image
        svg.append("image")
          .attr("x", plotX).attr("y", plotY)
          .attr("width", plotW).attr("height", plotH)
          .attr("preserveAspectRatio", "none")
          .attr("href", url);
        svg.append("rect")
          .attr("x", plotX).attr("y", plotY)
          .attr("width", plotW).attr("height", plotH)
          .attr("fill", "none").attr("stroke", "#334155").attr("stroke-width", 0.5);

        // Cylinder ellipse
        svg.append("ellipse")
          .attr("cx", xSc(CYL_X)).attr("cy", zSc(0))
          .attr("rx", (CYL_R / (xVals[xVals.length-1] - xVals[0])) * plotW)
          .attr("ry", (CYL_R / (zVals[zVals.length-1] - zVals[0])) * plotH)
          .attr("fill", "none").attr("stroke", "rgba(255,255,255,0.7)")
          .attr("stroke-width", 1.2).attr("stroke-dasharray", "4,3");

        // Pipe walls
        for (const z of [R_PIPE, -R_PIPE]) {
          const yp = zSc(z);
          svg.append("line")
            .attr("x1", plotX).attr("x2", plotX + plotW).attr("y1", yp).attr("y2", yp)
            .attr("stroke", "rgba(148,163,184,0.5)").attr("stroke-width", 0.8)
            .attr("stroke-dasharray", "3,3");
        }

        // X-axis (bottom)
        svg.append("g").attr("transform", `translate(0,${plotY + plotH})`)
          .call(d3.axisBottom(xSc).ticks(4).tickSize(3))
          .call(ax => {
            ax.select(".domain").attr("stroke", "#475569");
            ax.selectAll("text").attr("fill", "#64748b").attr("font-size", "10px");
            ax.selectAll("line").attr("stroke", "#475569");
          });
        svg.append("text")
          .attr("x", plotX + plotW/2).attr("y", plotY + plotH + cfg.marginB - 2)
          .attr("text-anchor", "middle").attr("fill", "#64748b").attr("font-size", "10px")
          .text("x [m]");

        // Z-axis (left, col 0 only)
        if (col === 0) {
          svg.append("g").attr("transform", `translate(${plotX},0)`)
            .call(d3.axisLeft(zSc).tickValues([zVals[0], zVals[zVals.length-1]]).tickSize(3))
            .call(ax => {
              ax.select(".domain").attr("stroke", "#475569");
              ax.selectAll("text").attr("fill", "#64748b").attr("font-size", "10px");
              ax.selectAll("line").attr("stroke", "#475569");
            });

          // Row label
          svg.append("text")
            .attr("transform", `translate(${cfg.outerPadL + 12},${plotY + plotH/2}) rotate(-90)`)
            .attr("text-anchor", "middle").attr("fill", "#94a3b8")
            .attr("font-size", "12px").attr("font-weight", "600")
            .text(ROW_LABELS[row]);

          // z [m] label
          svg.append("text")
            .attr("transform", `translate(${cfg.outerPadL + 28},${plotY + plotH/2}) rotate(-90)`)
            .attr("text-anchor", "middle").attr("fill", "#64748b").attr("font-size", "10px")
            .text("z [m]");
        }

        // Column label (row 0 only)
        if (row === 0) {
          svg.append("text")
            .attr("x", plotX + plotW/2).attr("y", oy + cfg.colLabelH - 2)
            .attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", "12px")
            .text(FIELD_LABELS[col]);
        }

        // Colorbar
        const cbX  = plotX + plotW + cfg.cbGap;
        const gId  = `cb${gradCount++}`;
        const grad = svg.append("defs").append("linearGradient")
          .attr("id", gId).attr("x1", "0%").attr("x2", "0%").attr("y1", "100%").attr("y2", "0%");
        for (let i = 0; i <= 20; i++) {
          const t = i / 20;
          grad.append("stop").attr("offset", `${(t*100).toFixed(0)}%`)
            .attr("stop-color", CMAPS[cmapName](t));
        }
        svg.append("rect")
          .attr("x", cbX).attr("y", plotY).attr("width", cfg.cbW).attr("height", plotH)
          .attr("fill", `url(#${gId})`);
        svg.append("rect")
          .attr("x", cbX).attr("y", plotY).attr("width", cfg.cbW).attr("height", plotH)
          .attr("fill", "none").attr("stroke", "#334155").attr("stroke-width", 0.5);
        svg.append("g")
          .attr("transform", `translate(${cbX + cfg.cbW},0)`)
          .call(d3.axisRight(d3.scaleLinear().domain([vmin, vmax]).range([plotY + plotH, plotY]))
            .ticks(3).tickSize(3).tickFormat(d3.format(".2g")))
          .call(ax => {
            ax.select(".domain").remove();
            ax.selectAll("text").attr("fill", "#64748b").attr("font-size", "10px");
            ax.selectAll("line").attr("stroke", "#475569");
          });
      }
    }
  }, [cases, activeCase]);

  useEffect(() => { draw(); }, [draw]);

  // Resize observer
  useEffect(() => {
    if (!containerRef.current || !cases) return;
    const ro = new ResizeObserver(() => { draw(); });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [cases, draw]);

  if (error) return (
    <div className="glass" style={{ padding: "1.5rem", color: "#f87171" }}>
      Failed to load heatmap data: {error}
    </div>
  );

  const caseIds = cases ? [...cases.keys()] : [];

  return (
    <div className="glass" style={{ padding: "1rem" }}>
      {/* Case toggle buttons */}
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.75rem" }}>
        {caseIds.map(cid => {
          const c    = cases.get(cid);
          const label = `Re ${c.Re.toFixed(0)}, U_in ${c.U_in.toFixed(2)}`;
          const active = cid === activeCase;
          return (
            <button key={cid} onClick={() => setActiveCase(cid)}
              style={{
                padding: "3px 10px", borderRadius: "4px", fontSize: "0.8rem", cursor: "pointer",
                background: active ? "rgba(56,189,248,0.1)" : "#111827",
                border: `1px solid ${active ? "#38bdf8" : "#334155"}`,
                color: active ? "#38bdf8" : "#94a3b8",
                fontWeight: active ? 600 : 400,
              }}>
              {label}
            </button>
          );
        })}
      </div>

      {/* SVG container */}
      {loading && (
        <div style={{ color: "#94a3b8", padding: "2rem", textAlign: "center" }}>
          Loading heatmap data (~5 MB)...
        </div>
      )}
      <div ref={containerRef} style={{ width: "100%", display: loading ? "none" : "block", overflowX: "auto" }}>
        <svg ref={svgRef} style={{ display: "block" }} />
      </div>
    </div>
  );
}

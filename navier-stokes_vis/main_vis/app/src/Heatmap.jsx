import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

const L_PIPE = 10.0;
const R_PIPE = 0.5;
const CYL_X  = 3.0;
const CYL_R  = 0.25;

// 1.5× scale: 108 px per metre on both axes → true 10:1 physical aspect
const PX_PER_M = 108;
const IW = PX_PER_M * 10;   // 1080 px  (x: 0 → 10 m)
const IH = PX_PER_M * 1;    //  108 px  (z: −0.5 → 0.5 m)
const M  = { top: 38, right: 20, bottom: 54, left: 62 };
const W  = IW + M.left + M.right;   // 1162
const H  = IH + M.top  + M.bottom;  //  200

const MIN_W = 480;
const MAX_W = 3200;

export default function Heatmap({ sliceData, colorMode, colorFn, vmin, vmax, loading, maxWidth = MAX_W }) {
  const svgRef = useRef(null);
  const [displayW, setDisplayW] = useState(() => Math.min(W, maxWidth));
  const displayH = Math.round(displayW * H / W);
  const dragStartRef = useRef(null);

  // Clamp current width whenever the available canvas space shrinks
  useEffect(() => {
    setDisplayW(w => Math.min(w, maxWidth));
  }, [maxWidth]);

  // Keep latest colorMode/vmin/vmax in refs so D3 can read them without
  // being in the effect dependency array — prevents re-renders on every
  // toggle/slider change before new sliceData has actually arrived.
  const colorModeRef = useRef(colorMode);
  colorModeRef.current = colorMode;
  const vminRef = useRef(vmin);
  vminRef.current = vmin;
  const vmaxRef = useRef(vmax);
  vmaxRef.current = vmax;

  // ── Drag-to-resize ─────────────────────────────────────────────────────────
  const onHandleMouseDown = (e) => {
    e.preventDefault();
    dragStartRef.current = { startX: e.clientX, startW: displayW };

    const onMove = (ev) => {
      const { startX, startW } = dragStartRef.current;
      // Dragging left (decreasing clientX) → panel grows
      const newW = Math.max(MIN_W, Math.min(maxWidth, startW + (startX - ev.clientX)));
      setDisplayW(newW);
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // Re-render D3 only when sliceData changes (i.e. after a completed fetch)
  useEffect(() => {
    if (!svgRef.current || !sliceData?.x?.length) return;
    const colorMode = colorModeRef.current;
    const vmin      = vminRef.current;
    const vmax      = vmaxRef.current;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const xScale = d3.scaleLinear().domain([0, L_PIPE]).range([0, IW]);
    const zScale = d3.scaleLinear().domain([-R_PIPE, R_PIPE]).range([IH, 0]);

    // ── Clip path ────────────────────────────────────────────────────────────
    const clipId = "hm-clip";
    svg.append("defs").append("clipPath").attr("id", clipId)
      .append("rect").attr("width", IW).attr("height", IH);

    // ── Plot group (clipped) ─────────────────────────────────────────────────
    const g = svg.append("g")
      .attr("transform", `translate(${M.left},${M.top})`)
      .attr("clip-path", `url(#${clipId})`);

    // Dark background
    g.append("rect")
      .attr("width", IW).attr("height", IH)
      .attr("fill", "#0b0e18").attr("rx", 2);

    // ── Data cells ───────────────────────────────────────────────────────────
    const { Nx, Nz } = sliceData;
    const cellW = IW / Nx + 0.6;
    const cellH = IH / Nz + 0.6;

    const vals  = colorMode === "pressure" ? sliceData.p : sliceData.vmag;
    const vspan = vmax - vmin || 1;

    g.selectAll("rect.cell")
      .data(d3.range(sliceData.x.length))
      .join("rect")
      .attr("class", "cell")
      .attr("x",      i => xScale(sliceData.x[i]) - cellW / 2)
      .attr("y",      i => zScale(sliceData.z[i]) - cellH / 2)
      .attr("width",  cellW)
      .attr("height", cellH)
      .attr("fill",   i => {
        const t = (vals[i] - vmin) / vspan;
        const [r, gg, b] = colorFn(Math.max(0, Math.min(1, t)));
        return `rgb(${(r*255)|0},${(gg*255)|0},${(b*255)|0})`;
      });

    // Pipe boundary lines
    for (const zEdge of [R_PIPE, -R_PIPE]) {
      g.append("line")
        .attr("x1", 0).attr("x2", IW)
        .attr("y1", zScale(zEdge)).attr("y2", zScale(zEdge))
        .attr("stroke", "rgba(255,255,255,0.25)").attr("stroke-width", 1);
    }

    // Cylinder obstacle ellipse (physically correct for the stretched axes)
    g.append("ellipse")
      .attr("cx", xScale(CYL_X))
      .attr("cy", zScale(0))
      .attr("rx", xScale(CYL_R) - xScale(0))
      .attr("ry", Math.abs(zScale(0) - zScale(CYL_R)))
      .attr("fill", "none")
      .attr("stroke", "rgba(255,255,255,0.7)")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "5,3");

    // ── Axes group (unclipped) ────────────────────────────────────────────────
    const ga = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);

    const styleAxis = sel => {
      sel.selectAll("text")
        .attr("fill", "#94a3b8")
        .attr("font-size", 11)
        .attr("font-family", "system-ui, sans-serif");
      sel.selectAll("line, path").attr("stroke", "#334155");
    };

    // X axis
    ga.append("g")
      .attr("transform", `translate(0,${IH})`)
      .call(d3.axisBottom(xScale).ticks(10).tickSize(4).tickPadding(5))
      .call(styleAxis)
      .append("text")
        .attr("x", IW / 2)
        .attr("y", 40)
        .attr("fill", "#cbd5e1")
        .attr("font-size", 13)
        .attr("font-weight", 600)
        .attr("font-family", "system-ui, sans-serif")
        .attr("text-anchor", "middle")
        .text("x (m)");

    // Z axis
    ga.append("g")
      .call(d3.axisLeft(zScale).ticks(5).tickSize(4).tickPadding(5))
      .call(styleAxis)
      .append("text")
        .attr("x", -48)
        .attr("y", IH / 2)
        .attr("dy", "0.35em")
        .attr("fill", "#cbd5e1")
        .attr("font-size", 13)
        .attr("font-weight", 600)
        .attr("font-family", "system-ui, sans-serif")
        .attr("text-anchor", "middle")
        .text("z (m)");

    // ── Title ────────────────────────────────────────────────────────────────
    const fieldLabel = colorMode === "pressure" ? "Pressure" : "Velocity magnitude";
    svg.append("text")
      .attr("x", M.left + IW / 2)
      .attr("y", 20)
      .attr("fill", "#e2e8f0")
      .attr("font-size", 13)
      .attr("font-weight", 600)
      .attr("font-family", "system-ui, sans-serif")
      .attr("text-anchor", "middle")
      .text(`Top-down slice  y = ${sliceData.y_val?.toFixed(2)} m  —  ${fieldLabel}`);

  }, [sliceData]);  // colorMode/vmin/vmax read from refs — not deps

  return (
    <div style={containerStyle}>
      <div
        style={handleStyle}
        onMouseDown={onHandleMouseDown}
        title="Drag to resize"
      />
      <svg
        ref={svgRef}
        width={displayW}
        height={displayH}
        viewBox={`0 0 ${W} ${H}`}
        style={{
          display: "block",
          // Blur snaps on instantly, fades off gently once new data renders
          transition: loading ? "none" : "filter 0.4s ease",
          filter: loading ? "grayscale(1) blur(4px)" : "none",
        }}
      />
    </div>
  );
}

const containerStyle = {
  position: "absolute",
  bottom: 14,
  right: 14,
  background: "rgba(8,10,18,0.88)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 10,
  backdropFilter: "blur(8px)",
  padding: "6px 6px 4px",
  zIndex: 10,
  boxShadow: "0 4px 24px rgba(0,0,0,0.5)",
};

const handleStyle = {
  position: "absolute",
  top: 5,
  left: 5,
  width: 16,
  height: 16,
  cursor: "nw-resize",
  borderLeft: "2px solid rgba(56,189,248,0.55)",
  borderTop: "2px solid rgba(56,189,248,0.55)",
  borderRadius: "2px 0 0 0",
  zIndex: 1,
};

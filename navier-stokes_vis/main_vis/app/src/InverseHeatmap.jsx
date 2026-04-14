import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

const L_PIPE = 10.0;
const R_PIPE = 0.5;
const CYL_X  = 3.0;
const CYL_R  = 0.25;

const PX_PER_M = 108;
const IW = PX_PER_M * 10;   // 1080 px
const IH = PX_PER_M * 1;    //  108 px
const M  = { top: 38, right: 20, bottom: 54, left: 62 };
const W  = IW + M.left + M.right;   // 1162
const H  = IH + M.top  + M.bottom;  //  200

const MIN_W = 480;
const MAX_W = 3200;

export default function InverseHeatmap({ sliceData, rePred, colorMode, colorFn, vmin, vmax, onClose }) {
  const svgRef   = useRef(null);
  const panelRef = useRef(null);
  const [displayW, setDisplayW] = useState(Math.min(W, 900));
  const displayH = Math.round(displayW * H / W);

  // null = default CSS (bottom/right); after first drag = { top, left }
  const [pos, setPos] = useState(null);

  const colorModeRef = useRef(colorMode);
  colorModeRef.current = colorMode;
  const vminRef = useRef(vmin);
  vminRef.current = vmin;
  const vmaxRef = useRef(vmax);
  vmaxRef.current = vmax;

  // ── Drag header to reposition ────────────────────────────────────────────────
  const onHeaderMouseDown = (e) => {
    if (e.target.tagName === "BUTTON") return;
    e.preventDefault();

    const rect       = panelRef.current.getBoundingClientRect();
    const parentRect = panelRef.current.offsetParent.getBoundingClientRect();
    let curTop  = rect.top  - parentRect.top;
    let curLeft = rect.left - parentRect.left;
    let prevX = e.clientX;
    let prevY = e.clientY;

    const onMove = (ev) => {
      curTop  += ev.clientY - prevY;
      curLeft += ev.clientX - prevX;
      prevX = ev.clientX;
      prevY = ev.clientY;
      setPos({ top: curTop, left: curLeft });
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // ── Drag corner to resize ────────────────────────────────────────────────────
  const dragResizeRef = useRef(null);
  const onResizeMouseDown = (e) => {
    e.preventDefault();
    e.stopPropagation();
    dragResizeRef.current = { startX: e.clientX, startW: displayW, startLeft: pos?.left ?? null };

    const onMove = (ev) => {
      const { startX, startW, startLeft } = dragResizeRef.current;
      const delta = startX - ev.clientX;  // positive = drag left = expand
      const newW  = Math.max(MIN_W, Math.min(MAX_W, startW + delta));
      setDisplayW(newW);
      // When anchored by left (after a drag), recompute left from the start
      // so right edge stays fixed — avoids cumulative drift on each event
      if (startLeft !== null) {
        setPos(p => p ? { ...p, left: startLeft - (newW - startW) } : p);
      }
    };
    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  };

  // ── D3 render ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!svgRef.current || !sliceData?.x?.length) return;
    const colorMode = colorModeRef.current;
    const vmin      = vminRef.current;
    const vmax      = vmaxRef.current;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const xScale = d3.scaleLinear().domain([0, L_PIPE]).range([0, IW]);
    const zScale = d3.scaleLinear().domain([-R_PIPE, R_PIPE]).range([IH, 0]);

    const clipId = "inv-hm-clip";
    svg.append("defs").append("clipPath").attr("id", clipId)
      .append("rect").attr("width", IW).attr("height", IH);

    const g = svg.append("g")
      .attr("transform", `translate(${M.left},${M.top})`)
      .attr("clip-path", `url(#${clipId})`);

    g.append("rect")
      .attr("width", IW).attr("height", IH)
      .attr("fill", "#0b0e18").attr("rx", 2);

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

    for (const zEdge of [R_PIPE, -R_PIPE]) {
      g.append("line")
        .attr("x1", 0).attr("x2", IW)
        .attr("y1", zScale(zEdge)).attr("y2", zScale(zEdge))
        .attr("stroke", "rgba(255,255,255,0.25)").attr("stroke-width", 1);
    }

    g.append("ellipse")
      .attr("cx", xScale(CYL_X))
      .attr("cy", zScale(0))
      .attr("rx", xScale(CYL_R) - xScale(0))
      .attr("ry", Math.abs(zScale(0) - zScale(CYL_R)))
      .attr("fill", "none")
      .attr("stroke", "rgba(255,255,255,0.7)")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "5,3");

    const ga = svg.append("g").attr("transform", `translate(${M.left},${M.top})`);
    const styleAxis = sel => {
      sel.selectAll("text")
        .attr("fill", "#94a3b8").attr("font-size", 11)
        .attr("font-family", "system-ui, sans-serif");
      sel.selectAll("line, path").attr("stroke", "#334155");
    };

    ga.append("g")
      .attr("transform", `translate(0,${IH})`)
      .call(d3.axisBottom(xScale).ticks(10).tickSize(4).tickPadding(5))
      .call(styleAxis)
      .append("text")
        .attr("x", IW / 2).attr("y", 40)
        .attr("fill", "#cbd5e1").attr("font-size", 13).attr("font-weight", 600)
        .attr("font-family", "system-ui, sans-serif").attr("text-anchor", "middle")
        .text("x (m)");

    ga.append("g")
      .call(d3.axisLeft(zScale).ticks(5).tickSize(4).tickPadding(5))
      .call(styleAxis)
      .append("text")
        .attr("x", -48).attr("y", IH / 2).attr("dy", "0.35em")
        .attr("fill", "#cbd5e1").attr("font-size", 13).attr("font-weight", 600)
        .attr("font-family", "system-ui, sans-serif").attr("text-anchor", "middle")
        .text("z (m)");

    const fieldLabel = colorMode === "pressure" ? "Pressure" : "Velocity magnitude";
    svg.append("text")
      .attr("x", M.left + IW / 2).attr("y", 20)
      .attr("fill", "#f472b6").attr("font-size", 13).attr("font-weight", 600)
      .attr("font-family", "system-ui, sans-serif").attr("text-anchor", "middle")
      .text(`Inverse model  Re\u200a=\u200a${rePred.toFixed(1)}  —  ${fieldLabel}`);

  }, [sliceData, rePred]);

  const posStyle = pos
    ? { top: pos.top, left: pos.left }
    : { bottom: 244, right: 14 };

  return (
    <div ref={panelRef} style={{ ...containerStyle, ...posStyle }}>
      {/* ── Drag handle (resize, top-left) ───────────────────────────────────── */}
      <div style={resizeHandleStyle} onMouseDown={onResizeMouseDown} title="Drag to resize" />

      {/* ── Header bar ───────────────────────────────────────────────────────── */}
      <div style={headerStyle} onMouseDown={onHeaderMouseDown}>
        <span style={headerTitleStyle}>Inverse model visualization</span>
        <button style={closeBtnStyle} onClick={onClose} title="Close">×</button>
      </div>

      {/* ── Heatmap SVG ──────────────────────────────────────────────────────── */}
      <svg
        ref={svgRef}
        width={displayW}
        height={displayH}
        viewBox={`0 0 ${W} ${H}`}
        style={{ display: "block" }}
      />
    </div>
  );
}

const containerStyle = {
  position: "absolute",
  background: "rgba(8,10,18,0.92)",
  border: "1px solid rgba(244,114,182,0.25)",
  borderRadius: 10,
  backdropFilter: "blur(8px)",
  padding: "0 6px 4px",
  zIndex: 25,
  boxShadow: "0 4px 28px rgba(0,0,0,0.6)",
  userSelect: "none",
};

const resizeHandleStyle = {
  position: "absolute",
  top: 5,
  left: 5,
  width: 16,
  height: 16,
  cursor: "nw-resize",
  borderLeft: "2px solid rgba(244,114,182,0.55)",
  borderTop: "2px solid rgba(244,114,182,0.55)",
  borderRadius: "2px 0 0 0",
  zIndex: 1,
};

const headerStyle = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "6px 4px 4px 24px",
  cursor: "grab",
  gap: 10,
};

const headerTitleStyle = {
  fontSize: 12,
  fontWeight: 600,
  letterSpacing: "0.05em",
  textTransform: "uppercase",
  color: "#f472b6",
  fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
  pointerEvents: "none",
};

const closeBtnStyle = {
  background: "none",
  border: "none",
  color: "#64748b",
  fontSize: 20,
  lineHeight: 1,
  cursor: "pointer",
  padding: "0 2px",
  borderRadius: 4,
  fontFamily: "system-ui, sans-serif",
  flexShrink: 0,
};

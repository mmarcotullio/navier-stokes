import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

const MARGIN  = { top: 28, right: 130, bottom: 60, left: 70 };
const COLORS  = ["#4361ee", "#f77f00", "#2dc653"];

export default function ReScatterPlot() {
  const containerRef = useRef(null);
  const svgRef       = useRef(null);
  const [data, setData]   = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    d3.csv("/data/re_predictions.csv").then(raw => {
      const rows = raw.map(r => ({
        true_re: +r.true_re,
        pred_re: +r.pred_re,
        uin:     +r.uin,
      }));
      setData(rows);
    }).catch(err => setError(err.message));
  }, []);

  useEffect(() => {
    if (!data || !containerRef.current) return;

    const container = containerRef.current;
    const totalW = container.clientWidth || 500;
    const totalH = Math.max(340, Math.round(totalW * 0.72));
    const iW = totalW - MARGIN.left - MARGIN.right;
    const iH = totalH - MARGIN.top  - MARGIN.bottom;

    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current)
      .attr("width",  totalW)
      .attr("height", totalH);

    const g = svg.append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    // Tooltip
    let tipEl = container.querySelector(".rsp-tooltip");
    if (!tipEl) {
      tipEl = document.createElement("div");
      tipEl.className = "rsp-tooltip";
      Object.assign(tipEl.style, {
        position: "absolute", pointerEvents: "none",
        background: "rgba(17,24,39,0.96)", border: "1px solid rgba(255,255,255,0.12)",
        borderRadius: "10px", padding: "14px 20px", fontSize: "14px",
        color: "#e2e8f0", display: "none", lineHeight: "1.8",
        backdropFilter: "blur(4px)", minWidth: "200px",
      });
      container.style.position = "relative";
      container.appendChild(tipEl);
    }

    // Scales
    const x = d3.scaleLinear().domain([100, 1000]).range([0, iW]);
    const y = d3.scaleLinear().domain([100, 1000]).range([iH, 0]);

    // Colour scale for U_in
    const uinVals = [...new Set(data.map(d => d.uin))].sort(d3.ascending);
    const colour  = d3.scaleOrdinal().domain(uinVals).range(COLORS);

    // Grid
    g.append("g").attr("class", "grid")
      .attr("transform", `translate(0,${iH})`)
      .call(d3.axisBottom(x).ticks(5).tickSize(-iH).tickFormat(""))
      .call(ax => { ax.select(".domain").remove(); ax.selectAll("line").attr("stroke", "rgba(255,255,255,0.07)"); });

    g.append("g").attr("class", "grid")
      .call(d3.axisLeft(y).ticks(5).tickSize(-iW).tickFormat(""))
      .call(ax => { ax.select(".domain").remove(); ax.selectAll("line").attr("stroke", "rgba(255,255,255,0.07)"); });

    // Axes
    g.append("g").attr("transform", `translate(0,${iH})`)
      .call(d3.axisBottom(x).ticks(6).tickFormat(d3.format("d")))
      .call(ax => {
        ax.select(".domain").attr("stroke", "#475569");
        ax.selectAll("text").attr("fill", "#94a3b8").attr("font-size", "13px");
        ax.selectAll("line").attr("stroke", "#475569");
      });

    g.append("g")
      .call(d3.axisLeft(y).ticks(6).tickFormat(d3.format("d")))
      .call(ax => {
        ax.select(".domain").attr("stroke", "#475569");
        ax.selectAll("text").attr("fill", "#94a3b8").attr("font-size", "13px");
        ax.selectAll("line").attr("stroke", "#475569");
      });

    // Axis labels
    svg.append("text")
      .attr("x", MARGIN.left + iW/2).attr("y", totalH - 8)
      .attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", "14px")
      .text("True Re");

    svg.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -(MARGIN.top + iH/2)).attr("y", 18)
      .attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", "14px")
      .text("Predicted Re");

    // Ideal 1:1 line
    g.append("line")
      .attr("x1", x(100)).attr("y1", y(100))
      .attr("x2", x(1000)).attr("y2", y(1000))
      .attr("stroke", "#e63946").attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "6,4");

    g.append("text")
      .attr("x", x(820)).attr("y", y(860))
      .attr("fill", "#e63946").attr("font-size", "15px")
      .attr("transform", `rotate(-45,${x(820)},${y(860)})`)
      .text("ideal");

    // Dots
    g.selectAll(".dot")
      .data(data)
      .join("circle")
      .attr("class", "dot")
      .attr("cx", d => x(d.true_re))
      .attr("cy", d => y(d.pred_re))
      .attr("r", 5)
      .attr("fill", d => colour(d.uin))
      .attr("stroke", "#111827").attr("stroke-width", 1.5)
      .style("cursor", "pointer")
      .on("mousemove", (event, d) => {
        const resid = (d.pred_re - d.true_re).toFixed(2);
        const sign  = d.pred_re >= d.true_re ? "+" : "";
        tipEl.style.display = "block";
        tipEl.innerHTML = `<b>Re</b> true = ${d.true_re.toFixed(0)}<br>
          <b>Re</b> pred = ${d.pred_re.toFixed(1)}<br>
          residual = ${sign}${resid}<br>
          U_in = ${d.uin.toFixed(2)} m/s<br>
          \u03bd true = ${(d.uin / d.true_re).toExponential(2)} m\u00b2/s<br>
          \u03bd pred = ${(d.uin / d.pred_re).toExponential(2)} m\u00b2/s`;
        const rect = container.getBoundingClientRect();
        let left = event.clientX - rect.left + 14;
        let top  = event.clientY - rect.top  - 40;
        const tipW = tipEl.offsetWidth;
        const tipH = tipEl.offsetHeight;
        if (left + tipW > rect.width)  left = event.clientX - rect.left - tipW - 14;
        if (top  + tipH > rect.height) top  = event.clientY - rect.top  - tipH - 8;
        tipEl.style.left = Math.max(0, left) + "px";
        tipEl.style.top  = Math.max(0, top)  + "px";
      })
      .on("mouseleave", () => { tipEl.style.display = "none"; });

    // Legend
    const legX = MARGIN.left + iW + 14;
    const legY = MARGIN.top + 14;
    svg.append("text")
      .attr("x", legX).attr("y", legY - 6)
      .attr("fill", "#64748b").attr("font-size", "12px").attr("font-weight", "600")
      .text("U_in (m/s)");

    uinVals.forEach((u, i) => {
      const ly = legY + i * 26;
      svg.append("circle")
        .attr("cx", legX + 6).attr("cy", ly + 5)
        .attr("r", 6).attr("fill", colour(u)).attr("stroke", "#111827").attr("stroke-width", 1.5);
      svg.append("text")
        .attr("x", legX + 18).attr("y", ly + 9)
        .attr("fill", "#94a3b8").attr("font-size", "13px")
        .text(`${u.toFixed(2)} m/s`);
    });

  }, [data]);

  // Resize observer
  useEffect(() => {
    if (!containerRef.current || !data) return;
    const ro = new ResizeObserver(() => { setData(d => d ? [...d] : d); });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [data]);

  if (error) return (
    <div className="glass" style={{ padding: "1.5rem", color: "#f87171" }}>
      Failed to load data: {error}
    </div>
  );

  return (
    <div className="glass" style={{ padding: "1rem", overflow: "hidden" }}>
      {!data && (
        <div style={{ color: "#94a3b8", padding: "2rem", textAlign: "center" }}>Loading…</div>
      )}
      <div ref={containerRef} style={{ width: "100%", display: data ? "block" : "none" }}>
        <svg ref={svgRef} style={{ display: "block", width: "100%" }} />
      </div>
    </div>
  );
}

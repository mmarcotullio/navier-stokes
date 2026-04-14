import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";

const MARGIN = { top: 30, right: 44, bottom: 55, left: 70 };

export default function SpeedupChart() {
  const containerRef = useRef(null);
  const svgRef       = useRef(null);
  const [data, setData]   = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    d3.csv("/data/mesh_refinement_runtime.csv").then(raw => {
      const rows = raw
        .filter(r => r.status === "ok")
        .map(r => ({
          n_cells: +r.n_cells,
          speedup: +r.solver_cpu_s / +r.fno_inference_s,
        }))
        .sort((a, b) => a.n_cells - b.n_cells);
      setData(rows);
    }).catch(err => setError(err.message));
  }, []);

  useEffect(() => {
    if (!data || !containerRef.current) return;

    const container = containerRef.current;
    const totalW = container.clientWidth || 700;
    const totalH = 320;
    const iW = totalW - MARGIN.left - MARGIN.right;
    const iH = totalH - MARGIN.top  - MARGIN.bottom;

    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current)
      .attr("width",  totalW)
      .attr("height", totalH);

    const g = svg.append("g")
      .attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);

    const xSc = d3.scaleLog().domain([700, 120000]).range([0, iW]);
    const ySc = d3.scaleLog().domain([1, 500]).range([iH, 0]);

    // Grid
    g.append("g")
      .call(d3.axisLeft(ySc).tickValues([1, 5, 20, 100, 500]).tickSize(-iW).tickFormat(""))
      .call(ax => { ax.select(".domain").remove(); ax.selectAll("line").attr("stroke", "rgba(255,255,255,0.07)"); });
    g.append("g").attr("transform", `translate(0,${iH})`)
      .call(d3.axisBottom(xSc).ticks(5, "~s").tickSize(-iH).tickFormat(""))
      .call(ax => { ax.select(".domain").remove(); ax.selectAll("line").attr("stroke", "rgba(255,255,255,0.07)"); });

    // Area fill
    const defs = svg.append("defs");
    const grad = defs.append("linearGradient")
      .attr("id", "speedup-grad").attr("x1","0%").attr("x2","0%").attr("y1","0%").attr("y2","100%");
    grad.append("stop").attr("offset","0%").attr("stop-color","#38bdf8").attr("stop-opacity", 0.28);
    grad.append("stop").attr("offset","100%").attr("stop-color","#38bdf8").attr("stop-opacity", 0.02);

    g.append("path").datum(data)
      .attr("fill", "url(#speedup-grad)")
      .attr("d", d3.area()
        .x(d => xSc(d.n_cells)).y0(ySc(1)).y1(d => ySc(d.speedup))
        .curve(d3.curveMonotoneX));

    // 1× reference
    g.append("line")
      .attr("x1", 0).attr("x2", iW).attr("y1", ySc(1)).attr("y2", ySc(1))
      .attr("stroke", "#475569").attr("stroke-width", 1).attr("stroke-dasharray", "4,3");

    // Line
    g.append("path").datum(data)
      .attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-width", 2.5)
      .attr("d", d3.line().x(d => xSc(d.n_cells)).y(d => ySc(d.speedup)).curve(d3.curveMonotoneX));

    // Dots
    g.selectAll(".sc-dot").data(data).join("circle")
      .attr("class", "sc-dot")
      .attr("cx", d => xSc(d.n_cells)).attr("cy", d => ySc(d.speedup))
      .attr("r", 4).attr("fill", "#38bdf8")
      .attr("stroke", "#111827").attr("stroke-width", 1.5)
      .style("cursor", "pointer");

    // Annotation on last point
    const last = data[data.length - 1];
    g.append("text")
      .attr("x", xSc(last.n_cells) - 10).attr("y", ySc(last.speedup) - 10)
      .attr("text-anchor", "end").attr("fill", "#f59e0b")
      .attr("font-size", "13px").attr("font-weight", "600")
      .text(`${last.speedup.toFixed(0)}× faster`);

    // Axes
    g.append("g").attr("transform", `translate(0,${iH})`)
      .call(d3.axisBottom(xSc).ticks(6, "~s"))
      .call(ax => {
        ax.select(".domain").attr("stroke", "#475569");
        ax.selectAll("text").attr("fill", "#94a3b8").attr("font-size", "13px");
        ax.selectAll("line").attr("stroke", "#475569");
      });
    g.append("g")
      .call(d3.axisLeft(ySc).tickValues([1, 5, 20, 100, 500]).tickFormat(d => d + "×"))
      .call(ax => {
        ax.select(".domain").attr("stroke", "#475569");
        ax.selectAll("text").attr("fill", "#94a3b8").attr("font-size", "13px");
        ax.selectAll("line").attr("stroke", "#475569");
      });

    // Axis labels
    svg.append("text")
      .attr("x", MARGIN.left + iW / 2).attr("y", totalH - 8)
      .attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", "14px")
      .text("Mesh cells (log scale)");
    svg.append("text")
      .attr("transform", `translate(16,${MARGIN.top + iH / 2}) rotate(-90)`)
      .attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", "14px")
      .text("Speedup factor (log scale)");

    // Tooltip
    let tipEl = container.querySelector(".sc-tooltip");
    if (!tipEl) {
      tipEl = document.createElement("div");
      tipEl.className = "sc-tooltip";
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
    g.selectAll(".sc-dot")
      .on("mouseover", (event, d) => {
        tipEl.style.display = "block";
        tipEl.innerHTML = `<strong>${d.n_cells.toLocaleString()} cells</strong><br>Speedup: <strong style="color:#38bdf8">${d.speedup.toFixed(1)}×</strong>`;
      })
      .on("mousemove", event => {
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
      .on("mouseout", () => { tipEl.style.display = "none"; });

  }, [data]);

  useEffect(() => {
    if (!containerRef.current || !data) return;
    const ro = new ResizeObserver(() => setData(d => d ? [...d] : d));
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
      {!data && <div style={{ color: "#94a3b8", padding: "2rem", textAlign: "center" }}>Loading…</div>}
      <div ref={containerRef} style={{ width: "100%", display: data ? "block" : "none" }}>
        <svg ref={svgRef} style={{ display: "block", width: "100%" }} />
      </div>
    </div>
  );
}

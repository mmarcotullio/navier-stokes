import { useState, useEffect, useRef } from "react";
import Section from "../components/Section.jsx";
import Cite from "../components/Cite.jsx";

const subheadStyle = {
  fontSize: "1.5rem",
  fontWeight: 700,
  color: "#cbd5e1",
  margin: "2.5rem 0 0.75rem",
};

const pStyle = { marginBottom: "1.5rem", fontSize: "1.5rem", color: "#f8fafc", lineHeight: 1.7, maxWidth: "none" };

const liStyle = { fontSize: "1.5rem", color: "#f8fafc", lineHeight: 1.7, marginBottom: "0.5rem" };

export default function MainVisualization() {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    const tryTypeset = () => {
      if (window.MathJax?.startup?.promise) {
        const el = document.getElementById("demo");
        window.MathJax.startup.promise
          .then(() => window.MathJax.typesetPromise(el ? [el] : []))
          .catch(console.error);
      } else {
        setTimeout(tryTypeset, 100);
      }
    };
    tryTypeset();
  }, []);

  useEffect(() => {
    const onFsChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      containerRef.current?.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  }

  const text = (
    <div style={{ fontSize: "1.5rem", color: "#f8fafc", lineHeight: 1.7 }}>

      <p style={{ marginBottom: "0.75rem", fontSize: "1.1rem" }}>
        <span
          onClick={() => {
            const el = document.getElementById("demo-vis");
            if (!el) return;
            const headerH = document.querySelector(".reading-header")?.offsetHeight ?? 0;
            window.scrollTo({ top: el.getBoundingClientRect().top + window.scrollY - headerH - 12, behavior: "smooth" });
          }}
          onMouseEnter={e => { e.currentTarget.style.textDecoration = "underline"; }}
          onMouseLeave={e => { e.currentTarget.style.textDecoration = "none"; }}
          style={{ color: "#60c8f5", cursor: "pointer", fontWeight: 600 }}
        >Click here</span> to skip to demo.
      </p>

      <p style={pStyle}>
        The interactive visualization below embeds a real-time CFD simulation
        explorer. It simulates flow through a pipe of length 10 and radius 0.5. 
        It features a spherical obstacle of radius 0.25, with its center at \(x = 3\). The forward surrogate produces the full steady-state 3D flow
        field in a single forward pass. 
      </p>

      <h3 style={subheadStyle}>Physical parameters</h3>

      <p style={{ ...pStyle, marginBottom: "0.75rem" }}>
        The left panel exposes two independent physical parameters:
      </p>

      <ul style={{ paddingLeft: "1.5rem", marginBottom: "1.5rem" }}>
        <li style={liStyle}>
          <strong>Reynolds number</strong> ({String.raw`\(Re \in [100, 1000]\)`}) controls the
          balance between inertial and viscous forces.
        </li>
        <li style={liStyle}>
          <strong>Inlet velocity</strong> ({String.raw`\(U_{\mathrm{in}} \in [0.1, 1.0]\)`} m/s)
          sets the mean speed of the incoming fluid. The kinematic viscosity{" "}
          {String.raw`\(\nu = U_{\mathrm{in}} D / Re\)`} is calculated and displayed
          alongside the closest matching reference fluid (e.g. water, oil). See
          references <Cite n={8} /> and <Cite n={9} /> for source data.
        </li>
      </ul>

      <h3 style={subheadStyle}>Grid resolution</h3>

      <p style={pStyle}>
        The cross-sectional resolution {String.raw`\(N_y = N_z\)`} can be adjusted from
        9 to 51 grid points. Given {String.raw`\(N_y = N_z\)`},{" "}
        {String.raw`\(N_x\)`} is automatically calculated to provide uniform distance
        between points in all directions. Coarser grids produce results faster,
        and finer grids take more time to load.
      </p>

      <h3 style={subheadStyle}>Field encoding</h3>

      <p style={pStyle}>
        The <em>Encode by</em> toggle switches the colour mapping between velocity
        magnitude {String.raw`\(|\mathbf{u}|\)`} (m/s) and kinematic pressure{" "}
        {String.raw`\(p / U_{\mathrm{in}}^2\)`} (dimensionless). The{" "}
        <em>Color range</em> toggle switches between <em>Auto</em> (color scale
        recomputed on each parameter update, useful for inspecting absolute values) and{" "}
        <em>Fixed</em> (scale locked to the current range, useful for comparing
        fields across different parameter settings).
      </p>

      <h3 style={subheadStyle}>Wall clip</h3>

      <p style={pStyle}>
        The <em>Wall Clip</em> slider reveals the interior dynamics of the flow.
        The no-slip boundary condition ({String.raw`\(\mathbf{u} = 0\)`} at the pipe
        wall) means the outermost voxels carry zero velocity; without clipping,
        these zero-valued points dominate the color scale and obscures the
        variation in the interior when viewing the velocity field.
      </p>

      <h3 style={subheadStyle}>Inverse model</h3>

      <p style={pStyle}>
        The <em>Recover Parameter</em> button triggers the inverse surrogate. Given
        the current flow field (at the specified {String.raw`\(Re\)`} and{" "}
        {String.raw`\(U_{\mathrm{in}}\)`}), the CNN predicts the Reynolds number that
        produced it. The predicted {String.raw`\(\hat{Re}\)`}, together with the
        implied {String.raw`\(\hat{\nu}\)`}, is displayed alongside the true values so
        that the residual error is immediately visible. A second button renders the
        forward model's output at the predicted {String.raw`\(\hat{Re}\)`}, placing it
        alongside the original field for direct visual comparison.
      </p>

      <p>Panels are resizable. Press Fullscreen to expand.</p>
    </div>
  );

  const visual = (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <button
          onClick={toggleFullscreen}
          style={{
            padding: "6px 16px", borderRadius: "999px", fontSize: "12px",
            cursor: "pointer", fontFamily: "inherit",
            background: "rgba(56,189,248,0.1)",
            border: "1px solid rgba(56,189,248,0.3)",
            color: "#38bdf8", transition: "all 0.15s",
          }}>
          {isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
        </button>
      </div>
      <div
        id="demo-vis"
        ref={containerRef}
        className="glass"
        style={{ overflow: "hidden", padding: 0, background: "#0f172a" }}
      >
        <iframe
          src="http://localhost:5173"
          title="CFD Explorer"
          style={{ width: "100%", height: "600px", border: "none", display: "block" }}
        />
      </div>
    </div>
  );

  return (
    <Section
      id="demo"
      side="full"
      wide
      label="05 · Interactive Demo"
      title="Real-Time CFD Simulation Explorer"
      text={text}
      visual={visual}
    />
  );
}

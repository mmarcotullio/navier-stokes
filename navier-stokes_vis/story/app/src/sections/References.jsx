import { useState, useEffect } from "react";

const REFS = [
  {
    text: "N. H. Nelsen and Y. Yang, \"Operator Learning Meets Inverse Problems: A Probabilistic Perspective,\" arXiv preprint arXiv:2508.20207, 2025.",
  },
  {
    text: "NASA Glenn Research Center, \"Conservation of Mass,\"",
    url: "https://www.grc.nasa.gov/www/k-12/airplane/mass.html",
  },
  {
    text: "NASA Glenn Research Center, \"The Navier-Stokes Equations,\"",
    url: "https://www.grc.nasa.gov/www/k-12/airplane/nseqs.html",
  },
  {
    text: "C. E. Brennen, \"Poiseuille Flow,\" California Institute of Technology,",
    url: "http://brennen.caltech.edu/fluidbook/basicfluiddynamics/Navierstokesexactsolutions/poiseuilleflow.pdf",
  },
  {
    text: "NASA Glenn Research Center, \"Boundary Conditions,\" presented at TFAWS, 2007,",
    url: "https://www.grc.nasa.gov/WWW/wind/TFAWS2007/BConds.pdf",
  },
  {
    text: "Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, and A. Anandkumar, \"Fourier Neural Operator for Parametric Partial Differential Equations,\" in Proc. Int. Conf. Learning Representations (ICLR), 2021.",
  },
  {
    text: "OpenCFD Ltd., \"OpenFOAM: The Open Source CFD Toolbox,\" OpenFOAM.",
    url: "https://www.openfoam.com/",
  },
  {
    text: "F. M. White, Fluid Mechanics, 8th ed. New York: McGraw-Hill, 2011, App. A, Table A.1.",
  },
  {
    text: "The Engineering ToolBox, \"Kinematic Viscosity,\"",
    url: "https://www.engineeringtoolbox.com/kinematic-viscosity-d_397.html",
  },
];

const liBase = {
  display: "flex",
  gap: "0.75rem",
  padding: "0.6rem 0.9rem",
  borderRadius: "8px",
  fontSize: "0.95rem",
  lineHeight: "1.6",
  transition: "background 0.3s, border 0.3s",
};

export default function References() {
  const [open, setOpen] = useState(false);
  const [highlighted, setHighlighted] = useState(null);

  useEffect(() => {
    const handler = e => {
      const n = e.detail;
      setOpen(true);
      setHighlighted(n);
      setTimeout(() => {
        document.getElementById(`ref-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 350);
      setTimeout(() => setHighlighted(null), 2500);
    };
    window.addEventListener("open-ref", handler);
    return () => window.removeEventListener("open-ref", handler);
  }, []);

  return (
    <section
      id="references"
      style={{
        padding: "2vh 8vw 4vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
      }}
    >
      <div style={{ width: "100%", maxWidth: "72ch" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "center",
            marginBottom: open ? "1.25rem" : 0,
          }}
        >
          <button
            onClick={() => setOpen(o => !o)}
            aria-expanded={open}
            style={{
              padding: "10px 32px", borderRadius: "999px", fontSize: "1rem",
              cursor: "pointer", fontFamily: "inherit",
              background: open ? "rgba(56,189,248,0.1)" : "rgba(255,255,255,0.04)",
              border: open ? "1px solid rgba(56,189,248,0.3)" : "1px solid rgba(255,255,255,0.1)",
              color: open ? "#38bdf8" : "#94a3b8",
              transition: "all 0.15s",
            }}
          >
            {open ? "Collapse References" : "Show References"}
          </button>
        </div>

        <div
          style={{
            overflow: "hidden",
            maxHeight: open ? "1100px" : "0",
            transition: "max-height 0.35s ease",
          }}
        >
          <ol
            style={{
              listStyle: "none",
              padding: 0,
              margin: 0,
              display: "flex",
              flexDirection: "column",
              gap: "0.6rem",
            }}
          >
            {REFS.map((ref, i) => {
              const isHighlighted = highlighted === i + 1;
              return (
                <li
                  key={i}
                  id={`ref-${i + 1}`}
                  style={{
                    ...liBase,
                    background: isHighlighted
                      ? "rgba(56,189,248,0.12)"
                      : "rgba(255,255,255,0.03)",
                    border: isHighlighted
                      ? "1px solid rgba(56,189,248,0.4)"
                      : "1px solid rgba(255,255,255,0.06)",
                  }}
                >
                  <span style={{ color: "#38bdf8", fontWeight: 600, minWidth: "1.6rem" }}>
                    [{i + 1}]
                  </span>
                  <span style={{ color: "#cbd5e1" }}>
                    {ref.text}{" "}
                    {ref.url && (
                      <a
                        href={ref.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        style={{ color: "#38bdf8" }}
                      >
                        {ref.url}
                      </a>
                    )}
                  </span>
                </li>
              );
            })}
          </ol>

          <div style={{ marginTop: "1.5rem" }}>
            <h4 style={{ fontSize: "0.85rem", fontWeight: 600, color: "#60c8f5", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: "0.75rem" }}>
              Images
            </h4>
            <div style={{ ...liBase, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}>
              <span style={{ color: "#cbd5e1", fontSize: "0.95rem", lineHeight: 1.6 }}>
                National Aeronautics and Space Administration, "Spinoff 2019," <em>NASA Spinoff</em>. {" "}
                <a href="https://spinoff.nasa.gov/Spinoff2019/it_5.html" target="_blank" rel="noopener noreferrer" style={{ color: "#60c8f5" }}>
                  https://spinoff.nasa.gov/Spinoff2019/it_5.html
                </a>.
              </span>
            </div>
            <div style={{ ...liBase, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)" }}>
              <span style={{ color: "#cbd5e1", fontSize: "0.95rem", lineHeight: 1.6 }}>
                marekuliasz, "Kinetic energy equation on blackboard," <em>iStock</em>, Jun. 2, 2010.{" "}
                <a href="https://www.istockphoto.com/photos/quantum-physics-on-a-blackboard" target="_blank" rel="noopener noreferrer" style={{ color: "#60c8f5" }}>
                  https://www.istockphoto.com/photos/quantum-physics-on-a-blackboard
                </a>.
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

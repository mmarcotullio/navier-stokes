import { useEffect, useRef } from "react";
import Section from "../components/Section.jsx";
import Cite from "../components/Cite.jsx";
import HeatmapPanel from "../components/HeatmapPanel.jsx";
import VelocityProfilePanel from "../components/VelocityProfilePanel.jsx";
import SpeedupChart from "../components/SpeedupChart.jsx";

const subheadStyle = {
  fontSize: "1.5rem",
  fontWeight: 700,
  color: "#cbd5e1",
  margin: "2.5rem 0 0.75rem",
};

const eqStyle = {
  margin: "0.75rem 0",
  overflowX: "auto",
  fontSize: "1.4rem",
};

const tableStyle = {
  width: "100%",
  borderCollapse: "collapse",
  fontSize: "1.3rem",
  margin: "1rem 0 1.5rem",
  color: "#e2e8f0",
};

const thStyle = {
  padding: "0.6rem 1rem",
  background: "rgba(56,189,248,0.1)",
  border: "1px solid rgba(255,255,255,0.12)",
  fontWeight: 600,
  color: "#7dd3fc",
  textAlign: "left",
};

const tdStyle = {
  padding: "0.55rem 1rem",
  border: "1px solid rgba(255,255,255,0.08)",
};

const pStyle = { marginBottom: "1.5rem", fontSize: "1.5rem", color: "#f8fafc", lineHeight: 1.7, maxWidth: "none" };

function scrollAndHighlight(id, highlight = true) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!highlight) {
    const headerH = document.querySelector(".reading-header")?.offsetHeight ?? 0;
    const top = el.getBoundingClientRect().top + window.scrollY - headerH - 12;
    window.scrollTo({ top, behavior: "smooth" });
    return;
  }
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  el.style.transition = "box-shadow 0.3s";
  el.style.boxShadow = "0 0 0 3px rgba(56,189,248,0.7), 0 0 20px rgba(56,189,248,0.3)";
  el.style.borderRadius = "8px";
  setTimeout(() => {
    el.style.transition = "box-shadow 1.5s";
    el.style.boxShadow = "";
    setTimeout(() => { el.style.transition = ""; el.style.borderRadius = ""; }, 1500);
  }, 1000);
}

const navLink = (id, label, highlight = true) => (
  <span
    onClick={() => scrollAndHighlight(id, highlight)}
    onMouseEnter={e => { e.currentTarget.style.textDecoration = "underline"; }}
    onMouseLeave={e => { e.currentTarget.style.textDecoration = "none"; }}
    style={{ color: "#60c8f5", cursor: "pointer", fontWeight: 600 }}
  >{label}</span>
);

export default function ForwardOperator() {
  const contentRef = useRef(null);

  useEffect(() => {
    const tryTypeset = () => {
      if (window.MathJax?.startup?.promise) {
        const el = document.getElementById("forward");
        window.MathJax.startup.promise
          .then(() => window.MathJax.typesetPromise(el ? [el] : []))
          .catch(console.error);
      } else {
        setTimeout(tryTypeset, 100);
      }
    };
    tryTypeset();
  }, []);

  // ── Text: vis floats right, prose wraps around and beneath it ────────────
  const text = (
    <div style={{ fontSize: "1.5rem", color: "#f8fafc", lineHeight: 1.7 }}>
      <div id="speedup-chart-float" style={{ float: "right", width: "45%", marginLeft: "3rem", marginBottom: "1.5rem" }}>
        <span className="section-chip" style={{ fontSize: "0.82rem", marginBottom: "0.4rem" }}>Speedup</span>
        <h3 style={{ fontSize: "1.2rem", color: "#f1f5f9", margin: "0.3rem 0 0.5rem" }}>
          FNO vs OpenFOAM: Runtime Speedup
        </h3>
        <SpeedupChart />
      </div>
      <p style={pStyle}>
        To address the high computational costs of traditional solvers, this
        project builds a forward surrogate model: a neural network that
        takes as input a compact descriptor of the physical parameters and the
        geometry, and directly produces the converged steady-state velocity and
        pressure fields on a structured 3D grid. The chosen architecture is the
        Fourier Neural Operator (FNO) introduced by Li et al. <Cite n={6} />, extended
        here to three spatial dimensions.
      </p>

      <p style={pStyle}>
        The surrogate is trained on a dataset of 1,000 OpenFOAM <Cite n={7} /> (simpleFoam) 
        cases, with Reynolds numbers {String.raw`\(Re \in [100, 1000]\)`} and inlet
        velocities {String.raw`\(U_{in} \in [0.1, 1.0]\)`}.
      </p>

      <p style={pStyle}>
        At inference time, the surrogate model produces the full 3D field on
        the grid in a single forward pass, and can predict at any resolution, 
        allowing for real-time adaptive mesh refinement.
      </p>

      <h3 style={subheadStyle}>Performance</h3>

      <p style={pStyle}>
        The speedup visualization to the {navLink("speedup-chart-float", "right")} shows the speedup factor of the FNO's inference
        time compared with OpenFOAM's runtime. Directly {navLink("field-comparison", "below")} is a field comparison showing the
        actual pressure and velocity fields compared with the FNO's predictions, and the error. Further {navLink("velocity-profile", "down")},
        you'll find a visualization that compares the axial velocity along the pipe's x-axis for the FNO and CFD actual. Interact with these
        visualizations to explore the degree to which this implementation adheres to the standards of <b>accuracy</b> and <b>computational efficiency</b>.  
      </p>

      


      <div style={{ clear: "both" }} />

      <div id="field-comparison" style={{ padding: "0 0 2rem" }}>
        <div style={{ marginBottom: "0.75rem" }}>
          <span className="section-chip" style={{ fontSize: "0.82rem", marginBottom: "0.5rem" }}>Field Comparison</span>
          <h3 style={{ fontSize: "clamp(1.1rem, 1.8vw, 1.6rem)", color: "#f1f5f9", marginTop: "0.4rem" }}>
            CFD vs FNO Prediction (Toggle Between Cases)
          </h3>
          <p style={{ color: "#cbd5e1", fontSize: "1rem", marginTop: "0.3rem" }}>
            Select a (Re, U_in) pair to view the cross section y = 0
          </p>
        </div>
        <HeatmapPanel />
      </div>

      <div id="velocity-profile" style={{ padding: "0 0 2rem" }}>
        <div style={{ marginBottom: "0.75rem" }}>
          <span className="section-chip" style={{ fontSize: "0.82rem", marginBottom: "0.5rem" }}>Velocity Profile</span>
          <h3 style={{ fontSize: "clamp(1.1rem, 1.8vw, 1.6rem)", color: "#f1f5f9", marginTop: "0.4rem" }}>
            Axial Velocity (Full Pipe &amp; Cross-Section)
          </h3>
          <p style={{ color: "#cbd5e1", fontSize: "1rem", marginTop: "0.3rem" }}>
            Toggle between cases, and drag along the pipe to explore ũₓ at any axial position, comparing actual velocity profile
            with FNO predicted velocity profile. 
          </p>
        </div>
        <VelocityProfilePanel />
      </div>


      <p style={pStyle}>
        <b>{navLink("fno-section", "Keep reading", false)} to get technical, otherwise <a href="#inverse">proceed to the next section</a> to learn about the inverse problem.</b>
      </p>


      <h3 id="fno-section" style={subheadStyle}>Fourier Neural Operator</h3>

      <p style={pStyle}>
        Classic neural networks learn a mapping{" "}
        {String.raw`\(f : \mathbb{R}^d \xrightarrow{} \mathbb{R}^m\)`} between
        finite-dimensional spaces. However, as done by Li et al. <Cite n={6} />, this FNO
        implementation learns a mapping between function spaces, or more
        specifically, two infinite dimensional spaces from a finite collection
        of observed input-output pairs. We try to approximate
      </p>

      <div style={eqStyle}>
        {String.raw`\[ G^\dagger : A(D; \mathbb{R}^{d_a}) \rightarrow U(D; \mathbb{R}^{d_u}), \]`}
      </div>

      <p style={pStyle}>
        where {String.raw`\(A\)`} is the space of input functions,{" "}
        {String.raw`\(D\)`} is the spatial domain (physical geometry in which
        the fluid flow exists), {String.raw`\(\mathbb{R}^{d_a}\)`} is the
        value at those coordinates, and {String.raw`\(U\)`} is the space of
        output functions of the same form <Cite n={6} />, with
      </p>

      <div style={eqStyle}>
        {String.raw`\[ G_\theta : A \rightarrow U, \theta \in \Theta, \]`}
      </div>

      <p style={pStyle}>
        where {String.raw`\(G_\theta\)`} is the neural operator,{" "}
        {String.raw`\(\theta\)`} represents the learnable parameters in the
        neural network (weights, biases), and {String.raw`\(\Theta\)`} is the
        finite-dimensional parameter space.
      </p>

      <p style={pStyle}>
        The important operation in each FNO layer is the application of an
        integral operator:
      </p>

      <div style={eqStyle}>
        {String.raw`\[ (\mathcal{K}v_t)(x) = \int_D \kappa(x, y)\, v_t(y)\, dy, \]`}
      </div>

      <p style={pStyle}>
        where {String.raw`\(x, y \in D \subset \mathbb{R}^3\)`} are
        three-dimensional spatial coordinates. This can be thought of as a
        learned weighted average: for each output point {String.raw`\(x\)`},
        the kernel {String.raw`\(\kappa(x, y)\)`} determines how much every
        other point {String.raw`\(y \in D\)`} contributes to the result,
        allowing the network to capture spatial dependencies. However, learning{" "}
        {String.raw`\(\kappa(x, y)\)`} directly is expensive, since it must
        account for every possible pair of points in the volume.
      </p>

      <p style={pStyle}>
        The FNO uses the convolution theorem to get around this. When the
        kernel is translation-invariant, (it depends only on the relative
        displacement {String.raw`\(x - y\)`} instead of absolute positions)
        the integral operator reduces to a 3D convolution. The convolution
        theorem then states that this is equivalent to pointwise multiplication
        in 3D Fourier space:
      </p>

      <div style={eqStyle}>
        {String.raw`\[ \left(\mathcal{K}(\phi)v_t\right)(x) = \mathcal{F}^{-1}\left(R_\phi \cdot \left(\mathcal{F}v_t\right)\right)(x), \]`}
      </div>

      <p style={pStyle}>
        where {String.raw`\(k = (k_1, k_2, k_3)\)`} is a three-dimensional
        frequency mode and {String.raw`\(\mathcal{F}\)`},{" "}
        {String.raw`\(\mathcal{F}^{-1}\)`} denote the 3D Fourier transform and
        its inverse. The FNO therefore learns {String.raw`\(R_\phi\)`} (a set
        of complex-valued weights in 3D Fourier space). To keep this
        manageable, only the lowest{" "}
        {String.raw`\(k_{\max,1} \times k_{\max,2} \times k_{\max,3}\)`} modes
        are retained; these capture the important, smoothly varying structure
        of the solution across all three spatial dimensions, while discarded
        high-frequency modes correspond to unimportant noise.
      </p>

      <p style={pStyle}>
        Each FNO layer computes this via a 3D Fast Fourier Transform (FFT),
        applies the learnable weights to the retained modes, and returns to
        physical space via an inverse 3D FFT. A pointwise linear
        transformation {String.raw`\(Wv_t(x)\)`} is added in parallel and the
        two are summed, then passed through a nonlinear activation function{" "}
        {String.raw`\(\sigma\)`}:
      </p>

      <div style={eqStyle}>
        {String.raw`\[ v_{t+1}(x) = \sigma\left(Wv_t(x) + \mathcal{F}^{-1}\left(R_\phi \cdot \mathcal{F}(v_t)\right)(x)\right). \]`}
      </div>

      <p style={pStyle}>
        The linear branch {String.raw`\(Wv_t(x)\)`} applies a learned matrix{" "}
        {String.raw`\(W\)`} pointwise at each spatial location, with no spatial
        mixing (preserving local detail that may be lost by the low-frequency
        truncation in the Fourier branch). The two are summed so that each
        layer captures both global structure (via the Fourier branch) and local
        structure (via {String.raw`\(W\)`}).
      </p>

      <p style={pStyle}>
        This is repeated across {String.raw`\(T\)`} layers (in this case,{" "}
        {String.raw`\(T=4\)`}) before a final projection to the output function
        space.
      </p>

    </div>
  );

  // ── Detail content: full-width below the two-column layout ────────────────
  const detail = (
    <div ref={contentRef} style={{ fontSize: "1.5rem", color: "#f8fafc", lineHeight: 1.7, padding: "0 0 2rem" }}>

      {/* ── FNO Architecture ── */}
      <h3 style={subheadStyle}>FNO Architecture</h3>

      <p style={pStyle}>
        See the figure below, pulled from by Li et al.'s <Cite n={6} />{" "}
        <em>Fourier Neural Operator for Parametric Partial Differential
        Equations</em>, for the full architecture of the Fourier neural
        operator.
      </p>

      <img
        src="/fno_architecture.png"
        alt="Architecture of the Fourier Neural Operator"
        style={{
          width: "100%",
          maxWidth: "900px",
          display: "block",
          margin: "0 auto 2rem",
          borderRadius: "8px",
          border: "1px solid rgba(255,255,255,0.08)",
        }}
      />

      {/* ── Non-Dimensionalization ── */}
      <h3 style={subheadStyle}>Non-Dimensionalization</h3>

      <p style={pStyle}>
        All fields are non-dimensionalized by the inlet velocity{" "}
        {String.raw`\(U_{in}\)`}:
      </p>

      <div style={eqStyle}>
        {String.raw`\[ \tilde{u} = \frac{u}{U_{in}}, \quad \tilde{p} = \frac{p}{U_{in}^2}. \]`}
      </div>

      <p style={pStyle}>
        The substitution of non-dimensional variables into the momentum equation
        yields a viscous coefficient of{" "}
        {String.raw`\(\frac{\nu}{U_{in} D}\)`}, which is the reciprocal of the
        Reynolds number ({String.raw`\(Re\)`}). The Reynolds number represents
        the ratio of inertial forces to viscous forces. In normalized variables,
        the momentum equation becomes:
      </p>

      <div style={eqStyle}>
        {String.raw`\[ (\tilde{u} \cdot \nabla)\tilde{u} = -\nabla\tilde{p} + \frac{1}{Re} \nabla^2 \tilde{u}. \]`}
      </div>

      <p style={pStyle}>
        This makes all outputs dimensionless and directly comparable across
        different ({String.raw`\(Re\)`}, {String.raw`\(U_{in}\)`}) conditions.
        The Fourier Neural Operator learns these normalized fields.
      </p>

      {/* ── Dataset representation ── */}
      <h3 style={subheadStyle}>Dataset Representation</h3>

      <p style={{ marginBottom: "1rem", fontSize: "1.5rem", color: "#f8fafc", lineHeight: 1.7, maxWidth: "none" }}>
        The input tensor{" "}
        {String.raw`\(X \in \mathbb{R}^{7 \times N_x \times N_y \times N_z}\)`}{" "}
        has seven channels that encode the geometry and physics parameters at
        every grid point. See below:
      </p>

      <table style={tableStyle}>
        <thead>
          <tr>
            <th style={thStyle}>Channel</th>
            <th style={thStyle}>Feature</th>
            <th style={thStyle}>Range</th>
            <th style={thStyle}>Description</th>
          </tr>
        </thead>
        <tbody>
          {[
            ["0", String.raw`\(\tilde{x}\)`,        "[0, 1]",   "Normalized axial coordinate"],
            ["1", String.raw`\(\tilde{y}\)`,        "[0, 1]",   "Normalized lateral coordinate"],
            ["2", String.raw`\(\tilde{z}\)`,        "[0, 1]",   "Normalized lateral coordinate"],
            ["3", "fluid mask",                     "{0, 1}",   "1 inside pipe, 0 outside"],
            ["4", "sphere mask",                    "{0, 1}",   "1 inside obstacle, 0 outside"],
            ["5", String.raw`\(\tilde{Re}\)`,        "[0, 1]",   String.raw`\((Re-100)/(1000-100)\)`],
            ["6", String.raw`\(\tilde{U_{in}}\)`,   "[0, 1]",   String.raw`\((U_{in}-0.1)/(1.0-0.1)\)`],
          ].map(([ch, feat, range, desc], i) => (
            <tr key={i} style={{ background: i % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent" }}>
              <td style={tdStyle}>{ch}</td>
              <td style={tdStyle}>{feat}</td>
              <td style={tdStyle}>{range}</td>
              <td style={tdStyle}>{desc}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <p style={{ marginBottom: "1rem", fontSize: "1.5rem", color: "#f8fafc", lineHeight: 1.7, maxWidth: "none" }}>
        The output tensor{" "}
        {String.raw`\(Y \in \mathbb{R}^{4 \times N_x \times N_y \times N_z}\)`}{" "}
        contains the equilibrium fields. See below:
      </p>

      <table style={tableStyle}>
        <thead>
          <tr>
            <th style={thStyle}>Channel</th>
            <th style={thStyle}>Field</th>
            <th style={thStyle}>Description</th>
          </tr>
        </thead>
        <tbody>
          {[
            ["0", String.raw`\(\tilde{u}_x\)`, "Axial velocity"],
            ["1", String.raw`\(\tilde{u}_y\)`, "Lateral velocity"],
            ["2", String.raw`\(\tilde{u}_z\)`, "Lateral velocity"],
            ["3", String.raw`\(\tilde{p}\)`,   "Mean-centered kinematic pressure"],
          ].map(([ch, field, desc], i) => (
            <tr key={i} style={{ background: i % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent" }}>
              <td style={tdStyle}>{ch}</td>
              <td style={tdStyle}>{field}</td>
              <td style={tdStyle}>{desc}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* ── Independent physical parameters ── */}
      <h3 style={subheadStyle}>Independent Physical Parameters</h3>

      <p style={pStyle}>
        The two independent physical parameters are the Reynolds number and
        inlet velocity:
      </p>

      <div style={eqStyle}>
        {String.raw`\[ Re = \frac{U_{in}D}{\nu}, \quad U_{in} \in [0.1, 1.0], \quad Re \in [100, 1000]. \]`}
      </div>

      <p style={pStyle}>
        The kinematic viscosity is derived as{" "}
        {String.raw`\(\nu = \frac{U_{in}D}{Re}\)`}, meaning{" "}
        {String.raw`\(Re\)`} is directly tied to viscosity. It is often the
        case in physical experiments to pick a fluid (fixing{" "}
        {String.raw`\(\nu\)`}) and changing the inlet velocity to observe what
        happens to {String.raw`\(Re\)`}. But in this ML scenario, we want the
        model to learn a specific range of {String.raw`\(Re\)`}; we then
        back-calculate what the {String.raw`\(\nu\)`} needs to be for the
        simulation. Treating {String.raw`\(Re\)`} as an independent variable
        makes it easier to produce well-representative training data, and
        therefore a model with uniform accuracy across the regime.
      </p>


    </div>
  );

  return (
    <>
      <Section
        id="forward"
        side="full"
        wide
        label="03 · Forward Model"
        title="Forward Operator Learning"
        text={text}
      >
        {detail}
      </Section>
    </>
  );
}

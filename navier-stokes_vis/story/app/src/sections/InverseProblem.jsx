import { useEffect } from "react";
import Section from "../components/Section.jsx";
import ReScatterPlot from "../components/ReScatterPlot.jsx";

const subheadStyle = {
  fontSize: "1.5rem",
  fontWeight: 700,
  color: "#cbd5e1",
  margin: "2.5rem 0 0.75rem",
};

const eqStyle = {
  margin: "0.5rem 0",
  overflowX: "auto",
  fontSize: "1.4rem",
};

const pStyle = { marginBottom: "0.9rem", fontSize: "1.5rem", color: "#f8fafc", lineHeight: 1.7, maxWidth: "none" };

function scrollAndHighlight(id) {
  const el = document.getElementById(id);
  if (!el) return;
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

const navLink = (id, label) => (
  <span
    onClick={() => scrollAndHighlight(id)}
    onMouseEnter={e => { e.currentTarget.style.textDecoration = "underline"; }}
    onMouseLeave={e => { e.currentTarget.style.textDecoration = "none"; }}
    style={{ color: "#60c8f5", cursor: "pointer", fontWeight: 600 }}
  >{label}</span>
);

export default function InverseProblem() {
  useEffect(() => {
    const tryTypeset = () => {
      if (window.MathJax?.startup?.promise) {
        const el = document.getElementById("inverse");
        window.MathJax.startup.promise
          .then(() => window.MathJax.typesetPromise(el ? [el] : []))
          .catch(console.error);
      } else {
        setTimeout(tryTypeset, 100);
      }
    };
    tryTypeset();
  }, []);

  // ── Text: scatter plot floats right, prose wraps around it ─────────────────
  const text = (
    <div style={{ fontSize: "1.5rem", color: "#f8fafc", lineHeight: 1.7 }}>
      <div id="re-scatter" style={{ float: "right", width: "45%", marginLeft: "3rem", marginBottom: "1.5rem" }}>
        <span className="section-chip" style={{ fontSize: "0.82rem", marginBottom: "0.4rem" }}>Results</span>
        <h3 style={{ fontSize: "1.2rem", color: "#f1f5f9", margin: "0.3rem 0 0.5rem" }}>
          Predicted vs True Re
        </h3>
        <ReScatterPlot />
      </div>
      <p style={pStyle}>
        While the forward surrogate produces the full 3D flow field from physical
        parameters, the inverse surrogate solves the inverse problem: given an
        observed steady-state velocity and pressure field, recover the Reynolds number{" "}
        {String.raw`\(Re\)`} ({String.raw`\(\nu = \frac{U_{in} \cdot D}{\text{Re}}\)`}) that produced it.
      </p>

      <h3 style={subheadStyle}>Architecture</h3>

      <p style={pStyle}>
        The inverse model is a 3D residual convolutional neural network. It consists of five residual blocks followed by global average pooling and a fully-connected head.
      </p>

      <h3 style={subheadStyle}>Input Representation</h3>

      <p style={pStyle}>
        The observed field is first made dimensionless by dividing by the inlet velocity{" "}
        {String.raw`\(U_{\mathrm{in}}\)`} and represented on the same structured{" "}
        {String.raw`\(265 \times 20 \times 20\)`} voxel grid used by the forward model's training data.
        Six channels are computed at each voxel:
      </p>

      <div style={eqStyle}>
        {String.raw`\[ \mathbf{v}(x) = \left( \frac{u_x}{U_{\mathrm{in}}},\; \frac{u_y}{U_{\mathrm{in}}},\; \frac{u_z}{U_{\mathrm{in}}},\; \frac{p}{U_{\mathrm{in}}^2},\; \frac{\partial p / \partial x}{U_{\mathrm{in}}^2},\; \frac{|\boldsymbol{\omega}|}{U_{\mathrm{in}}} \right) \]`}
      </div>

      <p style={pStyle}>
        where {String.raw`\(|\boldsymbol{\omega}|\)`} is the vorticity magnitude (measures the local rotation rate of fluid parcels).
        Voxels outside the fluid domain (pipe wall, sphere interior) are set to zero.
        Each channel is then z-score normalized using statistics computed from the training set
        (stabilizing gradient flow and preventing channels with larger absolute values from dominating during training).
        Dividing by {String.raw`\(U_{\mathrm{in}}\)`} before passing to the network removes the linear dependence on inlet velocity,
        allowing the network to focus on the Re-dependent structure of the flow.
      </p>

      <h3 style={subheadStyle}>Target Encoding</h3>

      <p style={pStyle}>
        The Reynolds number is encoded on a log scale and normalized to {String.raw`\([0, 1]\)`}:
      </p>

      <div style={eqStyle}>
        {String.raw`\[ \tilde{Re} = \frac{\log Re - \log 100}{\log 1000 - \log 100} \]`}
      </div>

      <p style={pStyle}>
        Log scaling is used because the flow structure varies much more
        rapidly at low {String.raw`\(Re\)`} than at high {String.raw`\(Re\)`}, so a uniform loss in log space penalises
        relative errors equally across the full range.
      </p>
      <div style={{ clear: "both" }} />
    </div>
  );

  // ── Detail: full-width below the two-column layout ────────────────────────
  const detail = (
    <div style={{ fontSize: "1.5rem", color: "#f8fafc", lineHeight: 1.7, padding: "0 0 2rem" }}>

      <h3 style={subheadStyle}>Loss Function</h3>

      <p style={pStyle}>
        The training objective combines a Huber loss with a soft boundary barrier:
      </p>

      <div style={eqStyle}>
        {String.raw`\[ \mathcal{L} = \mathcal{L}_{\mathrm{Huber}}(\delta{=}0.05) + 0.1 \cdot \left[\, \max(\tilde{Re}_{\mathrm{pred}} - 1,\; 0)^2 + \max(-\tilde{Re}_{\mathrm{pred}},\; 0)^2 \,\right] \]`}
      </div>

      <p style={pStyle}>
        The Huber loss is quadratic for small errors and linear for large
        ones, limiting the influence of outliers early in training. The barrier term
        softly penalizes predictions outside the valid normalized range {String.raw`\([0,1]\)`}.
      </p>

      <h3 style={subheadStyle}>Denormalization</h3>

      <p style={pStyle}>
        The denormalization step is seen below:
      </p>

      <div style={eqStyle}>
        {String.raw`\[ Re_{\mathrm{pred}} = \exp\!\left(\tilde{Re}_{\mathrm{pred}}\cdot(\log 1000 - \log 100) + \log 100\right) \]`}
      </div>

      <p style={pStyle}>
        You may look {navLink("re-scatter", "above")} to see a visualization of the predicted{" "}
        {String.raw`\(Re\)`} compared with the true {String.raw`\(Re\)`} for 10 cases.
        Hover over a point to see more information, such as the corresponding {String.raw`\(\nu\)`}.
      </p>
    </div>
  );

  return (
    <Section
      id="inverse"
      side="full"
      wide
      label="04 · Inverse Problem"
      title="The Inverse Problem"
      text={text}
    >
      {detail}
    </Section>
  );
}

import { useEffect, useRef } from "react";
import Section from "../components/Section.jsx";
import Cite from "../components/Cite.jsx";

const subheadStyle = {
  fontSize: "1.5rem",
  fontWeight: 700,
  color: "#cbd5e1",
  margin: "1.8rem 0 0.5rem",
};

const eqStyle = {
  margin: "0.75rem 0",
  overflowX: "auto",
  fontSize: "1.4rem",
};

export default function PhysicsFoundation() {
  const contentRef = useRef(null);

  useEffect(() => {
    const el = contentRef.current;
    if (!el) return;
    const tryTypeset = () => {
      if (window.MathJax?.startup?.promise) {
        window.MathJax.startup.promise
          .then(() => window.MathJax.typesetPromise([el]))
          .catch(console.error);
      } else {
        setTimeout(tryTypeset, 100);
      }
    };
    tryTypeset();
  }, []);

  const text = (
    <div ref={contentRef} className="physics-content">
      <p>
        The 3D incompressible steady-state Navier–Stokes equations describe the
        motion of viscous flow formed by a fluid. The equations represent the steady state (system at equilibrium), and are as follows:
      </p>

      <h3 style={subheadStyle}>1. Continuity Equation</h3>

      <div style={eqStyle}>
        {String.raw`\[ \nabla \cdot \mathbf{u} = 0, \]`}
      </div>

      <p>
        where {String.raw`\(\mathbf{u} = (u_x, u_y, u_z)\)`} is the velocity
        vector field. This expands to:
      </p>

      <div style={eqStyle}>
        {String.raw`\[
          \frac{\partial u_x}{\partial x}
          + \frac{\partial u_y}{\partial y}
          + \frac{\partial u_z}{\partial z} = 0
        \]`}
      </div>

      <p>
        If the sum of these three rates (the rate at which the velocity
        component in a specific direction ({String.raw`\(u_i\)`}) changes as
        you move along that same axis ({String.raw`\(i\)`})) is zero, it means
        that any fluid that speeds up in one direction must slow down or spread
        out in the others to keep the total volume constant. In other words,
        there is no accumulation or depletion of mass, so mass is conserved
        within the domain <Cite n={2} />. This concept defines incompressibility.
      </p>

      <h3 style={{ ...subheadStyle, marginTop: "3rem" }}>2. Momentum Equation</h3>

      <div style={eqStyle}>
        {String.raw`\[ (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \nu\,\nabla^2 \mathbf{u} \]`}
      </div>

      <p>This expands to:</p>

      <div style={eqStyle}>
        {String.raw`\[
          \begin{aligned}
            & u_x \frac{\partial u_x}{\partial x}
              + u_y \frac{\partial u_x}{\partial y}
              + u_z \frac{\partial u_x}{\partial z}
              = -\frac{\partial p}{\partial x}
              + \nu \left(
                  \frac{\partial^2 u_x}{\partial x^2}
                + \frac{\partial^2 u_x}{\partial y^2}
                + \frac{\partial^2 u_x}{\partial z^2}
                \right) \\[8pt]
            & u_x \frac{\partial u_y}{\partial x}
              + u_y \frac{\partial u_y}{\partial y}
              + u_z \frac{\partial u_y}{\partial z}
              = -\frac{\partial p}{\partial y}
              + \nu \left(
                  \frac{\partial^2 u_y}{\partial x^2}
                + \frac{\partial^2 u_y}{\partial y^2}
                + \frac{\partial^2 u_y}{\partial z^2}
                \right) \\[8pt]
            & u_x \frac{\partial u_z}{\partial x}
              + u_y \frac{\partial u_z}{\partial y}
              + u_z \frac{\partial u_z}{\partial z}
              = -\frac{\partial p}{\partial z}
              + \nu \left(
                  \frac{\partial^2 u_z}{\partial x^2}
                + \frac{\partial^2 u_z}{\partial y^2}
                + \frac{\partial^2 u_z}{\partial z^2}
                \right),
          \end{aligned}
        \]`}
      </div>

      <p>
        where the left hand side of the equations (
        {String.raw`\((\mathbf{u}\cdot\nabla)\mathbf{u}\)`}) are the convective
        terms, describing the physical process that occurs in a flow in which
        some property is transported by the ordered motion of the flow <Cite n={3} />{" "} 
        (momentum carried by the flow itself).
      </p>

      <p style={{ marginTop: "0.75rem" }}>
        On the right hand side, the pressure gradient terms (
        {String.raw`\(-\nabla p\)`}) describe the kinematic pressure's (
        {String.raw`\(\frac{P}{\rho}\)`}, where {String.raw`\(\rho\)`} =
        density) force, which accelerates fluid from areas of high pressure to
        areas of low pressure.
      </p>

      <p style={{ marginTop: "0.75rem" }}>
        Further right are the diffusion terms (
        {String.raw`\(\nu\,\nabla^2\mathbf{u}\)`}). Diffusion is a physical
        process that occurs in a flow in which some property is transported by
        the random motion of the molecules of the fluid <Cite n={3} />. This 
        causes quantities to spread from regions of high concentration to regions 
        of low concentration. In the Navier–Stokes equations, diffusion represents 
        the viscous spreading of momentum through the fluid. The Laplacian
        operator ({String.raw`\(\nabla^2\)`}) is the divergence of the
        gradient, and looks at the curvature of the velocity profile.
        Kinematic viscosity ({String.raw`\(\nu\)`}) is the scaling factor for
        the resistance. A high kinematic viscosity results in the viscous term
        dominating; any spikes in velocity are quickly smoothed out, resulting
        in laminar flow. A low viscous term is weak compared to the convection
        terms, which allows velocity spikes to persist, resulting in more
        turbulent flow.
      </p>
      <h3 style={{ ...subheadStyle, marginTop: "3rem" }}>3. Boundary Conditions</h3>

      <p>
        This simulation conforms to the conditions of Poiseuille flow, which
        describes the steady, axisymmetric flow in a long, circular pipe. The
        velocity field {String.raw`\(\mathbf{u}\)`} is driven by the pressure
        gradient vector {String.raw`\(-\nabla p\)`}. This implies that a change
        in pressure along any spatial coordinate {String.raw`\(x_i\)`} results
        in a corresponding acceleration in that direction <Cite n={4} />. To see
        the characteristic parabolic velocity distribution of Poiseuille flow, 
        navigate to the interactive <span
          onClick={() => window.dispatchEvent(new CustomEvent('navigate-to', { detail: { sectionId: 'forward', scrollTo: 'velocity-profile' } }))}
          onMouseEnter={e => { e.currentTarget.style.textDecoration = "underline"; }}
          onMouseLeave={e => { e.currentTarget.style.textDecoration = "none"; }}
          style={{ color: "#60c8f5", cursor: "pointer", fontWeight: 600 }}
        >velocity profile visualization</span> in the next section.
      </p>

      <p style={{ marginTop: "0.75rem" }}>
        With this in mind, the following describes the assumed boundary
        conditions:
      </p>

      <ul style={{ marginTop: "0.5rem", paddingLeft: "1.5rem", lineHeight: 1.9 }}>
        <li>
          <strong>Inlet:</strong> The inlet ({String.raw`\(x = 0\)`}) assumes a
          Poiseuille profile with mean velocity {String.raw`\(U_{\mathrm{in}}\)`}.
        </li>
        <li>
          <strong>Outlet:</strong> The outlet ({String.raw`\(x = L\)`}) has
          zero-gradient pressure (fixed{" "}
          {String.raw`\(p_{\mathrm{outlet}} = 0\)`}).
        </li>
        <li>
          <strong>Pipe wall:</strong> No-slip. The no-slip condition is that the
          tangential flow relative to the boundary is zero <Cite n={5} />. In other words, the
          fluid particles along the pipe walls have no velocity ({String.raw`\(u = 0\)`})
          (a boundary layer forms around the wall).
        </li>
        <li>
          <strong>Spherical obstacle surface:</strong> No-slip ({String.raw`\(u = 0\)`}).
        </li>
      </ul>
      <p style={{ marginTop: "0.75rem" }} >{" "}
        Now that you understand the basics of fluid dynamics, you may skip to the{" "} 
        <a href="#demo">Interactive Demo</a>, or proceed to the{" "} 
        <a href="#forward">next</a> {" "}section to continue learning.
      </p>
    </div>
  );

  return (
    <Section
      id="physics"
      side="full"
      wide
      label="02 · Physics"
      title="Physics Foundation"
      text={text}
    />
  );
}

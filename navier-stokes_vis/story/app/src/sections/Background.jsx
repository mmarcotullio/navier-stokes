import Section from "../components/Section.jsx";
import Cite from "../components/Cite.jsx";

export default function Background() {
  const text = (
    <div className="background-content">
      <p>
        Computational Fluid Dynamics is central to modern engineering, with
        applications in aerospace, electronics cooling, automotive design, energy
        systems, and more. By capturing how fluids interact with physical systems,
        CFD gives scientists and engineers deeper insight into the phenomena that
        inform design decisions. However, before a model can be adopted into the
        scientific and engineering workflow, it must be shown to be both accurate
        and computationally efficient.
      </p>

      <p>
        The current standard in CFD modeling is the Finite Element Method, a
        general numerical method for solving PDEs in two- or three-space
        dimensions. This is done through discretization in the space dimension,
        forming a mesh composed of finite elements that represent the numerical
        domain. The equations that model those elements are then gathered into a
        larger system of equations that models the entire problem.
      </p>

      <p>
        Solving these systems of equations is computationally expensive. Achieving
        scientific- or engineering-grade accuracy demands extremely fine meshes and
        highly complex geometries, often requiring lengthy runtimes or even
        supercomputing resources. This makes rapid design iteration difficult, and
        places CFD largely out of reach for non-experts who might otherwise explore
        fluid phenomena in an accessible scientific visualization environment.
      </p>

      <p>
        Further, an emerging subject in the scientific modeling and simulation
        domain is inverse modeling. "Inverse problems concern the recovery of
        unknown parameters from indirect, and often noisy, observations . . .
        [the] inherent nonlinearity and the high- or infinite-dimensional nature of
        many practical inverse problems can introduce significant mathematical and
        computational challenges" <Cite n={1} />. Inverse problems are often ill-posed, and
        "the inverse problem solution's sensitivity to noise . . . makes naive
        inversion schemes unstable and unreliable" <Cite n={1} />.
      </p>

      <p>
        Here, through a simple-implementation prototype, I tell two stories: (1) for the 
        initiated; <b>how the problems of cost and inverse modeling can be addressed</b>, and (2) 
        for the non-experts; <b>the basics of CFD</b>, and <b>how fluid behavior changes under 
        different physical parameters</b> through interactive parameter exploration.
      </p>

      <p>
        For the non-experts, you may read the{" "}
        <a href="#physics">Physics Foundation</a> section to brush up on the
        physics which govern fluid dynamics, or you may skip directly to the{" "}
        <a href="#demo">Interactive Demo</a>.
      </p>

      <p>
        For those interested in the larger problem: you'll encounter a brief
        description of how I implemented machine learning models to achieve
        near-instant CFD simulation, and how I tackle the inverse problem. As such,
        two sections lie beyond the <a href="#physics">Physics Foundation</a> section: the{" "}
        <a href="#forward">Forward Model</a> and the{" "}
        <a href="#inverse">Inverse Problem</a>. Once these have been digested, you
        may proceed to the{" "}
        <a href="#demo">Interactive Demo</a> to see the two in real-time
        action.
      </p>
    </div>
  );

  return (
    <Section
      id="background"
      side="full"
      wide
      label="01 · Background"
      title="Background"
      text={text}
    />
  );
}

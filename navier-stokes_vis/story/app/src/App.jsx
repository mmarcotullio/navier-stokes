import { useState, useEffect, useRef } from "react";
import Background from "./sections/Background.jsx";
import PhysicsFoundation from "./sections/PhysicsFoundation.jsx";
import ForwardOperator from "./sections/ForwardOperator.jsx";
import InverseProblem from "./sections/InverseProblem.jsx";
import MainVisualization from "./sections/MainVisualization.jsx";
import References from "./sections/References.jsx";
import SectionNav from "./components/SectionNav.jsx";

const SECTIONS = [
  { id: "background", label: "01 · Background",       title: "Background" },
  { id: "physics",    label: "02 · Physics",          title: "Physics Foundation" },
  { id: "forward",    label: "03 · Forward Model",    title: "Forward Operator Learning" },
  { id: "inverse",    label: "04 · Inverse Problem",  title: "The Inverse Problem" },
  { id: "demo",       label: "05 · Interactive Demo", title: "Real-Time CFD Simulation Explorer" },
];

const SECTION_COMPONENTS = {
  background: Background,
  physics:    PhysicsFoundation,
  forward:    ForwardOperator,
  inverse:    InverseProblem,
  demo:       MainVisualization,
};

const V = { display: "block", width: "100%", height: 130 };

const CARD_VISUALS = {
  // Background image
  background: (
    <img src="/background_img.jpg" alt="Background" style={{ ...V, objectFit: "cover" }} />
  ),

  // Physics image
  physics: (
    <img src="/physics.jpg" alt="Physics Foundation" style={{ ...V, objectFit: "cover" }} />
  ),

  // Neural network: params → NN → output field
  forward: (
    <img src="/forward.png" alt="Forward Operator Learning" style={{ ...V, objectFit: "cover" }} />
  ),

  // Neural network: observed field → NN → recovered params
  inverse: (
    <img src="/inverse.png" alt="The Inverse Problem" style={{ ...V, objectFit: "cover" }} />
  ),

  // Interactive vis image
  demo: (
    <img src="/interactive_vis.png" alt="Interactive Demo" style={{ ...V, objectFit: "cover" }} />
  ),
};

export default function App() {
  const [currentSection, setCurrentSection] = useState(null);
  const [scrollProgress, setScrollProgress] = useState(0);
  const pendingRef = useRef(null);
  const pendingScrollRef = useRef(null);

  useEffect(() => {
    if (currentSection === null) { setScrollProgress(0); return; }
    const update = () => {
      const el = document.documentElement;
      const scrollable = el.scrollHeight - el.clientHeight;
      setScrollProgress(scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0);
    };
    window.addEventListener("scroll", update, { passive: true });
    update();
    return () => window.removeEventListener("scroll", update);
  }, [currentSection]);

  const navigate = (id) => {
    setCurrentSection(id);
    window.scrollTo(0, 0);
  };

  // Intercept anchor links in section prose (e.g. <a href="#physics">)
  useEffect(() => {
    const sectionIds = new Set(SECTIONS.map(s => s.id));
    const handle = (e) => {
      const a = e.target.closest('a[href^="#"]');
      if (!a) return;
      const id = a.getAttribute('href').slice(1);
      if (sectionIds.has(id)) {
        e.preventDefault();
        navigate(id);
      }
    };
    document.addEventListener('click', handle);
    return () => document.removeEventListener('click', handle);
  }, []);

  // Cross-section navigate + scroll-to (dispatched by other sections)
  useEffect(() => {
    const handle = (e) => {
      const { sectionId, scrollTo } = e.detail;
      pendingScrollRef.current = scrollTo || null;
      navigate(sectionId);
    };
    window.addEventListener('navigate-to', handle);
    return () => window.removeEventListener('navigate-to', handle);
  }, []);

  // Execute pending scroll after section renders
  useEffect(() => {
    if (!pendingScrollRef.current) return;
    const target = pendingScrollRef.current;
    pendingScrollRef.current = null;
    setTimeout(() => {
      const el = document.getElementById(target);
      if (!el) return;
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      el.style.transition = 'box-shadow 0.3s';
      el.style.boxShadow = '0 0 0 3px rgba(96,200,245,0.7), 0 0 20px rgba(96,200,245,0.3)';
      el.style.borderRadius = '8px';
      setTimeout(() => {
        el.style.transition = 'box-shadow 1.5s';
        el.style.boxShadow = '';
        setTimeout(() => { el.style.transition = ''; el.style.borderRadius = ''; }, 1500);
      }, 1000);
    }, 400);
  }, [currentSection]);

  // When a citation is clicked inside a section, navigate home then open that ref
  useEffect(() => {
    const handle = (e) => {
      if (currentSection !== null) {
        pendingRef.current = e.detail;
        navigate(null);
      }
    };
    window.addEventListener('open-ref', handle);
    return () => window.removeEventListener('open-ref', handle);
  }, [currentSection]);

  // After landing on home with a pending ref, re-dispatch so References.jsx can handle it
  useEffect(() => {
    if (currentSection === null && pendingRef.current !== null) {
      const n = pendingRef.current;
      pendingRef.current = null;
      setTimeout(() => {
        window.dispatchEvent(new CustomEvent('open-ref', { detail: n }));
      }, 400);
    }
  }, [currentSection]);

  const header = (
    <header style={{ padding: "7vh 8vw 2vh", textAlign: "center" }}>
      <h1 style={{ fontSize: "clamp(1.4rem, 2.2vw, 2.5rem)", fontWeight: 700, color: "#f1f5f9" }}>
        Forward and Inverse Modeling of the 3D Incompressible <br/>(Steady-State) Navier–Stokes Equations with Neural Networks
      </h1>
      <h2 style={{ fontSize: "clamp(1.2rem, 2vw, 2rem)", fontWeight: 400, color: "#94a3b8", marginTop: "1rem" }}>
        A Story in Machine Learning-Driven Computational Fluid Dynamics
      </h2>
    </header>
  );

  if (currentSection === null) {
    return (
      <div key="home" className="view-enter" style={{ minHeight: "100vh" }}>
        {header}
        <div className="section-divider" />
        <div className="home-cards">
          {SECTIONS.map((s) => (
            <div key={s.id} className="home-card" onClick={() => navigate(s.id)}>
              <div style={{ overflow: "hidden", borderRadius: "14px 14px 0 0" }}>
                {CARD_VISUALS[s.id]}
              </div>
              <div style={{ padding: "1.4rem 1.75rem 1.6rem", display: "flex", flexDirection: "column", flex: 1 }}>
                <span className="section-chip" style={{ fontSize: "0.82rem", marginBottom: "0.85rem" }}>{s.label}</span>
                <h3 style={{ fontSize: "1.3rem", fontWeight: 700, color: "#f1f5f9", marginBottom: "auto" }}>{s.title}</h3>
                <span style={{ fontSize: "0.88rem", color: "#38bdf8", marginTop: "1.25rem", display: "block" }}>Read →</span>
              </div>
            </div>
          ))}
        </div>
        <div style={{ padding: "0.4rem 8vw 2vh", display: "flex", alignItems: "center", gap: "0.4rem" }}>
          <span style={{ fontSize: "0.95rem", color: "#38bdf8", letterSpacing: "0.04em" }}>↑ Start here</span>
        </div>
        <References />
      </div>
    );
  }

  const idx = SECTIONS.findIndex(s => s.id === currentSection);
  const ActiveComponent = SECTION_COMPONENTS[currentSection];

  const hasPrev = idx > 0;
  const hasNext = idx < SECTIONS.length - 1;

  return (
    <div key={currentSection} className="view-enter" style={{ minHeight: "100vh" }}>
      <div className="reading-header">
        <span className="reading-home" onClick={() => navigate(null)}>← Table of Contents</span>
        <span className="reading-progress">Chapter {idx + 1} of {SECTIONS.length} · {SECTIONS[idx].title}</span>
        {hasNext
          ? <span className="reading-home" onClick={() => navigate(SECTIONS[idx + 1].id)}>{SECTIONS[idx + 1].title} →</span>
          : <span style={{ visibility: "hidden" }}>placeholder</span>
        }
        <div className="reading-progress-bar" style={{ width: `${scrollProgress}%` }} />
      </div>
      <ActiveComponent />
      <SectionNav
        onHome={() => navigate(null)}
        onPrev={() => navigate(SECTIONS[idx - 1].id)}
        onNext={() => navigate(SECTIONS[idx + 1].id)}
        hasPrev={hasPrev}
        hasNext={hasNext}
        prevLabel={hasPrev ? SECTIONS[idx - 1].title : ""}
        nextLabel={hasNext ? SECTIONS[idx + 1].title : ""}
      />
    </div>
  );
}

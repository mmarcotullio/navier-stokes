export default function SectionNav({ onPrev, onNext, onHome, hasPrev, hasNext, prevLabel, nextLabel }) {
  const btn = (onClick, children) => (
    <button
      onClick={onClick}
      style={{
        background: "transparent",
        border: "1px solid rgba(56,189,248,0.3)",
        color: "#94a3b8",
        fontSize: "0.95rem",
        padding: "0.55rem 1.4rem",
        borderRadius: "8px",
        cursor: "pointer",
        transition: "border-color 0.15s, color 0.15s",
      }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = "#38bdf8"; e.currentTarget.style.color = "#38bdf8"; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = "rgba(56,189,248,0.3)"; e.currentTarget.style.color = "#94a3b8"; }}
    >
      {children}
    </button>
  );

  return (
    <div className="section-nav">
      {btn(onHome, "⌂ Home")}
      {hasPrev && btn(onPrev, `← ${prevLabel}`)}
      {hasNext && btn(onNext, `${nextLabel} →`)}
    </div>
  );
}

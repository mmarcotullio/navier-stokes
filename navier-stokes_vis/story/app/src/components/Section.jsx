/**
 *   reusable scrollytelling section wrapper.
 *
 *   id       string        – anchor id
 *   side     "left" | "right" | "full"
 *   label    string        – e.g. "01 · Physics"
 *   title    string        – section heading
 *   text     ReactNode     – body text block (shown on the text side)
 *   visual   ReactNode?    – visualization block (shown on the visual side)
 *   children ReactNode?    – extra content below the columns (full width)
 */
export default function Section({ id, side, label, title, text, visual, children, textFlex, wide }) {
  const isLeft  = side === "left";
  const isFull  = side === "full";

  const sectionStyle = {
    padding: "6vh 8vw",
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    gap: "2rem",
  };

  const columnsStyle = {
    display: "flex",
    flexDirection: isFull ? "column" : (isLeft ? "row" : "row-reverse"),
    gap: "4rem",
    alignItems: "flex-start",
  };

  const textStyle = {
    flex: isFull ? "unset" : (textFlex ?? "0 0 38%"),
    maxWidth: isFull ? (wide ? "unset" : "72ch") : "unset",
    alignSelf: isFull ? "center" : "flex-start",
    width: isFull ? "100%" : "unset",
    display: "flex",
    flexDirection: "column",
    gap: "1.1rem",
  };

  const visualStyle = {
    flex: 1,
    minWidth: 0,
  };

  return (
    <section id={id} style={sectionStyle}>
      <div className="section-columns" style={columnsStyle}>
        {/* Text column */}
        <div style={textStyle}>
          {label && <span className="section-chip">{label}</span>}
          {title && (
            <h2 style={{ fontSize: "clamp(2rem, 3vw, 3rem)", marginBottom: "0.25rem" }}>
              {title}
            </h2>
          )}
          {text}
        </div>

        {/* Visual column */}
        {visual && !isFull && (
          <div style={visualStyle}>
            {visual}
          </div>
        )}
      </div>

      {/* Full-width slot below columns (used when side="full" has a visual) */}
      {visual && isFull && (
        <div style={{ width: "100%" }}>
          {visual}
        </div>
      )}

      {/* Extra full-width content */}
      {children && (
        <div style={{ width: "100%" }}>
          {children}
        </div>
      )}
    </section>
  );
}

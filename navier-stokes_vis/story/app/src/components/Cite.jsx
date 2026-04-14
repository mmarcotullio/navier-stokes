export default function Cite({ n }) {
  return (
    <a
      href={`#ref-${n}`}
      onClick={e => {
        e.preventDefault();
        window.dispatchEvent(new CustomEvent('open-ref', { detail: n }));
      }}
      style={{ color: "#60c8f5", textDecoration: "none", fontWeight: 600 }}
    >[{n}]</a>
  );
}

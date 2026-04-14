import { useEffect, useRef, useState } from "react";

// Scientific notation with Unicode superscripts, e.g. "1.00 × 10⁻³"
const _sup = ['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹'];
function sciStr(v) {
  if (v == null || !isFinite(v)) return "—";
  if (v === 0) return "0";
  const [mStr, eStr] = Math.abs(v).toExponential(2).split('e');
  const exp = parseInt(eStr, 10);
  const e   = String(Math.abs(exp)).split('').map(d => _sup[+d]).join('');
  return `${v < 0 ? '-' : ''}${mStr} × 10${exp < 0 ? '⁻' : ''}${e}`;
}

export default function InverseToolPanel({ Re, uIn, res, onVisualizeInverse, onResetInverse }) {
  const [status, setStatus] = useState("idle");   // "idle"|"loading"|"done"|"error"
  const [result, setResult] = useState(null);
  const [errMsg, setErrMsg] = useState("");
  const [visLoading, setVisLoading] = useState(false);
  const abortRef = useRef(null);
  const panelRef = useRef(null);

  useEffect(() => {
    if (status !== "done" && panelRef.current) {
      panelRef.current.style.height = "";
    }
  }, [status]);

  // Reset whenever Re or uIn changes
  useEffect(() => {
    abortRef.current?.abort();
    setStatus("idle");
    setResult(null);
    setErrMsg("");
    onResetInverse?.();
  }, [Re, uIn]);

  const handleRun = async () => {
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setStatus("loading");
    setResult(null);
    setErrMsg("");

    try {
      const resp = await fetch(`/api/inverse?Re=${Re}&U_in=${uIn}`, {
        signal: ctrl.signal,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      if (data.error) throw new Error(data.error);
      setResult(data);
      setStatus("done");
    } catch (err) {
      if (err.name !== "AbortError") {
        setErrMsg(err.message);
        setStatus("error");
      }
    }
  };

  const handleVisualize = async () => {
    setVisLoading(true);
    try {
      const resp = await fetch(`/api/predict?Re=${result.re_pred}&U_in=${uIn}&Ny=${res}&Nz=${res}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      if (data.error) throw new Error(data.error);
      onVisualizeInverse?.(data.slice, result.re_pred);
    } catch (_err) {
      // silently ignore — forward model errors are secondary here
    } finally {
      setVisLoading(false);
    }
  };

  const dynPanel = status === "done"
    ? { ...panelStyle, resize: "vertical", overflow: "auto", minHeight: 200, maxHeight: 520 }
    : panelStyle;

  return (
    <div ref={panelRef} style={dynPanel}>
      <div style={titleStyle}>Inverse Modeling Tool</div>

      {status === "idle" && (
        <button style={btnStyle} onClick={handleRun}>
          Recover Parameter
        </button>
      )}

      {status === "loading" && (
        <div style={loadingRow}>
          <div style={spinnerStyle} />
          <span style={loadingText}>Running inference…</span>
        </div>
      )}

      {status === "done" && result && (
        <div style={resultsStyle}>
          <Row label="Predicted Re" value={result.re_pred.toFixed(1)} />
          <Row label="Actual Re"    value={result.re_true.toFixed(1)} />
          <Row label="Re residual"   value={result.re_mae.toFixed(1)}  accent />
          <div style={dividerStyle} />
          <Row label="Predicted ν"  value={`${sciStr(result.nu_pred)} m²/s`} mono />
          <Row label="Actual ν"     value={`${sciStr(result.nu_true)} m²/s`} mono />
          <Row label="ν residual"   value={`${sciStr(result.nu_mae)} m²/s`}  mono accent />
          <div style={dividerStyle} />
          <Row label="Predicted ν"  value={`${(result.nu_pred * 1e6).toFixed(2)} cSt`} mono />
          <Row label="Actual ν"     value={`${(result.nu_true * 1e6).toFixed(2)} cSt`} mono />
          <Row label="ν residual"   value={result.nu_mae != null ? `${(result.nu_mae * 1e6).toFixed(2)} cSt` : "—"} mono accent />
          <div style={{ display: "flex", gap: 8, marginTop: 16 }}>
            <button style={{ ...btnStyle, flex: 1 }} onClick={handleRun}>
              Re-run
            </button>
            <button style={{ ...btnStyle, flex: 1 }} onClick={() => { setStatus("idle"); setResult(null); onResetInverse?.(); }}>
              Reset
            </button>
          </div>
          <button
            style={{ ...btnStyle, marginTop: 8, opacity: visLoading ? 0.6 : 1 }}
            onClick={handleVisualize}
            disabled={visLoading}
          >
            {visLoading ? "Loading…" : "Visualize Inverse Model"}
          </button>
        </div>
      )}

      {status === "error" && (
        <div>
          <div style={errorText}>{errMsg}</div>
          <button style={{ ...btnStyle, marginTop: 14 }} onClick={handleRun}>
            Retry
          </button>
        </div>
      )}
    </div>
  );
}

function Row({ label, value, accent, mono }) {
  return (
    <div style={rowStyle}>
      <span style={labelStyle}>{label}</span>
      <span style={{
        ...valueStyle,
        color: accent ? "#f472b6" : "#e2e8f0",
        fontFamily: mono ? "monospace" : "inherit",
        fontSize: mono ? 15 : 18,
      }}>
        {value}
      </span>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────
const panelStyle = {
  position: "absolute",
  top: 16,
  right: 16,
  zIndex: 30,
  width: 420,
  background: "rgba(10,12,20,0.92)",
  backdropFilter: "blur(8px)",
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 14,
  padding: "20px 20px 18px",
  color: "#e2e8f0",
  fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
};

const titleStyle = {
  fontSize: 18,
  fontWeight: 700,
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  color: "#7dd3fc",
  marginBottom: 16,
};

const btnStyle = {
  width: "100%",
  padding: "12px 0",
  borderRadius: 8,
  border: "1px solid rgba(56,189,248,0.4)",
  background: "rgba(56,189,248,0.1)",
  color: "#38bdf8",
  fontSize: 18,
  fontWeight: 600,
  cursor: "pointer",
  letterSpacing: "0.04em",
};

const loadingRow = {
  display: "flex",
  alignItems: "center",
  gap: 12,
};

const spinnerStyle = {
  width: 28,
  height: 28,
  border: "3px solid rgba(56,189,248,0.2)",
  borderTop: "3px solid #38bdf8",
  borderRadius: "50%",
  animation: "spin 0.9s linear infinite",
  flexShrink: 0,
};

const loadingText = {
  fontSize: 18,
  color: "#94a3b8",
};

const resultsStyle = {
  display: "flex",
  flexDirection: "column",
  gap: 9,
};

const rowStyle = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "baseline",
  gap: 10,
};

const labelStyle = {
  fontSize: 16,
  color: "#64748b",
  whiteSpace: "nowrap",
};

const valueStyle = {
  fontSize: 18,
  fontVariantNumeric: "tabular-nums",
  fontWeight: 600,
  textAlign: "right",
};

const dividerStyle = {
  borderTop: "1px solid rgba(255,255,255,0.07)",
  margin: "7px 0",
};

const errorText = {
  fontSize: 16,
  color: "#f87171",
  wordBreak: "break-word",
};

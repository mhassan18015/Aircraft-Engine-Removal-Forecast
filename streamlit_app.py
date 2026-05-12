"""Streamlit frontend for the A320NEO Engine RCL Forecast.

Wraps the inference helpers from api_inference.py so the same LSTM/GRU/CNN
ensemble, sensor-driven degradation classifier, and Cox-PH hazard score are
served in-process — no FastAPI server, no localhost calls. Designed to deploy
as a single app on Streamlit Community Cloud.

Run locally:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import io
import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Aircraft Engine RCL Forecast",
    page_icon="✈",
    layout="centered",
)


# ---------------------------------------------------------------------------
# Model + helper loading (cached across reruns).
# Importing api_inference triggers Keras model loads + scaler reads, so we do
# it inside a cached function to keep cold-start cost paid only once per
# container lifetime.
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading LSTM / GRU / CNN models…")
def _load_inference_module():
    import api_inference as api  # noqa: WPS433 — deliberate lazy import
    return api


api = _load_inference_module()


# ---------------------------------------------------------------------------
# CSV → list[FlightRow-dict] conversion. Mirrors the JS rowToFlight() in
# index.html so users can keep uploading the same CSV format.
# ---------------------------------------------------------------------------
RAW_SENSOR_COLS = [
    "egt_probe_average", "fuel_flw", "core_spd",
    "oil_temp", "oil_pres", "pt2_(fan_inlet_tot_pres)",
]


def _row_to_flight(row: dict) -> dict:
    def num(k: str) -> float:
        v = row.get(k)
        if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
            raise ValueError(f'Missing or non-numeric value for column "{k}".')
        return float(v)

    def intv(k: str) -> int:
        v = row.get(k)
        if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
            return 0
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0

    # Cyclical time encodings — prefer pre-computed columns, fall back to deriving
    # from flight_datetime_c using pandas (Monday=0..Sunday=6, matching training).
    if row.get("month_sin") not in (None, "") and not (isinstance(row.get("month_sin"), float) and math.isnan(row["month_sin"])):
        month_sin = num("month_sin")
        month_cos = num("month_cos")
        dow_sin = num("dayofweek_sin")
        dow_cos = num("dayofweek_cos")
    elif row.get("flight_datetime_c") not in (None, ""):
        ts = pd.to_datetime(row["flight_datetime_c"], errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Could not parse flight_datetime_c: {row['flight_datetime_c']!r}")
        month = ts.month
        day_of_week = ts.dayofweek  # Mon=0..Sun=6
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)
        dow_sin = math.sin(2 * math.pi * day_of_week / 7)
        dow_cos = math.cos(2 * math.pi * day_of_week / 7)
    else:
        raise ValueError("Row needs either month_sin/cos+dayofweek_sin/cos or flight_datetime_c.")

    # Flight-phase one-hots
    if "flight_phase_CRUISE" in row or "flight_phase_TAKEOFF" in row:
        phase_cruise = intv("flight_phase_CRUISE")
        phase_takeoff = intv("flight_phase_TAKEOFF")
    elif row.get("flight_phase"):
        phase_cruise = 1 if row["flight_phase"] == "CRUISE" else 0
        phase_takeoff = 1 if row["flight_phase"] == "TAKEOFF" else 0
    else:
        raise ValueError("Row needs either flight_phase or flight_phase_CRUISE/_TAKEOFF.")

    # engtype one-hots
    if "engtype_LEAP-1A26" in row or "engtype_LEAP-1A26E1" in row:
        et_a26 = intv("engtype_LEAP-1A26")
        et_a26e1 = intv("engtype_LEAP-1A26E1")
    elif row.get("eng_type"):
        et_a26 = 1 if row["eng_type"] == "LEAP-1A26" else 0
        et_a26e1 = 1 if row["eng_type"] == "LEAP-1A26E1" else 0
    else:
        raise ValueError("Row needs either eng_type or engtype_LEAP-1A26/_E1 columns.")

    return {
        "egt_probe_average": num("egt_probe_average"),
        "fuel_flw": num("fuel_flw"),
        "core_spd": num("core_spd"),
        "oil_temp": num("oil_temp"),
        "oil_pres": num("oil_pres"),
        "pt2_(fan_inlet_tot_pres)": num("pt2_(fan_inlet_tot_pres)"),
        "egt_probe_average_norm": num("egt_probe_average_norm"),
        "fuel_flw_norm": num("fuel_flw_norm"),
        "core_spd_norm": num("core_spd_norm"),
        "oil_temp_norm": num("oil_temp_norm"),
        "oil_pres_norm": num("oil_pres_norm"),
        "pt2_(fan_inlet_tot_pres)_norm": num("pt2_(fan_inlet_tot_pres)_norm"),
        "engtype_LEAP-1A26": et_a26,
        "engtype_LEAP-1A26E1": et_a26e1,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "dayofweek_sin": dow_sin,
        "dayofweek_cos": dow_cos,
        "flight_phase_CRUISE": phase_cruise,
        "flight_phase_TAKEOFF": phase_takeoff,
        "flight_cycle": int(num("flight_cycle")),
    }


def _parse_uploaded_csv(uploaded) -> tuple[list[dict], str | None]:
    """Return (flights_list, eng_number) using only the latest WINDOW_SIZE rows."""
    df = pd.read_csv(uploaded)
    if len(df) < api.WINDOW_SIZE:
        raise ValueError(
            f"CSV has only {len(df)} rows; need at least {api.WINDOW_SIZE} "
            "(the model's input window). Upload more flight history for this engine."
        )
    tail = df.tail(api.WINDOW_SIZE).copy()

    # Detect engine id column for per-engine baseline lookup
    eng_col = "esn" if "esn" in df.columns else ("eng_number" if "eng_number" in df.columns else None)
    eng_number: str | None = None
    if eng_col:
        engines_in_window = {str(v).strip() for v in tail[eng_col].dropna() if str(v).strip()}
        if len(engines_in_window) > 1:
            raise ValueError(
                f"CSV contains data for {len(engines_in_window)} different engines "
                f"({', '.join(sorted(engines_in_window))}). Upload one engine at a time."
            )
        if engines_in_window:
            eng_number = next(iter(engines_in_window))

    flights = [_row_to_flight(r) for r in tail.to_dict(orient="records")]
    return flights, eng_number


# ---------------------------------------------------------------------------
# Inference — same logic as the FastAPI /predict endpoint, called in-process.
# ---------------------------------------------------------------------------
def _run_prediction(
    flights: list[dict],
    model_choice: str,
    eng_number: str | None,
    expected_max_cycle: float | None,
) -> dict[str, Any]:
    req = api.PredictionRequest(
        flights=[api.FlightRow(**f) for f in flights],
        model_choice=model_choice,
        eng_number=eng_number,
        expected_max_cycle=expected_max_cycle,
    )

    targets = list(api.models) if model_choice == "average" else [model_choice]
    bad = [t for t in targets if not api.SCHEMA_MATCH.get(t, False)]
    if bad:
        raise ValueError(
            f"Model artifacts {bad} were trained on an older feature schema "
            f"(shapes={ {t: api.MODEL_SHAPES[t] for t in bad} }); current schema is "
            f"window={api.WINDOW_SIZE}, n_features={api.N_FEATURES}. Retrain via "
            "Lstm_PM_py.ipynb to regenerate the .keras files."
        )

    X, latest_cycle, raw_flights = api._request_to_tensor(req)
    baseline_rcl, baseline_source = api._engine_baseline(
        eng_number, expected_max_cycle, latest_cycle
    )
    degradation = api._compute_degradation(raw_flights)
    cox_hazard = api._compute_cox_hazard(raw_flights)
    if cox_hazard and cox_hazard.get("available"):
        history = api._log_cox_prediction(eng_number, cox_hazard)
        if history:
            cox_hazard["history"] = history

    if model_choice == "average":
        residuals = {name: api._predict_residual(m, X) for name, m in api.models.items()}
        residual = float(np.mean(list(residuals.values())))
    else:
        residuals = {model_choice: api._predict_residual(api.models[model_choice], X)}
        residual = residuals[model_choice]

    rcl = baseline_rcl + residual
    warnings: list[str] = []
    if rcl < 0:
        warnings.append(
            f"Raw prediction {rcl:.1f} was negative; clipped to 0. "
            "Engine has likely already exceeded its expected life."
        )
        rcl = 0.0

    return {
        "RCL_prediction": int(round(rcl)),
        "model_choice": model_choice,
        "baseline_rcl": baseline_rcl,
        "baseline_source": baseline_source,
        "residual": residual,
        "per_model_residuals": residuals,
        "degradation": degradation,
        "cox_ph": cox_hazard,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
SEVERITY_COLORS = {
    "stable":   "#28a745",
    "mild":     "#b8860b",
    "elevated": "#d97706",
    "severe":   "#dc3545",
}
BUCKET_COLORS = {
    "low":      "#28a745",
    "moderate": "#b8860b",
    "elevated": "#d97706",
    "high":     "#dc3545",
}


def _render_health_bar(rcl: int, fleet_max: float) -> None:
    pct = max(0.0, min(100.0, (rcl / fleet_max) * 100.0))
    st.markdown(
        f"""
        <div style="margin-top:4px;">
          <div style="background:#e8e8e8;border-radius:6px;overflow:hidden;height:18px;">
            <div style="width:{pct:.1f}%;height:100%;
                        background:linear-gradient(90deg,#dc3545,#f0ad4e 50%,#28a745);"></div>
          </div>
          <div style="display:flex;justify-content:space-between;
                      font-size:0.78rem;color:#777;margin-top:4px;">
            <span>0 cycles</span><span>/ {int(fleet_max)} cycles</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_result(body: dict[str, Any]) -> None:
    is_warn = bool(body.get("warnings"))
    title = "Predicted RCL (warning)" if is_warn else "Predicted Remaining Cycle Life"
    border = "#f0ad4e" if is_warn else "#007BFF"
    bg = "#fff8e6" if is_warn else "#f0f7ff"
    rcl_color = "#dc3545" if is_warn else "#007BFF"

    st.markdown(
        f"""
        <div style="margin-top:14px;padding:16px;background:{bg};
                    border-left:4px solid {border};border-radius:6px;">
          <h3 style="margin:0 0 6px;font-size:1.1rem;color:#222;">{title}</h3>
          <div style="font-size:2.2rem;font-weight:bold;color:{rcl_color};margin:4px 0 8px;">
            {body['RCL_prediction']} cycles
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Friendly baseline-source label
    source = body.get("baseline_source", "fleet_median_fallback")
    if source.startswith("lookup("):
        source_text = f"Per-engine lookup: {source[7:-1]}"
    elif source == "request_override":
        source_text = "Per-engine (override supplied in request)"
    elif source == "fleet_median_fallback":
        source_text = "Fleet median (engine not in lookup table — calibration may be off)"
    else:
        source_text = source

    per_model = body.get("per_model_residuals") or {}
    per_model_str = ", ".join(f"{k}={float(v):.2f}" for k, v in per_model.items())

    lines = [
        f"**Model:** {body['model_choice']}",
        f"**Baseline RCL:** {body['baseline_rcl']} cycles (= expected_max_cycle − latest flight_cycle)",
        f"**Baseline source:** {source_text}",
        f"**Model residual:** {float(body['residual']):.2f} cycles",
    ]
    if per_model_str:
        lines.append(f"**Per-model residuals:** {per_model_str}")
    for w in body.get("warnings", []):
        lines.append(f"⚠ {w}")
    st.markdown("  \n".join(lines))

    _render_health_bar(body["RCL_prediction"], api.FLEET_MAX_LIFE)


def _render_degradation(d: dict[str, Any] | None) -> None:
    if not d:
        return
    sev = d.get("overall_severity", "stable")
    border = SEVERITY_COLORS.get(sev, "#888")
    bg_map = {"stable": "#e6f6ec", "mild": "#fffbf0", "elevated": "#fff3d6", "severe": "#fde8ea"}
    bg = bg_map.get(sev, "#fffbf0")

    st.markdown(
        f"""
        <div style="margin-top:18px;padding:16px;background:{bg};
                    border-left:4px solid {border};border-radius:6px;">
          <h4 style="margin:0 0 8px;font-size:1rem;color:#222;">
            Sensor-driven degradation analysis
            <span style="display:inline-block;padding:3px 10px;border-radius:12px;
                         background:#fff;color:{border};font-weight:bold;margin-left:8px;
                         font-size:0.85rem;text-transform:uppercase;">{sev}</span>
          </h4>
          <div style="font-size:0.88rem;color:#555;">
            Worst-sensor percentile: <b>{d['max_sensor_percentile']}</b> &nbsp;·&nbsp;
            Composite (mean) percentile: <b>{d['mean_sensor_percentile']}</b> &nbsp;·&nbsp;
            Slopes computed across the uploaded {api.WINDOW_SIZE}-flight window vs flight_cycle.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sensor_pretty = {
        "egt_probe_average": "EGT (rises with degradation)",
        "fuel_flw":          "Fuel flow (rises with degradation)",
        "core_spd":          "Core speed (drops with degradation)",
        "oil_pres":          "Oil pressure (drops with degradation)",
    }
    rows = []
    for sensor, info in d.get("per_sensor", {}).items():
        rows.append({
            "Sensor": sensor_pretty.get(sensor, sensor),
            "Slope per cycle": f"{info['slope_per_cycle']:+.4f}",
            "Fleet percentile": f"P{info['fleet_percentile']}",
            "Severity": info["severity"].upper(),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # Random Forest probabilities (when degradation_rf.pkl is loaded)
    rf = d.get("rf")
    if rf:
        cv = rf.get("cv_accuracy_train")
        cv_text = (
            f"Trained RF, CV accuracy {cv * 100:.1f}% (engine-grouped 5-fold)"
            if cv is not None else "Trained RF"
        )
        agree = "agrees with rule" if rf.get("agrees_with_rule") else "disagrees with rule"
        agree_color = "#28a745" if rf.get("agrees_with_rule") else "#dc3545"
        st.markdown(
            f"**{cv_text}** &nbsp;·&nbsp; "
            f"<span style='color:{agree_color};'>{agree}</span>",
            unsafe_allow_html=True,
        )
        for label in ["stable", "mild", "elevated", "severe"]:
            p = float(rf.get("probabilities", {}).get(label, 0.0)) * 100
            arrow = "  ← predicted" if label == rf.get("severity") else ""
            color = SEVERITY_COLORS.get(label, "#888")
            st.markdown(
                f"""
                <div style="margin:4px 0;">
                  <div style="display:flex;justify-content:space-between;font-size:0.82rem;">
                    <span><b style="color:{color};">{label.upper()}</b>{arrow}</span>
                    <span style="font-family:Consolas,monospace;">{p:.1f}%</span>
                  </div>
                  <div style="background:#eee;border-radius:3px;overflow:hidden;height:9px;">
                    <div style="width:{max(1, p):.1f}%;height:100%;background:{color};"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_cox(c: dict[str, Any] | None) -> None:
    if not c or not c.get("available"):
        return
    bucket = c.get("risk_bucket", "low")
    border = BUCKET_COLORS.get(bucket, "#888")
    pct = max(0.0, min(100.0, float(c.get("fleet_percentile") or 0)))
    cidx = c.get("model_c_index")
    cidx_str = f"{cidx:.3f}" if isinstance(cidx, (int, float)) else "—"
    lp = float(c.get("linear_predictor", 0.0))
    lp_sign = "+" if lp >= 0 else ""

    st.markdown(
        f"""
        <div style="margin-top:18px;padding:16px;background:#fffbf0;
                    border-left:4px solid {border};border-radius:6px;">
          <h4 style="margin:0 0 8px;font-size:1rem;color:#222;">
            Cox-PH survival hazard
            <span style="display:inline-block;padding:3px 10px;border-radius:12px;
                         background:#fff;color:{border};font-weight:bold;margin-left:8px;
                         font-size:0.85rem;text-transform:uppercase;">{bucket}</span>
          </h4>
          <div style="font-size:0.88rem;color:#555;">
            Hazard score <b>{c.get('hazard_score')}</b> &nbsp;·&nbsp;
            Linear predictor <b>{lp_sign}{lp:.4f}</b> &nbsp;·&nbsp;
            Fleet percentile <b>P{c.get('fleet_percentile')}</b> &nbsp;·&nbsp;
            Model C-index {cidx_str}
          </div>

          <div style="font-size:0.8rem;color:#444;margin-top:12px;margin-bottom:6px;">
            Fleet hazard percentile &nbsp;·&nbsp;
            <span style="color:#888;font-style:italic;">healthier engines on the left, more at-risk on the right</span>
          </div>
          <div style="height:14px;border-radius:4px;
                      background:linear-gradient(90deg,#28a745 0%,#28a745 25%,#b8860b 50%,#d97706 75%,#dc3545 100%);
                      border:1px solid #ccc;"></div>
          <div style="position:relative;height:24px;margin-top:2px;">
            <div style="position:absolute;left:calc({pct}% - 7px);top:0;
                        width:0;height:0;border-left:7px solid transparent;
                        border-right:7px solid transparent;
                        border-bottom:9px solid #1f3a68;"></div>
            <div style="position:absolute;left:calc({pct}% - 22px);top:10px;
                        font-size:0.72rem;font-weight:bold;color:#1f3a68;
                        white-space:nowrap;">P{pct:.0f}</div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#888;margin-top:2px;">
            <span>0 — lowest risk</span><span>25</span>
            <span>50 — fleet median</span><span>75</span>
            <span>100 — highest risk</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = c.get("history") or []
    if len(history) > 1:
        with st.expander(f"Trend across {len(history)} predictions for this engine"):
            for i, h in enumerate(history):
                arrow = "🟢 current" if i == len(history) - 1 else ""
                lp_h = float(h.get("linear_predictor", 0.0))
                st.markdown(
                    f"`#{i+1}` &nbsp;LP {'+' if lp_h >= 0 else ''}{lp_h:.3f} "
                    f"&nbsp;·&nbsp; P{h.get('fleet_percentile')} {arrow}",
                    unsafe_allow_html=True,
                )

    if c.get("note"):
        st.caption(c["note"])


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("✈ Aircraft Engine RCL Forecast")
st.caption("LEAP-1A · LSTM / GRU / CNN ensemble · 49-engine combined fleet")

with st.container(border=True):
    schema_ok = all(api.SCHEMA_MATCH.values())
    cols = st.columns([3, 1])
    cols[0].markdown(
        f"**Models loaded:** {', '.join(api.models)} &nbsp;·&nbsp; "
        f"window={api.WINDOW_SIZE}, features={api.N_FEATURES}"
    )
    cols[1].markdown(
        f"<span style='color:{'#28a745' if schema_ok else '#dc3545'};font-weight:bold;'>"
        f"{'healthy' if schema_ok else 'schema mismatch'}</span>",
        unsafe_allow_html=True,
    )
    if not schema_ok:
        st.warning(
            "Loaded model artifacts predate the current feature schema. "
            "Retrain via Lstm_PM_py.ipynb to regenerate them."
        )

st.subheader(f"Input: last {api.WINDOW_SIZE} flights of one engine")

uploaded = st.file_uploader(
    "Upload CSV (one row per flight, oldest first)",
    type=["csv"],
    help=(
        "Required columns: flight_cycle, egt_probe_average, fuel_flw, core_spd, "
        "oil_temp, oil_pres, pt2_(fan_inlet_tot_pres) and their _norm variants, "
        "plus eng_type and either flight_phase or its one-hots, and a "
        "flight_datetime_c column (used to derive month/day-of-week sin/cos). "
        "Rows from Data/LEAB_engines_data_cleaned.csv already match this layout."
    ),
)

model_choice = st.selectbox(
    "Model",
    options=["average", "lstm", "gru", "cnn"],
    format_func=lambda v: {
        "average": "Ensemble average (LSTM + GRU + CNN)",
        "lstm": "LSTM only",
        "gru": "GRU only",
        "cnn": "CNN only",
    }[v],
)

predict_btn = st.button(
    "Predict RCL",
    type="primary",
    width="stretch",
    disabled=(uploaded is None) or (not schema_ok),
)

if predict_btn and uploaded is not None:
    try:
        flights, eng_number = _parse_uploaded_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")
        st.stop()

    if eng_number:
        st.info(f"Engine detected from CSV: **{eng_number}**")

    with st.spinner("Running inference…"):
        try:
            body = _run_prediction(flights, model_choice, eng_number, None)
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.stop()

    _render_result(body)
    _render_degradation(body.get("degradation"))
    _render_cox(body.get("cox_ph"))

    with st.expander("Show raw request / response"):
        st.json({
            "request": {
                "model_choice": model_choice,
                "eng_number": eng_number,
                "flights": f"[{len(flights)} rows omitted for brevity]",
            },
            "response": body,
        })

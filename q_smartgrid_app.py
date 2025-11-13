
# Q‑SmartGrid: Generative‑AI & Quantum Energy Optimizer for Sustainable Campuses
# -------------------------------------------------------------
# This Streamlit demo is offline‑friendly. It simulates:
# 1) Forecasting (baseline + simple moving average)
# 2) "Quantum" optimization (mock QAOA with optional Cirq sampling)
# 3) Generative AI recommendations (rule‑based text generator with placeholders to plug Gemini/Vertex later)
#
# How to run:
#   1) pip install streamlit numpy pandas matplotlib
#      (optional) pip install cirq
#   2) streamlit run q_smartgrid_app.py
#
# Optional integrations (placeholders marked TODO):
#   - Vertex AI for forecasting
#   - Cirq QAOA for optimization
#   - Gemini for natural‑language recommendations
#
# Author: Team NextGen — Q‑SmartGrid
# -------------------------------------------------------------

import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

# Try to import Cirq (optional). If not available, we fallback to a heuristic optimizer.
try:
    import cirq  # type: ignore
    CIRQ_AVAILABLE = True
except Exception:
    CIRQ_AVAILABLE = False

# -------------------------
# UI CONFIG
# -------------------------
st.set_page_config(
    page_title="Q‑SmartGrid — Quantum + Generative AI for Energy Optimization",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for nicer look
st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.big-number {font-size: 36px; font-weight: 700; margin: 0;}
.kpi-box {padding: 1rem; border-radius: 16px; box-shadow: 0 6px 20px rgba(0,0,0,0.05);}
h1, h2, h3 { font-weight: 700; }
.caption-ar { color: #666; font-size: 0.9rem; }
.section { border: 1px solid #eee; border-radius: 16px; padding: 1rem 1.25rem; background: #fff;}
hr { border-top: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# SIDEBAR CONTROLS
# -------------------------
st.sidebar.title("⚙️ Controls / الإعدادات")
days = st.sidebar.slider("Days to simulate (الأيام)", 3, 14, 7)
seed = st.sidebar.number_input("Random seed (اختياري)", value=42, step=1)
peak_reduction_goal = st.sidebar.slider("Target peak reduction % (هدف خفض الذروة)", 0, 30, 12, step=1)
co2_factor = st.sidebar.number_input("CO₂ factor (kg/kWh)", value=0.45, step=0.01, format="%.2f")
price_per_kwh = st.sidebar.number_input("Price per kWh ($)", value=0.12, step=0.01, format="%.2f")

st.sidebar.markdown("---")
st.sidebar.subheader("Loads / الأحمال")
num_buildings = st.sidebar.slider("Number of buildings (عدد المباني)", 3, 12, 6)
min_load = st.sidebar.number_input("Min base load per building (kW)", value=50, step=5)
max_load = st.sidebar.number_input("Max base load per building (kW)", value=250, step=5)

st.sidebar.markdown("---")
st.sidebar.subheader("Optional Integrations (اختياري)")
use_gemini = st.sidebar.checkbox("Use Gemini (placeholder only)")
use_vertex = st.sidebar.checkbox("Use Vertex AI (placeholder only)")
use_cirq = st.sidebar.checkbox("Use Cirq sampling (if installed)", value=CIRQ_AVAILABLE, disabled=not CIRQ_AVAILABLE)

random.seed(seed); np.random.seed(seed)

# -------------------------
# DATA SIMULATION
# -------------------------
def simulate_baseline(days:int, num_buildings:int, min_load:int, max_load:int):
    """
    Simulate per-hour campus load (kW) over N days across multiple buildings.
    Returns (df_long, df_total) where:
      - df_long has columns [timestamp, building, load_kw]
      - df_total has columns [timestamp, total_kw]
    """
    start = datetime.now().replace(minute=0, second=0, microsecond=0)
    ts = [start + timedelta(hours=h) for h in range(days*24)]

    records = []
    for b in range(num_buildings):
        base = np.random.uniform(min_load, max_load)  # base draw per building
        # Daily seasonality + random noise
        daily_pattern = 1.0 + 0.35*np.sin(np.linspace(0, 2*math.pi*days, days*24) + np.random.rand()*2*math.pi)
        noise = np.random.normal(0, 0.07, size=days*24)
        loads = np.clip(base * daily_pattern * (1+noise), a_min=10, a_max=None)
        # add working-hours bump (8:00-18:00)
        for h, t in enumerate(ts):
            if 8 <= t.hour < 18:
                loads[h] *= np.random.uniform(1.05, 1.25)
        for h, t in enumerate(ts):
            records.append({"timestamp": ts[h], "building": f"B{b+1}", "load_kw": float(loads[h])})

    df_long = pd.DataFrame(records)
    df_total = df_long.groupby("timestamp", as_index=False)["load_kw"].sum()
    df_total.rename(columns={"load_kw": "total_kw"}, inplace=True)
    return df_long, df_total

def moving_average(series: pd.Series, window:int=6):
    return series.rolling(window=window, min_periods=1).mean()

df_long, df_total = simulate_baseline(days, num_buildings, min_load, max_load)
df_total["forecast_kw"] = moving_average(df_total["total_kw"], window=6)

# -------------------------
# "QUANTUM" OPTIMIZATION (MOCK QAOA + OPTIONAL CIRQ)
# -------------------------
def cost_peak(bitstring: np.ndarray, forecast: np.ndarray, penalty=0.0):
    """
    Simple cost: max power (peak) after shedding chosen loads.
    bitstring[i]=1 means shed that proportion from the i-th shed slot.
    Here we'll map bits to % reduct increments to approximate a control policy.
    """
    # map bitstring to a reduction schedule between 0..goal
    if len(bitstring) == 0:
        return np.max(forecast)
    max_reduct = peak_reduction_goal / 100.0
    step = max_reduct / max(1, len(bitstring))
    reduct = np.sum(bitstring) * step  # aggregate reduction fraction
    reduct = min(max_reduct, reduct)
    optimized = forecast * (1.0 - reduct)
    return np.max(optimized) + penalty*np.sum(bitstring)  # tiny penalty for turning off many switches

def greedy_optimizer(n_bits:int, forecast: np.ndarray):
    # Try toggling bits greedily to reduce peak
    best = np.zeros(n_bits, dtype=int)
    best_cost = cost_peak(best, forecast)
    improved = True
    while improved:
        improved = False
        for i in range(n_bits):
            cand = best.copy()
            cand[i] = 1 - cand[i]
            c = cost_peak(cand, forecast)
            if c < best_cost:
                best, best_cost = cand, c
                improved = True
    return best, best_cost

def cirq_sampler(n_bits:int, trials:int=128):
    """Sample random bitstrings from a shallow circuit (proxy for QAOA sampling)."""
    if not CIRQ_AVAILABLE:
        return None
    qubits = [cirq.LineQubit(i) for i in range(n_bits)]
    circuit = cirq.Circuit()
    # simple layer of Hadamards + CZ ring + (optional) another layer
    circuit.append(cirq.H.on_each(*qubits))
    for i in range(n_bits):
        circuit.append(cirq.CZ(qubits[i], qubits[(i+1)%n_bits]))
    circuit.append(cirq.measure(*qubits, key="m"))
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions=trials)
    m = result.measurements["m"]
    return m  # shape (trials, n_bits)

def run_quantum_optimization(forecast: np.ndarray, n_bits:int=6):
    """
    Try Cirq sampling first (if available), score samples by cost, keep best.
    Fallback to a greedy optimizer when Cirq is unavailable.
    """
    best_bits = None
    best_cost = float("inf")

    if CIRQ_AVAILABLE and use_cirq:
        samples = cirq_sampler(n_bits=n_bits, trials=256)
        if samples is not None and len(samples) > 0:
            for s in samples:
                c = cost_peak(s, forecast)
                if c < best_cost:
                    best_cost, best_bits = c, s

    if best_bits is None:
        # fallback greedy search
        best_bits, best_cost = greedy_optimizer(n_bits, forecast)

    # Compute reduction implied by bits
    max_reduct = peak_reduction_goal / 100.0
    step = max_reduct / max(1, len(best_bits))
    reduct = min(max_reduct, np.sum(best_bits) * step)
    optimized = forecast * (1.0 - reduct)
    return best_bits, reduct, optimized, best_cost

# Apply optimization on forecasted horizon (last 24h window for clarity)
window = min(len(df_total), 24)
recent = df_total.tail(window).copy()
bits, reduct_frac, optimized_curve, optimized_peak = run_quantum_optimization(
    forecast=recent["forecast_kw"].values, n_bits=6
)
recent["optimized_kw"] = optimized_curve

baseline_peak = float(np.max(recent["forecast_kw"].values))
savings_frac = 1.0 - float(np.sum(recent["optimized_kw"])) / float(np.sum(recent["forecast_kw"]))
peak_drop_pct = (baseline_peak - optimized_peak) / baseline_peak * 100.0
energy_saved_kwh = float(np.sum(recent["forecast_kw"] - recent["optimized_kw"]))  # per hour sum ~ kWh
co2_avoided = energy_saved_kwh * co2_factor
cost_saved = energy_saved_kwh * price_per_kwh

# -------------------------
# GENERATIVE RECOMMENDATIONS (RULE‑BASED + PLACEHOLDERS)
# -------------------------
def gen_recommendations(baseline_peak, peak_drop_pct, energy_saved_kwh, co2_avoided):
    recs = []
    if peak_drop_pct >= 10:
        recs.append("Shift non‑critical HVAC setpoints by +1.0°C during 12:00–17:00 (expected peak hours).")
    else:
        recs.append("Start with a mild HVAC setback of +0.5°C during 12:00–17:00 to trim the peak.")

    if energy_saved_kwh > 200:
        recs.append("Schedule batch‑loads (laundry labs, pumps) after 20:00 to flatten the peak curve.")
    else:
        recs.append("Defer small batch‑loads (e.g., printing farms) to evening hours to reduce mid‑day spikes.")

    if co2_avoided > 50:
        recs.append("Publicize the CO₂ reduction dashboard to drive behavioral change across campus.")
    else:
        recs.append("Pilot recommendations on 1–2 buildings, then scale once impact is verified.")

    # Arabic mirrors (concise)
    recs_ar = [
        "اضبطي نقاط ضبط التكييف +0.5–1°C وقت الذروة (12–17).",
        "جدوِلِي الأحمال الدورية لِما بعد الساعة 20:00 لتسطيح الذروة.",
        "اعملي لوحة مؤشرات للحد من الانبعاثات لرفع الوعي."
    ]
    return recs, recs_ar

recs_en, recs_ar = gen_recommendations(baseline_peak, peak_drop_pct, energy_saved_kwh, co2_avoided)

# TODO: If you have Gemini API, replace the above with a real call:
# from google import genai
# client = genai.Client(api_key="YOUR_KEY")
# prompt = f"Suggest campus energy‑savings based on: peak_drop={peak_drop_pct:.1f}%, kWh_saved={energy_saved_kwh:.1f}."
# recs_en = client.models.generate_content(model="gemini-2.0-pro", contents=prompt).text

# -------------------------
# HEADER
# -------------------------
st.title("⚡ Q‑SmartGrid — Generative‑AI & Quantum Energy Optimizer")
st.caption("جامعة / Campus demo • AI + Quantum • Nearly‑Completed Project (جاهز للعرض)")
st.markdown("---")

# -------------------------
# KPI ROW
# -------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="kpi-box"><div class="big-number">%.1f kW</div>Baseline Peak<br><span class="caption-ar">ذروة الاستهلاك</span></div>' % baseline_peak, unsafe_allow_html=True)
with col2:
    st.markdown('<div class="kpi-box"><div class="big-number">%.1f%%</div>Peak Reduction<br><span class="caption-ar">خفض الذروة</span></div>' % peak_drop_pct, unsafe_allow_html=True)
with col3:
    st.markdown('<div class="kpi-box"><div class="big-number">%.1f kWh</div>Energy Saved<br><span class="caption-ar">الطاقة الموفَّرة</span></div>' % energy_saved_kwh, unsafe_allow_html=True)
with col4:
    st.markdown('<div class="kpi-box"><div class="big-number">$%.2f</div>Est. Cost Saved<br><span class="caption-ar">توفير تقديري</span></div>' % cost_saved, unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# CHARTS
# -------------------------
st.subheader("📈 Forecast vs Optimized • التنبؤ مقابل التحسين")
st.caption("Last 24 hours window • نافذة آخر 24 ساعة")

chart_df = recent[["timestamp", "forecast_kw", "optimized_kw"]].set_index("timestamp")
st.line_chart(chart_df)

with st.expander("Advanced details • تفاصيل متقدمة"):
    st.write("Selected bits (proxy policy):", bits.tolist())
    st.write(f"Reduction fraction used: {reduct_frac*100:.1f}% of forecast")
    st.write("Use Cirq sampling:", CIRQ_AVAILABLE and use_cirq)

st.markdown("---")

# -------------------------
# LOAD BREAKDOWN (SIMULATED)
# -------------------------
st.subheader("🏢 Building Loads • أحمال المباني (محاكاة)")
# Show average load per building (last 24h)
last_window = df_long[df_long["timestamp"].isin(recent["timestamp"])]
avg_by_building = last_window.groupby("building")["load_kw"].mean().sort_values(ascending=False)
st.bar_chart(avg_by_building)

st.markdown("---")

# -------------------------
# RECOMMENDATIONS
# -------------------------
st.subheader("🧠 AI Recommendations • توصيات الذكاء الاصطناعي")
tab1, tab2 = st.tabs(["English", "العربية"])
with tab1:
    for i, r in enumerate(recs_en, 1):
        st.write(f"{i}. {r}")
with tab2:
    for i, r in enumerate(recs_ar, 1):
        st.write(f"{i}. {r}")

st.markdown("---")

# -------------------------
# IMPACT SUMMARY
# -------------------------
st.subheader("🌱 Impact • الأثر")
colA, colB = st.columns(2)
with colA:
    st.write(f"**Energy saved (kWh):** {energy_saved_kwh:.1f}")
    st.write(f"**CO₂ avoided (kg):** {co2_avoided:.1f}")
with colB:
    st.write(f"**Peak reduction:** {peak_drop_pct:.1f}%")
    st.write(f"**Estimated cost saved ($):** {cost_saved:.2f}")

st.info("Tip: Adjust targets from the sidebar to simulate scenarios • جرّبي تغيير الإعدادات من الشريط الجانبي لمحاكاة سيناريوهات مختلفة.")

# -------------------------
# FOOTER / INTEGRATION NOTES
# -------------------------
st.markdown("---")
st.markdown("""
### 🔌 Integration Notes (for real data)
- **Vertex AI (Forecasting)**: Replace `moving_average` with calls to your deployed model; pass the resulting forecast to the optimizer.
- **Cirq (QAOA)**: Swap the sampler with your QUBO‑based circuit; score bitstrings via `cost_peak` or a refined cost.
- **Gemini (Generative)**: Replace rule‑based `gen_recommendations` with a real prompt to Gemini (`gemini-2.0-pro`).
- **Data Sources**: Plug building‑level meters or SCADA/BMS feeds; keep the same dataframe schema to reuse the UI.
""")

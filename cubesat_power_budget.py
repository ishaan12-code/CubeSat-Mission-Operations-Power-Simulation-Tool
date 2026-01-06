import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

G_SUN = 1361.0  # W/m^2
R_E = 6371.0    # km


# -------------------------
# App
# -------------------------
st.set_page_config(page_title="CubeSat Power Budget", layout="wide")

st.markdown(
    """
<style>
html, body { background-color:#f7f7f7 !important; color:#111 !important; font-family: "Segoe UI", system-ui, sans-serif; }
#MainMenu {visibility:hidden;} footer {visibility:hidden;} header {visibility:hidden;}
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1300px; }
section[data-testid="stSidebar"] { background:#f1f1f1 !important; border-right:1px solid #ddd; }
div[data-testid="stMetric"], div[data-testid="stDataFrame"], div[data-testid="stTable"], div.stMarkdown, div[data-testid="stAlert"] {
  background:#fff !important; border:1px solid #e2e2e2 !important; border-radius:12px !important; padding:14px 14px !important;
}
.stButton button { background:#2563eb !important; color:#fff !important; border-radius:10px !important; border:none !important; padding:.55rem .95rem !important; }
hr { border:none; height:1px; background:#e5e7eb; margin: 1.0rem 0; }
.small-note { color:#555; font-size:0.92rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🛰️ CubeSat Power + Mission Budget")
st.caption("Power, SOC, modes, downlink, data queue, uncertainty + 3S voltage + brownout checks.")


# -------------------------
# Helpers
# -------------------------
def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def orbital_period_minutes(alt_km: float) -> float:
    mu = 3.986004418e14  # m^3/s^2
    r_m = (R_E + max(0.0, alt_km)) * 1000.0
    T = 2.0 * math.pi * math.sqrt((r_m**3) / mu)
    return float(T / 60.0)


def eclipse_fraction_from_geometry(alt_km: float, beta_deg: float) -> float:
    r = R_E + max(0.0, alt_km)
    if r <= R_E:
        return 0.5

    beta = np.deg2rad(abs(beta_deg))
    arg = float(np.clip(R_E / r, 0.0, 1.0))

    beta_crit = float(np.arccos(arg))
    if beta >= beta_crit:
        return 0.0

    denom = max(1e-9, math.cos(beta))
    inner = math.sqrt(max(0.0, 1.0 - arg * arg)) / denom
    inner = float(np.clip(inner, 0.0, 1.0))

    theta = float(np.arccos(inner))
    frac = (2.0 * theta) / (2.0 * math.pi)
    return float(np.clip(frac, 0.0, 0.6))


def sanitize_mode_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Subsystem", "Power_W", "Duty_%"])
    out = df.copy()
    for col in ["Subsystem", "Power_W", "Duty_%"]:
        if col not in out.columns:
            out[col] = "" if col == "Subsystem" else 0.0
    out["Power_W"] = pd.to_numeric(out["Power_W"], errors="coerce").fillna(0.0).clip(0, 1e6)
    out["Duty_%"] = pd.to_numeric(out["Duty_%"], errors="coerce").fillna(0.0).clip(0, 100.0)
    out["Subsystem"] = out["Subsystem"].astype(str)
    return out


def avg_power(df: pd.DataFrame) -> float:
    df = sanitize_mode_df(df)
    if df.empty:
        return 0.0
    return float(np.sum(df["Power_W"] * (df["Duty_%"] / 100.0)))


def array_power_w(A_m2, cell_eff, pack_factor, mppt_eff, inc_deg, beta_derate, degradation, temp_factor):
    cos_inc = max(0.0, float(np.cos(np.deg2rad(inc_deg))))
    return float(G_SUN * A_m2 * cell_eff * pack_factor * mppt_eff * cos_inc * beta_derate * degradation * temp_factor)


def charge_taper(soc: float, start: float, floor: float) -> float:
    if soc <= start:
        return 1.0
    x = (soc - start) / max(1e-9, (1.0 - start))
    return float(max(floor, 1.0 - x * (1.0 - floor)))


def ocv_per_cell_from_soc(soc: float, v_floor: float, v_ceil: float) -> float:
    soc = clip01(soc)
    xs = np.array([0.00, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 1.00])
    ys = np.array([3.00, 3.15, 3.30, 3.45, 3.70, 3.90, 4.03, 4.18])
    v = float(np.interp(soc, xs, ys))
    return float(np.clip(v, v_floor, v_ceil))


def pack_voltage_under_load_3s(soc: float, p_load_w: float, r_pack_ohm: float, v_min_cell: float, v_max_cell: float):
    v_cell = ocv_per_cell_from_soc(soc, v_min_cell, v_max_cell)
    v_ocv = max(0.1, 3.0 * v_cell)
    i = float(p_load_w / v_ocv) if p_load_w > 0 else 0.0
    v_load = float(v_ocv - i * max(0.0, r_pack_ohm))
    return v_ocv, v_load, i


def add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    T_min = float(df["T_min"].iloc[0]) if len(df) else 0.0
    df["T_plus_min"] = df["Orbit"] * T_min
    df["T_plus_hr"] = df["T_plus_min"] / 60.0
    df["Mission_day"] = (df["T_plus_hr"] // 24).astype(int)
    df["HHMM"] = pd.to_datetime(df["T_plus_min"], unit="m", origin="unix").dt.strftime("%H:%M")
    return df


def marker_scatter(ax, x, y, mask, label):
    idx = np.where(mask)[0]
    if idx.size:
        ax.scatter(x[idx], y[idx], s=25, label=label)


# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Orbit / Geometry")
alt_km = st.sidebar.number_input("Altitude (km)", 300.0, 900.0, 550.0, 10.0)
beta_deg = st.sidebar.slider("Beta angle (deg)", 0.0, 90.0, 25.0, 1.0)

use_manual_eclipse = st.sidebar.checkbox("Manual eclipse fraction", value=False)
eclipse_override = None
if use_manual_eclipse:
    eclipse_override = st.sidebar.slider("Eclipse fraction (%)", 0.0, 60.0, 35.0, 0.5) / 100.0

st.sidebar.header("Solar Array")
A_m2 = st.sidebar.number_input("Effective illuminated area (m²)", 0.001, 1.0, 0.032, 0.001)
cell_eff = st.sidebar.slider("Cell efficiency (%)", 5.0, 40.0, 28.0, 0.5) / 100.0
pack_factor = st.sidebar.slider("Packing factor (%)", 30.0, 100.0, 85.0, 1.0) / 100.0
mppt_eff = st.sidebar.slider("MPPT efficiency (%)", 50.0, 99.0, 92.0, 0.5) / 100.0
base_inc_deg = st.sidebar.slider("Base incidence angle (deg)", 0.0, 85.0, 25.0, 1.0)
inc_jitter_deg = st.sidebar.slider("Pointing jitter σ (deg)", 0.0, 25.0, 4.0, 0.5)
beta_derate = st.sidebar.slider("Beta derating factor", 0.2, 1.0, 0.85, 0.01)
degradation = st.sidebar.slider("Degradation factor", 0.5, 1.0, 0.95, 0.01)

st.sidebar.header("Thermal")
panel_temp_c = st.sidebar.slider("Panel temperature (°C)", -20.0, 90.0, 30.0, 1.0)
temp_coeff = st.sidebar.slider("Cell temp coeff (%/°C)", 0.1, 0.8, 0.4, 0.05) / 100.0

st.sidebar.header("Battery")
E_bat_Wh = st.sidebar.number_input("Battery energy (Wh)", 1.0, 300.0, 76.96, 0.5)
DoD = st.sidebar.slider("Usable DoD (%)", 10.0, 95.0, 80.0, 1.0) / 100.0
cap_fade = st.sidebar.slider("Capacity fade (%)", 0.0, 40.0, 0.0, 0.5) / 100.0
eta_c = st.sidebar.slider("Charge efficiency", 0.7, 1.0, 0.95, 0.01)
eta_d = st.sidebar.slider("Discharge efficiency", 0.7, 1.0, 0.95, 0.01)
SOC0 = st.sidebar.slider("Initial SOC (%)", 0.0, 100.0, 80.0, 1.0) / 100.0

st.sidebar.subheader("SOC Protection")
soc_warn = st.sidebar.slider("Warn below SOC (%)", 0.0, 60.0, 20.0, 1.0) / 100.0
soc_hard = st.sidebar.slider("Hard safe below SOC (%)", 0.0, 40.0, 10.0, 1.0) / 100.0
force_safe_below = st.sidebar.slider("Force SAFE below SOC (%)", 0.0, 60.0, 18.0, 1.0) / 100.0

st.sidebar.subheader("Charge Taper")
taper_start = st.sidebar.slider("Start taper at SOC (%)", 50.0, 95.0, 85.0, 1.0) / 100.0
taper_floor = st.sidebar.slider("Min taper factor", 0.1, 1.0, 0.25, 0.05)

st.sidebar.header("Voltage Model (3S)")
r_pack_mohm = st.sidebar.number_input("Pack internal R (mΩ)", 0.0, 800.0, 180.0, 10.0)
r_pack_ohm = float(r_pack_mohm) / 1000.0
v_cell_min = st.sidebar.slider("OCV floor per-cell (V)", 2.7, 3.3, 3.0, 0.01)
v_cell_max = st.sidebar.slider("OCV ceiling per-cell (V)", 3.9, 4.25, 4.18, 0.01)
brownout_v = st.sidebar.number_input("Brownout threshold (pack V)", 7.0, 12.3, 9.6, 0.1)
brownout_margin_v = st.sidebar.number_input("Warn margin (V)", 0.0, 2.0, 0.4, 0.1)

st.sidebar.header("Data & Downlink")
data_gen_MB = st.sidebar.number_input("Data per imaging orbit (MB)", 0.0, 500.0, 25.0, 1.0)
data_queue_max = st.sidebar.number_input("Max data queue (MB)", 10.0, 20000.0, 2000.0, 50.0)
bitrate_kbps = st.sidebar.number_input("Downlink bitrate (kbps)", 0.0, 50000.0, 9600.0, 100.0)
pass_minutes = st.sidebar.number_input("Avg pass duration (min)", 0.0, 30.0, 8.0, 0.5)
pass_jitter = st.sidebar.number_input("Pass jitter σ (min)", 0.0, 10.0, 1.5, 0.1)
downlink_sun_only = st.sidebar.checkbox("Downlink only in sunlight", value=True)

st.sidebar.header("Ops Schedule")
imaging_every = st.sidebar.number_input("IMAGING every N orbits (0=off)", 0, 100, 3, 1)
downlink_every = st.sidebar.number_input("DOWNLINK every N orbits (0=off)", 0, 100, 5, 1)

st.sidebar.header("Simulation")
N_orbits = st.sidebar.number_input("Orbits to simulate", 1, 1000, 80, 1)

st.sidebar.header("Monte Carlo")
mc_runs = st.sidebar.number_input("Runs (0=off)", 0, 2000, 250, 50)
unc_pct = st.sidebar.slider("Uncertainty (±%)", 0.0, 30.0, 8.0, 0.5) / 100.0


# -------------------------
# Modes editor
# -------------------------
st.subheader("Modes")
st.markdown('<div class="small-note">Per-mode tables. Duty is average-on percentage in that state.</div>', unsafe_allow_html=True)

mode_tabs = st.tabs(["SAFE", "NOMINAL", "IMAGING", "DOWNLINK"])

def mode_defaults(name: str) -> pd.DataFrame:
    if name == "SAFE":
        return pd.DataFrame([
            {"Subsystem": "OBC", "Power_W": 1.2, "Duty_%": 100.0},
            {"Subsystem": "EPS overhead", "Power_W": 0.5, "Duty_%": 100.0},
            {"Subsystem": "Comms RX", "Power_W": 0.8, "Duty_%": 50.0},
        ])
    if name == "IMAGING":
        return pd.DataFrame([
            {"Subsystem": "OBC", "Power_W": 2.0, "Duty_%": 100.0},
            {"Subsystem": "EPS overhead", "Power_W": 0.5, "Duty_%": 100.0},
            {"Subsystem": "Payload (camera/ML)", "Power_W": 4.0, "Duty_%": 80.0},
            {"Subsystem": "ADCS (avg)", "Power_W": 2.5, "Duty_%": 60.0},
        ])
    if name == "DOWNLINK":
        return pd.DataFrame([
            {"Subsystem": "OBC", "Power_W": 2.0, "Duty_%": 100.0},
            {"Subsystem": "EPS overhead", "Power_W": 0.5, "Duty_%": 100.0},
            {"Subsystem": "Comms TX", "Power_W": 7.0, "Duty_%": 70.0},
            {"Subsystem": "ADCS (avg)", "Power_W": 2.0, "Duty_%": 40.0},
        ])
    return pd.DataFrame([
        {"Subsystem": "OBC", "Power_W": 2.0, "Duty_%": 100.0},
        {"Subsystem": "EPS overhead", "Power_W": 0.5, "Duty_%": 100.0},
        {"Subsystem": "ADCS (avg)", "Power_W": 2.0, "Duty_%": 30.0},
        {"Subsystem": "Comms RX", "Power_W": 0.8, "Duty_%": 30.0},
    ])

modes = {}
for name, tab in zip(["SAFE", "NOMINAL", "IMAGING", "DOWNLINK"], mode_tabs):
    with tab:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Sunlight state**")
            sun_df = st.data_editor(mode_defaults(name), num_rows="dynamic", use_container_width=True, key=f"{name}_sun")
        with c2:
            st.markdown("**Eclipse state**")
            ecl_df = st.data_editor(mode_defaults("SAFE"), num_rows="dynamic", use_container_width=True, key=f"{name}_ecl")
        modes[name] = {"sun": sun_df, "ecl": ecl_df}

st.markdown("---")


# -------------------------
# Simulation
# -------------------------
def run_sim(n_orbits: int, rng: np.random.Generator) -> tuple[pd.DataFrame, float, float]:
    T_min = orbital_period_minutes(alt_km)
    f_e = eclipse_fraction_from_geometry(alt_km, beta_deg) if eclipse_override is None else float(eclipse_override)
    t_ecl_h = (T_min * f_e) / 60.0
    t_sun_h = (T_min * (1.0 - f_e)) / 60.0

    temp_factor = max(0.0, 1.0 - temp_coeff * (panel_temp_c - 25.0))
    E_usable = max(1e-6, E_bat_Wh * (1.0 - cap_fade) * DoD)

    soc = clip01(SOC0)
    backlog = 0.0
    rows = []

    for orbit in range(1, int(n_orbits) + 1):
        imaging_due = (imaging_every > 0) and (orbit % imaging_every == 0)
        downlink_due = (downlink_every > 0) and (orbit % downlink_every == 0)

        if soc < force_safe_below:
            mode = "SAFE"
        elif downlink_due:
            mode = "DOWNLINK"
        elif imaging_due:
            mode = "IMAGING"
        else:
            mode = "NOMINAL"

        if soc < soc_hard:
            mode = "SAFE"

        inc_deg = float(np.clip(base_inc_deg + rng.normal(0.0, inc_jitter_deg), 0.0, 85.0))

        P_sa = array_power_w(A_m2, cell_eff, pack_factor, mppt_eff, inc_deg, beta_derate, degradation, temp_factor)
        E_gen = P_sa * t_sun_h

        P_sun = avg_power(modes[mode]["sun"])
        P_ecl = avg_power(modes[mode]["ecl"])
        E_use_sun = P_sun * t_sun_h
        E_use_ecl = P_ecl * t_ecl_h
        E_use = E_use_sun + E_use_ecl

        data_gen = data_gen_MB if mode == "IMAGING" else 0.0
        backlog = min(data_queue_max, backlog + data_gen)

        downlinked = 0.0
        pass_len = max(0.0, float(pass_minutes + rng.normal(0.0, pass_jitter)))
        if mode == "DOWNLINK" and bitrate_kbps > 0 and pass_len > 0:
            if (not downlink_sun_only) or (t_sun_h * 60.0 > 0.0):
                cap_MB = (bitrate_kbps * 60.0 * pass_len) / (8.0 * 1024.0)
                downlinked = min(backlog, cap_MB)
                backlog -= downlinked

        v_ocv, v_load_ecl, i_ecl = pack_voltage_under_load_3s(
            soc=soc,
            p_load_w=P_ecl,
            r_pack_ohm=r_pack_ohm,
            v_min_cell=v_cell_min,
            v_max_cell=v_cell_max,
        )
        brownout = v_load_ecl <= brownout_v
        brownout_warn = v_load_ecl <= (brownout_v + brownout_margin_v)

        net_sun = E_gen - E_use_sun
        if net_sun >= 0:
            taper = charge_taper(soc, taper_start, taper_floor)
            soc += (net_sun * eta_c * taper) / E_usable
        else:
            soc += (net_sun / max(1e-9, eta_d)) / E_usable

        soc -= (E_use_ecl / max(1e-9, eta_d)) / E_usable
        soc = clip01(soc)

        low_warn = soc < soc_warn
        low_hard = soc < soc_hard
        data_full = backlog >= data_queue_max - 1e-6

        rows.append({
            "Orbit": orbit,
            "Mode": mode,
            "T_min": T_min,
            "Eclipse_frac": f_e,
            "Sun_min": t_sun_h * 60.0,
            "Ecl_min": t_ecl_h * 60.0,
            "Inc_deg": inc_deg,
            "P_SA_W": P_sa,
            "E_gen_Wh": E_gen,
            "P_sun_W": P_sun,
            "P_ecl_W": P_ecl,
            "E_use_sun_Wh": E_use_sun,
            "E_use_ecl_Wh": E_use_ecl,
            "E_use_Wh": E_use,
            "Margin_Wh": E_gen - E_use,
            "SOC": soc,
            "Data_gen_MB": data_gen,
            "Downlinked_MB": downlinked,
            "Backlog_MB": backlog,
            "V_ocv_pack": v_ocv,
            "V_load_ecl": v_load_ecl,
            "I_ecl_A": i_ecl,
            "BROWNOUT_WARN": bool(brownout_warn),
            "BROWNOUT": bool(brownout),
            "LOW_WARN": bool(low_warn),
            "LOW_HARD": bool(low_hard),
            "DATA_FULL": bool(data_full),
        })

    return pd.DataFrame(rows), E_usable, temp_factor


df, E_usable, temp_factor = run_sim(int(N_orbits), np.random.default_rng(42))
df = add_time_columns(df)


# -------------------------
# Summary
# -------------------------
T_min = float(df["T_min"].iloc[0])
sun_min = float(df["Sun_min"].iloc[0])
ecl_min = float(df["Ecl_min"].iloc[0])
f_e = float(df["Eclipse_frac"].iloc[0])

soc_min = float(df["SOC"].min())
soc_end = float(df["SOC"].iloc[-1])
margin_p5 = float(np.percentile(df["Margin_Wh"], 5))
backlog_end = float(df["Backlog_MB"].iloc[-1])
vmin = float(df["V_load_ecl"].min())

any_hard = bool(df["LOW_HARD"].any())
any_data_full = bool(df["DATA_FULL"].any())
any_brown = bool(df["BROWNOUT"].any())
any_brown_warn = bool(df["BROWNOUT_WARN"].any())

status = "STABLE"
if any_brown or any_hard:
    status = "FAIL RISK"
elif (soc_min < soc_warn) or (margin_p5 < 0) or any_brown_warn:
    status = "MARGINAL"

st.subheader("Summary")
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("Period (min)", f"{T_min:.1f}")
m2.metric("Eclipse (%)", f"{f_e*100:.1f}")
m3.metric("Usable batt (Wh)", f"{E_usable:.2f}")
m4.metric("SOC min (%)", f"{soc_min*100:.1f}")
m5.metric("Margin p5 (Wh/orbit)", f"{margin_p5:.2f}")
m6.metric("Backlog end (MB)", f"{backlog_end:.1f}")
m7.metric("Min V(load,ecl) (V)", f"{vmin:.2f}")

if status == "STABLE":
    st.success(f"Status: {status} | Sun {sun_min:.1f} min • Eclipse {ecl_min:.1f} min | Temp factor {temp_factor:.3f}")
elif status == "MARGINAL":
    st.warning(f"Status: {status} | Sun {sun_min:.1f} min • Eclipse {ecl_min:.1f} min | Temp factor {temp_factor:.3f}")
else:
    st.error(f"Status: {status} | Sun {sun_min:.1f} min • Eclipse {ecl_min:.1f} min | Temp factor {temp_factor:.3f}")

if any_data_full:
    st.warning("Data queue hit max capacity at least once. Increase downlink, reduce imaging, or increase storage.")
if any_brown:
    st.error("Brownout occurred at least once (eclipse-load voltage fell below threshold).")
elif any_brown_warn:
    st.warning("Voltage got close to brownout (within warning margin).")

st.markdown("---")


# -------------------------
# Plots
# -------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("SOC Trace")
    x = df["Orbit"].to_numpy()
    y = (df["SOC"] * 100.0).to_numpy()

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(x, y)
    marker_scatter(ax, x, y, (df["Mode"] == "IMAGING").to_numpy(), "IMAGING")
    marker_scatter(ax, x, y, (df["Mode"] == "DOWNLINK").to_numpy(), "DOWNLINK")
    marker_scatter(ax, x, y, (df["Mode"] == "SAFE").to_numpy(), "SAFE")
    ax.set_xlabel("Orbit")
    ax.set_ylabel("SOC (%)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig)

with c2:
    st.subheader("3S Pack Voltage (Eclipse Load)")
    x = df["Orbit"].to_numpy()
    v = df["V_load_ecl"].to_numpy()

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(x, v)
    ax.axhline(brownout_v, linestyle="--")
    marker_scatter(ax, x, v, (df["BROWNOUT_WARN"]).to_numpy(), "Near threshold")
    marker_scatter(ax, x, v, (df["BROWNOUT"]).to_numpy(), "Brownout")
    ax.set_xlabel("Orbit")
    ax.set_ylabel("V_load_ecl (V)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig)

c3, c4 = st.columns(2)

with c3:
    st.subheader("Data Backlog")
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(df["Orbit"], df["Backlog_MB"])
    marker_scatter(
        ax,
        df["Orbit"].to_numpy(),
        df["Backlog_MB"].to_numpy(),
        (df["Downlinked_MB"] > 0).to_numpy(),
        "Downlink event",
    )
    ax.set_xlabel("Orbit")
    ax.set_ylabel("Backlog (MB)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    plt.tight_layout()
    st.pyplot(fig)

with c4:
    st.subheader("Mode Timeline")
    mode_map = {"SAFE": 0, "NOMINAL": 1, "IMAGING": 2, "DOWNLINK": 3}
    y = df["Mode"].map(mode_map).astype(int)
    fig = plt.figure()
    plt.step(df["Orbit"], y, where="mid")
    plt.yticks([0, 1, 2, 3], ["SAFE", "NOMINAL", "IMAGING", "DOWNLINK"])
    plt.xlabel("Orbit")
    plt.ylabel("Mode")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")


# -------------------------
# Timeline views
# -------------------------
st.subheader("Mission Timeline")

left, right = st.columns([1, 1])

with left:
    st.markdown("**Events only**")
    events = df[
        (df["Mode"] != "NOMINAL")
        | (df["LOW_WARN"])
        | (df["DATA_FULL"])
        | (df["BROWNOUT_WARN"])
        | (df["BROWNOUT"])
    ].copy()

    events["SOC_%"] = (events["SOC"] * 100.0).round(2)
    events["V_ecl"] = events["V_load_ecl"].round(2)

    flags_low_soc = pd.Series(np.where(events["LOW_WARN"], "LOW_SOC ", ""), index=events.index)
    flags_low_v = pd.Series(np.where(events["BROWNOUT_WARN"], "LOW_V ", ""), index=events.index)
    flags_data_full = pd.Series(np.where(events["DATA_FULL"], "DATA_FULL ", ""), index=events.index)
    flags_brown = pd.Series(np.where(events["BROWNOUT"], "BROWNOUT ", ""), index=events.index)

    events["Flags"] = (flags_low_soc + flags_low_v + flags_data_full + flags_brown).str.strip()

    show = events[
        ["Orbit", "Mission_day", "HHMM", "Mode", "SOC_%", "V_ecl", "I_ecl_A", "Downlinked_MB", "Backlog_MB", "Flags"]
    ]
    st.dataframe(show, use_container_width=True, height=320)

with right:
    st.markdown("**Compact timeline**")
    tl = df.copy()
    tl["SOC_%"] = (tl["SOC"] * 100.0).round(2)
    tl["V_ecl"] = tl["V_load_ecl"].round(2)

    tl["⚠️"] = np.where(tl["LOW_WARN"], "⚠️", "")
    tl["🧯"] = np.where(tl["LOW_HARD"], "🧯", "")
    tl["📦"] = np.where(tl["DATA_FULL"], "📦", "")
    tl["🔻V"] = np.where(tl["BROWNOUT_WARN"], "🔻", "")
    tl["💥V"] = np.where(tl["BROWNOUT"], "💥", "")

    tl["DL_MB"] = tl["Downlinked_MB"].round(2)

    tl_show = tl[
        ["Orbit", "Mission_day", "HHMM", "Mode", "SOC_%", "V_ecl", "I_ecl_A", "DL_MB", "Backlog_MB", "⚠️", "🧯", "🔻V", "💥V", "📦"]
    ]
    st.dataframe(tl_show, use_container_width=True, height=320)

st.markdown("---")


# -------------------------
# Orbit Log + export
# -------------------------
st.subheader("Orbit Log")
view = df.copy()
view["SOC_%"] = (view["SOC"] * 100.0).round(2)

cols = [
    "Orbit", "Mission_day", "HHMM", "Mode",
    "Inc_deg", "P_SA_W", "E_gen_Wh",
    "P_sun_W", "P_ecl_W",
    "E_use_sun_Wh", "E_use_ecl_Wh", "E_use_Wh",
    "Margin_Wh", "SOC_%",
    "V_ocv_pack", "V_load_ecl", "I_ecl_A",
    "Data_gen_MB", "Downlinked_MB", "Backlog_MB",
    "LOW_WARN", "LOW_HARD", "BROWNOUT_WARN", "BROWNOUT", "DATA_FULL"
]
st.dataframe(view[cols], use_container_width=True)

st.download_button(
    "Download orbit log (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="cubesat_orbit_log.csv",
    mime="text/csv",
)

st.markdown("---")


# -------------------------
# Monte Carlo
# -------------------------
st.subheader("Monte Carlo")

if mc_runs <= 0:
    st.info("Set runs > 0 to enable.")
else:
    rng = np.random.default_rng(7)
    results = []

    base = dict(
        A_m2=A_m2,
        cell_eff=cell_eff,
        pack_factor=pack_factor,
        mppt_eff=mppt_eff,
        beta_derate=beta_derate,
        degradation=degradation,
        E_bat_Wh=E_bat_Wh,
        base_inc_deg=base_inc_deg,
        panel_temp_c=panel_temp_c,
        r_pack_ohm=r_pack_ohm,
    )

    for _ in range(int(mc_runs)):
        p = dict(base)

        for k in ["A_m2", "cell_eff", "pack_factor", "mppt_eff", "beta_derate", "degradation", "E_bat_Wh"]:
            p[k] = max(1e-9, p[k] * float(rng.normal(1.0, unc_pct / 2.0))) if unc_pct > 0 else p[k]

        p["base_inc_deg"] = float(np.clip(p["base_inc_deg"] + rng.normal(0.0, 2.0), 0.0, 85.0))
        p["panel_temp_c"] = float(p["panel_temp_c"] + rng.normal(0.0, 3.0))
        p["r_pack_ohm"] = max(0.0, p["r_pack_ohm"] * float(rng.normal(1.0, unc_pct / 2.0))) if unc_pct > 0 else p["r_pack_ohm"]

        T_min_mc = orbital_period_minutes(alt_km)
        f_e_mc = eclipse_fraction_from_geometry(alt_km, beta_deg) if eclipse_override is None else float(eclipse_override)
        t_ecl_h_mc = (T_min_mc * f_e_mc) / 60.0
        t_sun_h_mc = (T_min_mc * (1.0 - f_e_mc)) / 60.0

        temp_factor_mc = max(0.0, 1.0 - temp_coeff * (p["panel_temp_c"] - 25.0))
        E_usable_mc = max(1e-6, p["E_bat_Wh"] * (1.0 - cap_fade) * DoD)

        soc = clip01(SOC0)
        backlog = 0.0

        margins = []
        vmins = []

        hard_any = False
        data_full_any = False
        brown_any = False

        for orbit in range(1, int(N_orbits) + 1):
            imaging_due = (imaging_every > 0) and (orbit % imaging_every == 0)
            downlink_due = (downlink_every > 0) and (orbit % downlink_every == 0)

            if soc < force_safe_below:
                mode = "SAFE"
            elif downlink_due:
                mode = "DOWNLINK"
            elif imaging_due:
                mode = "IMAGING"
            else:
                mode = "NOMINAL"

            if soc < soc_hard:
                mode = "SAFE"

            inc = float(np.clip(p["base_inc_deg"] + rng.normal(0.0, inc_jitter_deg), 0.0, 85.0))

            P_sa_mc = array_power_w(
                p["A_m2"], p["cell_eff"], p["pack_factor"], p["mppt_eff"],
                inc, p["beta_derate"], p["degradation"], temp_factor_mc
            )
            E_gen_mc = P_sa_mc * t_sun_h_mc

            P_sun_mc = avg_power(modes[mode]["sun"])
            P_ecl_mc = avg_power(modes[mode]["ecl"])
            E_use_sun_mc = P_sun_mc * t_sun_h_mc
            E_use_ecl_mc = P_ecl_mc * t_ecl_h_mc
            margin = E_gen_mc - (E_use_sun_mc + E_use_ecl_mc)
            margins.append(margin)

            _, vload, _ = pack_voltage_under_load_3s(soc, P_ecl_mc, p["r_pack_ohm"], v_cell_min, v_cell_max)
            vmins.append(vload)
            brown_any = brown_any or (vload <= brownout_v)

            data_gen = data_gen_MB if mode == "IMAGING" else 0.0
            backlog = min(data_queue_max, backlog + data_gen)

            if mode == "DOWNLINK" and bitrate_kbps > 0:
                plen = max(0.0, float(pass_minutes + rng.normal(0.0, pass_jitter)))
                if plen > 0 and ((not downlink_sun_only) or (t_sun_h_mc * 60.0 > 0.0)):
                    cap_MB = (bitrate_kbps * 60.0 * plen) / (8.0 * 1024.0)
                    backlog -= min(backlog, cap_MB)

            net = E_gen_mc - E_use_sun_mc
            if net >= 0:
                taper = charge_taper(soc, taper_start, taper_floor)
                soc += (net * eta_c * taper) / E_usable_mc
            else:
                soc += (net / max(1e-9, eta_d)) / E_usable_mc

            soc -= (E_use_ecl_mc / max(1e-9, eta_d)) / E_usable_mc
            soc = clip01(soc)

            hard_any = hard_any or (soc < soc_hard)
            data_full_any = data_full_any or (backlog >= data_queue_max - 1e-6)

        results.append({
            "SOC_min": float(min(clip01(x) for x in [soc] + [soc])) if False else float(np.nan),  # placeholder removed below
            "SOC_end": float(soc),
            "Margin_mean": float(np.mean(margins)) if margins else 0.0,
            "Margin_p5": float(np.percentile(margins, 5)) if margins else 0.0,
            "Vmin_load_ecl": float(np.min(vmins)) if vmins else 0.0,
            "Hard_any": bool(hard_any),
            "Data_full_any": bool(data_full_any),
            "Brownout_any": bool(brown_any),
        })

    mc = pd.DataFrame(results)

    socmins = []
    for _ in range(int(mc_runs)):
        socmins.append(np.nan)

    a, b, c, d, e = st.columns(5)
    a.metric("Fail risk (SOC<hard)", f"{(mc['Hard_any'].mean()*100):.1f}%")
    b.metric("Data overflow risk", f"{(mc['Data_full_any'].mean()*100):.1f}%")
    c.metric("Brownout risk", f"{(mc['Brownout_any'].mean()*100):.1f}%")
    d.metric("Vmin p5 (V)", f"{np.percentile(mc['Vmin_load_ecl'], 5):.2f}")
    e.metric("Margin p5 (Wh/orbit)", f"{np.percentile(mc['Margin_p5'], 5):.2f}")

    p1, p2 = st.columns(2)
    with p1:
        fig = plt.figure()
        plt.hist(mc["Vmin_load_ecl"], bins=30)
        plt.xlabel("Min V(load,ecl) over run (V)")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig)

    with p2:
        fig = plt.figure()
        plt.hist(mc["Margin_p5"], bins=30)
        plt.xlabel("Margin p5 (Wh/orbit)")
        plt.ylabel("Count")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.tight_layout()
        st.pyplot(fig)

    st.download_button(
        "Download Monte Carlo results (CSV)",
        data=mc.to_csv(index=False).encode("utf-8"),
        file_name="cubesat_mc_results.csv",
        mime="text/csv",
    )


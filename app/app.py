import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# ----------------------------
# Styles
# ----------------------------
st.set_page_config(page_title="ClimaCrete Systems", layout="wide")

plt.rcParams.update({
    'axes.facecolor': "#224130",   # plot background
    'figure.facecolor': "#224130", # entire figure background
    'axes.edgecolor': '#9dc038',   # frame color
    'xtick.color': '#9dc038',
    'ytick.color': '#9dc038',
    'axes.labelcolor': '#9dc038',
    'grid.color': '#9dc038',
    'grid.alpha': 0.3,
    'text.color': '#e7e9dc',       # for titles & annotations
})

st.markdown("""
    <style>
    /* Background */
    .stApp {
        background-color: #224130;
    }

    /* Main text (headings) */
    h1, h2, h3, h4, h5, h6 {
        color: #e7e9dc !important;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* Subheaders and smaller highlight sections */
    h3, h4 {
        color: #224130 !important;
        padding: 0.3em 0.5em;
        border-radius: 4px;
        display: inline-block;
    }

    /* General text */
    p, li, label, span, div {
        color: #e7e9dc;
    }

    /* Metric boxes */
    [data-testid="stMetricValue"] {
        color: #d6e0a5 !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #4a6d1d;
    }

    /* Tables */
    table {
        background-color: #616718 !important;
        color: #e7e9dc !important;
        border-radius: 6px;
    }
    thead {
        background-color: #d6e0a5 !important;
        color: #224130 !important;
    }

    /* Plot background */
    .stPlotlyChart, .stPyplot {
        background-color: #d6e0a5 !important;
        border-radius: 6px;
        padding: 10px;
    }

    /* Buttons */
    button[kind="primary"] {
        background-color: #616718 !important;
        color: #e7e9dc !important;
        border-radius: 6px;
    }
    /* Tooltips*/
    span[title] {
        position: relative;
        cursor: help;
    }

    span[title]:hover::after {
        content: attr(title);
        position: absolute;
        background-color: #d6e0a5;
        color: #224130;
        padding: 6px 8px;
        border-radius: 5px;
        top: 1.8em;
        left: 0;
        white-space: normal;
        font-size: 0.85em;
        width: 220px;
        z-index: 100;
        box-shadow: 0 0 5px rgba(0,0,0,0.3);
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        color: black;
        text-align: center;
    }
    </style>
    <div class="footer">
    <p>Developed with ❤ by <a href="https://github.com/pink-hat-hacker" target="_blank">Zoë Valladares</a></p>
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# Load trained models
# ----------------------------
STRENGTH_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/strength_model.pkl")
DURABILITY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models/durability_model.pkl")

strength_model = joblib.load(STRENGTH_MODEL_PATH)
durability_model = joblib.load(DURABILITY_MODEL_PATH)

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🪏 ClimaCrete Systems")
st.subheader("ML-powered Low-Carbon Concrete Performance Predictor")

st.markdown("""
*This MVP predicts **COMPRESSIVE STRENGTH**, **DURABILITY INDEX**, and **CO₂ IMPACT** 
for various mix designs. It also provides recommended use cases 
based on performance trade-offs.*
""")
st.markdown("---")
# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("Input Mix Design Parameters")

cement = st.sidebar.number_input("Cement (kg/m³)", 0.0, 800.0, 380.0, 10.0)
slag = st.sidebar.number_input("Slag (kg/m³)", 0.0, 800.0, 0.0, 10.0)
flyash = st.sidebar.number_input("Fly Ash (kg/m³)", 0.0, 800.0, 0.0, 10.0)
water = st.sidebar.number_input("Water (kg/m³)", 0.0, 400.0, 160.0, 5.0)
superplasticizer = st.sidebar.number_input("Superplasticizer (kg/m³)", 0.0, 20.0, 5.0, 0.5)
coarse_agg = st.sidebar.number_input("Coarse Aggregate (kg/m³)", 0.0, 1200.0, 1000.0, 10.0)
fine_agg = st.sidebar.number_input("Fine Aggregate (kg/m³)", 0.0, 1200.0, 700.0, 10.0)
age = st.sidebar.slider("Age (days)", 1, 90, 28)
recycled_agg_pct = st.sidebar.slider("Recycled Aggregate (%)", 0, 100, 0)
fiber_flag = st.sidebar.selectbox("Fiber Reinforcement", ["No", "Yes"]) == "Yes"

# Derived parameters
scm = slag + flyash
binder = cement + scm
w_b_ratio = water / binder if binder > 0 else 0
scm_frac = scm / binder if binder > 0 else 0

# ----------------------------
# Build model input
# ----------------------------
input_df = pd.DataFrame([{
    'cement': cement,
    'slag': slag,
    'flyash': flyash,
    'scm': scm,
    'water': water,
    'superplasticizer': superplasticizer,
    'coarse_agg': coarse_agg,
    'fine_agg': fine_agg,
    'binder': binder,
    'w_b_ratio': w_b_ratio,
    'scm_frac': scm_frac,
    'recycled_agg_pct': recycled_agg_pct,
    'fiber_flag': int(fiber_flag),
    'age': age
}])

# ----------------------------
# Prediction
# ----------------------------
if st.sidebar.button("Predict Performance"):
    strength = strength_model.predict(input_df)[0]
    durability = durability_model.predict(input_df)[0]

    # --- CO2 Emission Estimation ---
    # Simple embodied carbon factor (kg CO₂ per kg binder)
    co2_cement = 0.9      # ordinary Portland cement
    co2_scm = 0.15        # SCMs (fly ash, slag)
    total_co2 = (cement * co2_cement + scm * co2_scm)
    co2_reduction = (1 - total_co2 / (binder * co2_cement)) * 100 if binder > 0 else 0

    # --- Recommended Use Case ---
    if strength >= 60 and durability >= 80:
        use_case = "🌟 Structural / Precast Elements"
    elif strength >= 40 and durability >= 60:
        use_case = "🧩 General Structural Concrete"
    elif strength >= 25:
        use_case = "🪨 Non-Structural Applications"
    else:
        use_case = "⚠️ Mix likely unsuitable for most load-bearing uses"

    # --- Display Results ---
    st.subheader("Predicted Performance")
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Compressive Strength (MPa)", f"{strength:.2f}")
    col1.markdown(
        '<span title="Compressive Strength measures how much load concrete can withstand before failing — higher MPa means stronger concrete.">ℹ️</span>',
        unsafe_allow_html=True
    )
    
    col2.metric("Durability Index (0–100)", f"{durability:.1f}")
    col2.markdown(
        '<span title="Durability Index is an ML-estimated measure (0–100) of how well the concrete resists weathering, cracking, and chemical attack.">ℹ️</span>', 
        unsafe_allow_html=True
    )

    col3.metric("CO₂ Reduction vs. OPC (%)", f"{co2_reduction:.1f}")
    col3.markdown(
        '<span title="CO₂ Reduction compares your mix’s embodied carbon to traditional Portland cement — higher % means lower climate impact.">ℹ️</span>', 
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown(f"### **Recommended Use Case:** {use_case}")

    st.markdown("---")

    # --- Visualization: Trade-off Plot ---
    # st.subheader("Strength–Durability Trade-off by SCM Fraction")
    st.markdown("## Strength–Durability Trade-off by SCM Fraction <span title='This chart shows how increasing the use of supplementary cementitious materials (SCMs) — like fly ash or slag — affects both concrete strength and durability.'>ℹ️</span>", unsafe_allow_html=True)

    scm_range = np.linspace(0, 0.8, 20)
    tradeoff_data = []
    for s in scm_range:
        row = input_df.copy()
        row['scm_frac'] = s
        row['slag'] = s * binder
        row['scm'] = s * binder
        tradeoff_strength = strength_model.predict(row)[0]
        tradeoff_durability = durability_model.predict(row)[0]
        tradeoff_data.append((s, tradeoff_strength, tradeoff_durability))

    tradeoff_df = pd.DataFrame(tradeoff_data, columns=['SCM Fraction', 'Strength', 'Durability'])

    fig, ax1 = plt.subplots()
    ax1.grid(True, linestyle='--', linewidth=0.7, alpha=0.4)

    ax1.plot(tradeoff_df['SCM Fraction'], tradeoff_df['Strength'], color='#d2491c', label='Strength (MPa)')
    ax1.set_xlabel('SCM Fraction of Binder', color='#9dc038')
    ax1.set_ylabel('Compressive Strength (MPa)', color='#d2491c')

    ax2 = ax1.twinx()
    ax2.plot(tradeoff_df['SCM Fraction'], tradeoff_df['Durability'], color='#9ea2aa', label='Durability Index')
    ax2.set_ylabel('Durability Index (0–100)', color='#9ea2aa')

    fig.tight_layout()
    st.pyplot(fig)

    st.caption("Note: SCM (Supplementary Cementitious Materials) such as fly ash or slag generally reduce CO₂ while improving durability but can reduce early strength.")


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
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
fiber_flag = st.sidebar.selectbox("Fiber Reinforcement", ["No", "Yes"]) == "Yes" # done

# '''
# | Input (User Provides)         | Description                 | Derived / Used For                    |
# | ----------------------------- | --------------------------- | ------------------------------------- |
# | Cement, Slag, Fly Ash (kg/m³) | Core binder materials       | Determines binder mass & SCM fraction |
# | Water (kg/m³)                 | Hydration agent             | Used for water/binder ratio           |
# | Superplasticizer (kg/m³)      | Flow enhancer               | Affects workability                   |
# | Coarse & Fine Aggregates      | Load distribution & density | Affects strength indirectly           |
# | Age (days)                    | Curing time                 | Used in both models                   |
# | Recycled Aggregate (%)        | Sustainability input        | Impacts durability                    |
# | Fiber Reinforcement           | Cracking resistance         | Impacts durability                    |

# '''
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

    # --- CO2 Emission Estimation --- (kg CO₂ per kg binder)
    co2_cement = 0.9    # portland cement
    co2_scm = 0.15  # SCMs (fly ash, slag)
    total_co2 = (cement * co2_cement + scm * co2_scm)
    co2_reduction = (1 - total_co2 / (binder * co2_cement)) * 100 if binder > 0 else 0

    # --- Recommended Use Case ---
    # '''
    # | Category                 | Typical Strength (MPa) | Durability Range | Real-world Examples                               | Reasoning                                                        |
    # | ------------------------ | ---------------------- | ---------------- | ------------------------------------------------- | ---------------------------------------------------------------- |
    # | **Structural / Precast** | ≥ 60                   | ≥ 80             | Bridge decks, precast beams, high-rise columns    | High strength + durability needed under heavy loads and exposure |
    # | **General Structural**   | 40–60                  | 60–80            | Regular reinforced concrete (slabs, walls, beams) | Standard for most building-grade mixes                           |
    # | **Non-Structural**       | 25–40                  | Any              | Pavements, blocks, fillers                        | Load demands are lower; focus is on workability and cost         |
    # | **Unsuitable**           | < 25                   | —                | Lightweight or failed mix designs                 | Insufficient strength for most applications                      |

    # '''
    if strength >= 60 and durability >= 80:
        use_case = "🌟 Structural / Precast Elements"
    elif strength >= 40 and durability >= 60:
        use_case = "🧩 General Structural Concrete"
    elif strength >= 25:
        use_case = "🪨 Non-Structural Applications"
    else:
        use_case = "⚠️ Mix likely unsuitable for most load-bearing uses"

    # --- show results ---
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

    st.markdown("### Explore Further Insights")

    # --- Binder Composition by SCM Fraction ---
    with st.expander("Binder Composition by SCM Fraction", expanded=False):
        st.markdown(
            "<span title='This chart shows how the binder composition changes as cement is replaced by supplementary cementitious materials (SCMs) such as slag or fly ash. Higher SCM fractions reduce embodied carbon but may affect strength.'>ℹ️</span>",
            unsafe_allow_html=True
        )

        scm_range = np.linspace(0, 0.8, 20)
        cement_vals, slag_vals, flyash_vals = [], [], []

        for s in scm_range:
            cement_mass = (1 - s) * binder
            slag_mass = s * binder * (slag / scm if scm > 0 else 0.5)
            flyash_mass = s * binder * (flyash / scm if scm > 0 else 0.5)
            
            cement_vals.append(cement_mass)
            slag_vals.append(slag_mass)
            flyash_vals.append(flyash_mass)

        fig2, ax = plt.subplots(figsize=(8, 5))
        ax.bar(scm_range, cement_vals, label="Cement", color="#d2491c")
        ax.bar(scm_range, slag_vals, bottom=cement_vals, label="Slag", color="#9ea2aa")
        ax.bar(scm_range, flyash_vals, bottom=np.array(cement_vals) + np.array(slag_vals), label="Fly Ash", color="#d6e0a5")

        ax.set_xlabel("SCM Fraction of Binder", color="#e7e9dc")
        ax.set_ylabel("Binder Composition (kg/m³)", color="#e7e9dc")
        ax.set_title("Binder Composition vs. SCM Fraction", color="#d6e0a5", pad=10)
        ax.legend(facecolor="#224130", labelcolor="#d6e0a5")
        ax.grid(True, color="#9dc038", alpha=0.3)
        ax.set_facecolor("#224130")
        fig2.patch.set_facecolor("#224130")

        st.pyplot(fig2)
    # --- Strength and Durability Over Time ---
    with st.expander("Strength and Durability Over Time", expanded=False):
        st.markdown(
            "<span title='This shows how the concrete’s compressive strength and durability evolve as it cures. It uses the same mix inputs and highlights your selected curing age.'>ℹ️</span>",
            unsafe_allow_html=True
        )

        # Age points for the curve
        age_points = np.array([7, 14, 28, 56, 90])
        strength_curve, durability_curve = [], []

        for t in age_points:
            row = input_df.copy()
            row['age'] = t
            strength_curve.append(strength_model.predict(row)[0])
            durability_curve.append(durability_model.predict(row)[0])

        fig3, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.plot(age_points, strength_curve, color="#d2491c", marker="o", label="Strength (MPa)")
        ax1.set_xlabel("Age (days)", color="#e7e9dc")
        ax1.set_ylabel("Compressive Strength (MPa)", color="#d2491c")
        ax2.plot(age_points, durability_curve, color="#9ea2aa", marker="s", linestyle="--", label="Durability Index")
        ax2.set_ylabel("Durability Index (0–100)", color="#9ea2aa")

        if 'age' in input_df.columns:
            age_point = input_df['age'].values[0]
            if age_point in age_points:
                idx = np.where(age_points == age_point)[0][0]
                ax1.scatter(age_point, strength_curve[idx], color="#d2491c", s=80, edgecolor="#e7e9dc", zorder=5)
                ax2.scatter(age_point, durability_curve[idx], color="#9ea2aa", s=80, edgecolor="#e7e9dc", zorder=5)
            else:
                age_interp_strength = np.interp(age_point, age_points, strength_curve)
                age_interp_durability = np.interp(age_point, age_points, durability_curve)
                ax1.scatter(age_point, age_interp_strength, color="#d2491c", s=80, edgecolor="#e7e9dc", zorder=5)
                ax2.scatter(age_point, age_interp_durability, color="#9ea2aa", s=80, edgecolor="#e7e9dc", zorder=5)
            ax1.axvline(age_point, color="#d6e0a5", linestyle=":", linewidth=1)
            ax1.text(age_point + 1, ax1.get_ylim()[1]*0.9, f"Age: {int(age_point)} days",
                     color="#d6e0a5", fontsize=10, va="top")

        # styles
        ax1.grid(True, color="#9dc038", alpha=0.3)
        ax1.set_facecolor("#224130")
        fig3.patch.set_facecolor("#224130")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best", facecolor="#224130", labelcolor="#d6e0a5")

        st.pyplot(fig3)

        st.caption("Note: Strength typically increases with age as hydration continues, while durability stabilizes once the cement matrix densifies.")

# ----------------------------
# Optimal Mix Finder
# ----------------------------
st.markdown("## ♻️ ML-Recommended Optimal Mix Design")
st.markdown("The system searches for a mix with the lowest CO₂ impact that still meets structural performance requirements (≥40 MPa, ≥70 durability).")
st.markdown("Enter in your preferred mix and hit this button to get a comparison chart.")

if st.button("Find Optimal Mix"):
    cement_range = np.linspace(250, 600, 15)
    scm_frac_range = np.linspace(0.0, 0.7, 15)
    w_b_range = np.linspace(0.25, 0.65, 15)


    best_mix = None
    lowest_co2 = np.inf
    all_results = []
    best_score = -np.inf

    for c in cement_range:
        for s in scm_frac_range:
            for w_b in w_b_range:
                binder = 400.0
                cement = binder * (1 - s)
                water = w_b * binder
                scm = s * binder
                slag = scm  # simplified, or randomize between slag/flyash
                flyash = 0.0

                input_row = pd.DataFrame([{
                    'cement': c,
                    'slag': slag,
                    'flyash': flyash,
                    'scm': scm,
                    'water': water,
                    'superplasticizer': 5.0,
                    'coarse_agg': 1000.0,
                    'fine_agg': 700.0,
                    'binder': binder,
                    'w_b_ratio': w_b,
                    'scm_frac': s,
                    'recycled_agg_pct': 0,
                    'fiber_flag': 0,
                    'age': 28
                }])

                strength_pred = strength_model.predict(input_row)[0]
                durability_pred = durability_model.predict(input_row)[0]

                co2_cement = 0.9
                co2_scm = 0.15
                total_co2 = (c * co2_cement + scm * co2_scm)

                # fallback
                all_results.append({
                    'cement': cement, 'scm_frac': s, 'w_b_ratio': w_b,
                    'strength': strength_pred, 'durability': durability_pred,
                    'co2': total_co2
                })
                score = (strength_pred/60)*0.4 + (durability_pred/100)*0.4 + (1 - total_co2/400)*0.2
                if score > best_score:
                    best_score = score
                    best_mix = all_results[-1]

                if strength_pred >= 35 and durability_pred >= 55:
                    if total_co2 < lowest_co2:
                        lowest_co2 = total_co2
                        best_mix = {
                            'cement': c,
                            'scm_frac': s,
                            'w_b_ratio': w_b,
                            'strength': strength_pred,
                            'durability': durability_pred,
                            'co2': total_co2
                        }

    if best_mix:
        st.success("✅ Optimal low-carbon mix found:")
        st.write(best_mix)
        st.caption("This mix meets structural-grade requirements while minimizing carbon footprint.")
        
        # --- Visualization: Compare Input vs. AI Recommendation ---
        st.markdown("### 🧮 Comparison: Your Mix vs. ML Recommendation")

        # Compute values for user's input mix
        user_strength = strength_model.predict(input_df)[0]
        user_durability = durability_model.predict(input_df)[0]
        user_co2 = (cement * 0.9 + scm * 0.15)

        # Compute AI mix performance
        ai_strength = best_mix['strength']
        ai_durability = best_mix['durability']
        ai_co2 = best_mix['co2']

        labels = ["Compressive Strength (MPa)", "Durability Index", "CO₂ (kg/m³)"]
        user_vals = [user_strength, user_durability, user_co2]
        ai_vals = [ai_strength, ai_durability, ai_co2]

        x = np.arange(len(labels))
        width = 0.35

        fig4, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width/2, user_vals, width, label="Your Mix", color="#d2491c")
        rects2 = ax.bar(x + width/2, ai_vals, width, label="ML Mix", color="#9ea2aa")

        ax.set_ylabel("Value", color="#e7e9dc")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10, color="#e7e9dc")
        ax.legend(facecolor="#224130", labelcolor="#d6e0a5")
        ax.grid(True, color="#9dc038", alpha=0.3)
        ax.set_facecolor("#224130")
        fig4.patch.set_facecolor("#224130")

        # Annotate bars with numeric values
        for rects in [rects1, rects2]:
            for r in rects:
                ax.text(
                    r.get_x() + r.get_width()/2, r.get_height() + 1,
                    f"{r.get_height():.1f}",
                    ha='center', va='bottom', color="#e7e9dc", fontsize=9
                )

        st.pyplot(fig4)
        st.caption("Lower CO₂ while maintaining strength and durability shows improved sustainability.")
    else:
        st.warning("⚠️ No mix found within the tested ranges that meets the target criteria.")
        closest = min(all_results, key=lambda x: abs(x['strength']-40) + abs(x['durability']-70))
        st.write(closest)

    


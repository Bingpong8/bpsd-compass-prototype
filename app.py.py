import streamlit as st
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. DATABASE & PHARMACODYNAMIC DATA (pKi = -log10(K_i))
# ---------------------------------------------------------
DRUG_DATABASE = {
    "Brexpiprazole": {
        "pKi": {"5HT2A": 9.33, "D2": 9.52, "H1": 7.72, "alpha1": 8.42, "M1": 5.50},
        "Ar": {"5HT2A": 1.0, "D2": 1.0}
    },
    "Olanzapine": {
        "pKi": {"5HT2A": 8.40, "D2": 7.96, "H1": 8.15, "alpha1": 7.72, "M1": 7.59},
        "Ar": {"5HT2A": 1.0, "D2": 1.0}
    },
    "Risperidone": {
        "pKi": {"5HT2A": 9.30, "D2": 8.43, "H1": 7.70, "alpha1": 8.15, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": 1.0}
    },
    "Quetiapine": {
        "pKi": {"5HT2A": 6.70, "D2": 5.80, "H1": 8.00, "alpha1": 7.00, "M1": 6.00},
        "Ar": {"5HT2A": 1.0, "D2": 1.0}
    },
    "Haloperidol": {
        "pKi": {"5HT2A": 7.10, "D2": 8.70, "H1": 6.00, "alpha1": 7.20, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": -1.0}  # Full D2 antagonism opposes apathy goals
    },
    "Mirtazapine": {
        "pKi": {"5HT2A": 8.10, "D2": 5.00, "H1": 9.00, "alpha1": 7.20, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": 0.0}
    },
    "Mianserin": {
        "pKi": {"5HT2A": 8.20, "D2": 5.50, "H1": 8.80, "alpha1": 7.50, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": 0.0}
    },
    "Escitalopram": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "H1": 5.00, "alpha1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 0.0, "D2": 0.0}
    },
    "Sertraline": {
        "pKi": {"5HT2A": 5.00, "D2": 6.60, "H1": 5.00, "alpha1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 0.0, "D2": 0.5}
    },
    "Fluoxetine": {
        "pKi": {"5HT2A": 6.70, "D2": 5.00, "H1": 5.00, "alpha1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 0.5, "D2": 0.0}
    }
}

# ---------------------------------------------------------
# 2. CALCULATION ENGINE
# ---------------------------------------------------------
def calculate_match_score(drug_name, drug_data, weights, lambda_risks, mmse_score):
    pk = drug_data["pKi"]
    ar = drug_data["Ar"]
    
    # 1. Target Utility Component: sum(w_r * pKi * A_r)
    u_thera = (weights["5HT2A"] * pk["5HT2A"] * ar["5HT2A"]) + \
              (weights["D2"] * pk["D2"] * ar["D2"])
    
    # 2. Off-Target Risk Component: sum(lambda_r * pKi)
    # Includes D2 Full Antagonism risk if drug acts as inverse agonist/full antagonist
    d2_risk = (lambda_risks["D2_full"] * pk["D2"]) if ar["D2"] < 0 else 0.0
    
    u_risk = (lambda_risks["H1"] * pk["H1"]) + \
             (lambda_risks["alpha1"] * pk["alpha1"]) + \
             d2_risk
    
    # 3. Discontinuous Anticholinergic Cognitive Burden Penalty (PACB)
    if mmse_score < 10:
        c_patient = 3.0
    elif mmse_score <= 20:
        c_patient = 2.0
    else:
        c_patient = 1.0
        
    pacb = c_patient * 2.0 if pk["M1"] >= 7.0 else 0.0
    
    # Final Net Score Formula: M_j = U_thera - U_risk - P_ACB
    m_j = u_thera - u_risk - pacb
    
    return {
        "Drug": drug_name,
        "Net Score (Mj)": round(m_j, 2),
        "Therapeutic Gain": round(u_thera, 2),
        "Risk Deductions": round(u_risk, 2),
        "ACB Penalty (PACB)": round(pacb, 2),
        "M1 Potency (pKi)": pk["M1"]
    }

# ---------------------------------------------------------
# 3. STREAMLIT FRONTEND / USER INTERFACE
# ---------------------------------------------------------
st.set_page_config(page_title="BPSD Prescribing Compass", layout="wide")

st.title("🧠 BPSD Pharmacologic Decision-Support Compass")
st.caption("Parameter-driven clinical matching algorithm based on Neurotransmitter-Receptor Affinities")

st.markdown("---")

# Quick Clinical Presets (Enhances User-Friendliness)
st.sidebar.header("📋 Clinical Quick Presets")
preset = st.sidebar.selectbox(
    "Load Preset Patient Profile",
    ["Custom Inputs", "Severe Psychotic Agitation + High Fall Risk", "Severe Apathy + Parkinsonism Risk"]
)

# Preset Logic
if preset == "Severe Psychotic Agitation + High Fall Risk":
    init_w_5ht2a, init_w_d2 = 0.9, 0.2
    init_l_h1, init_l_a1, init_l_d2 = 0.9, 0.7, 0.0
    init_mmse = 12
elif preset == "Severe Apathy + Parkinsonism Risk":
    init_w_5ht2a, init_w_d2 = 0.2, 0.9
    init_l_h1, init_l_a1, init_l_d2 = 0.3, 0.2, 1.0
    init_mmse = 18
else:
    init_w_5ht2a, init_w_d2 = 0.9, 0.8
    init_l_h1, init_l_a1, init_l_d2 = 0.8, 0.7, 0.0
    init_mmse = 15

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Target Symptom Severity (Normalized ωr)")
    ω_5ht2a = st.slider("Psychotic Agitation Severity (5-HT2A)", 0.0, 1.0, init_w_5ht2a, 0.1)
    ω_d2 = st.slider("Apathy / Executive Dysfunction Severity (D2)", 0.0, 1.0, init_w_d2, 0.1)
    
    weights = {"5HT2A": ω_5ht2a, "D2": ω_d2}

with col2:
    st.subheader("Patient Risk Profile & Vulnerabilities (λr)")
    l_h1 = st.slider("Fall / Sedation Vulnerability (λH1)", 0.0, 1.0, init_l_h1, 0.1)
    l_alpha1 = st.slider("Orthostatic Hypotension Risk (λα1)", 0.0, 1.0, init_l_a1, 0.1)
    l_d2_full = st.slider("Parkinsonism / EPS Vulnerability (λD2)", 0.0, 1.0, init_l_d2, 0.1)
    
    mmse = st.number_input("Baseline MMSE Score (Cognitive Assessment)", min_value=0, max_value=30, value=init_mmse)
    
    lambda_risks = {"H1": l_h1, "alpha1": l_alpha1, "D2_full": l_d2_full}

st.markdown("---")
st.subheader("Calculated Drug Match Dashboard")

# Calculate results for all candidate drugs
results = [calculate_match_score(d, data, weights, lambda_risks, mmse) for d, data in DRUG_DATABASE.items()]
df_results = pd.DataFrame(results).sort_values(by="Net Score (Mj)", ascending=False).reset_index(drop=True)

# Highlight Top Recommended Option
top_drug = df_results.iloc[0]

st.markdown(
    f"""
    <div style="background-color: #d1e7dd; border-left: 8px solid #0f5132; padding: 18px; border-radius: 6px; margin-bottom: 20px;">
        <span style="font-size: 14px; color: #0f5132; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">Top Recommended Option</span>
        <h1 style="color: #0f5132; margin: 4px 0 0 0; font-size: 32px; font-weight: 800;">
            🏆 {top_drug['Drug']}
        </h1>
        <p style="color: #0f5132; font-size: 18px; margin: 6px 0 0 0;">
            Net Match Score (Mj): <strong>{top_drug['Net Score (Mj)']}</strong> 
            &nbsp;|&nbsp; Therapeutic Gain: <strong>+{top_drug['Therapeutic Gain']}</strong> 
            &nbsp;|&nbsp; Risk Penalty: <strong>-{top_drug['Risk Deductions']}</strong>
            &nbsp;|&nbsp; ACB Penalty: <strong>-{top_drug['ACB Penalty (PACB)']}</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Apply Styling Matrix to Net Score
def apply_traffic_lights(val):
    if val > 1.0:
        return 'background-color: #d4edda; color: #155724; font-weight: bold;'
    elif val >= -2.0:
        return 'background-color: #fff3cd; color: #856404;'
    else:
        return 'background-color: #f8d7da; color: #721c24;'

# Render Interactive Table
st.dataframe(
    df_results.style.map(apply_traffic_lights, subset=['Net Score (Mj)']),
    use_container_width=True
)

st.info("**Traffic Light Guide:** Green = Optimal Match (Mj > 1.0) | Yellow = Proceed with Caution (-2.0 ≤ Mj ≤ 1.0) | Red = High Risk Flag (Mj < -2.0)")

# Explainable AI Component (Expander Tooltips)
with st.expander("🔍 Breakdown of Receptor Calculation"):
    st.markdown(
        """
        The Net Score ($M_j$) combines target therapeutic efficacy, off-target toxicity risk, and cognitive impairment adjustments:
        $$M_j = \\sum (w_r \\cdot pK_{i,r} \\cdot A_r) - \\sum (\\lambda_r \\cdot pK_{i,r}) - P_{\\text{ACB}}$$
        """
    )
    st.table(df_results)

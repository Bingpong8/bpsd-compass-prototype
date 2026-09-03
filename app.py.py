import streamlit as st
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. DATABASE & PHARMACODYNAMIC DATA (pKi = -log10(K_i))
# ---------------------------------------------------------
DRUG_DATABASE = {
    "Brexpiprazole": {
        "pKi": {"5HT2A": 9.33, "D2": 9.52, "H1": 7.72, "alpha1": 8.42, "M1": 5.50},
        "A_r": {"5HT2A": 1.0, "D2": 1.0}
    },
    "Olanzapine": {
        "pKi": {"5HT2A": 8.40, "D2": 7.96, "H1": 8.15, "alpha1": 7.72, "M1": 7.59},
        "A_r": {"5HT2A": 1.0, "D2": 1.0}
    },
    "Risperidone": {
        "pKi": {"5HT2A": 9.30, "D2": 8.43, "H1": 7.70, "alpha1": 8.15, "M1": 5.00},
        "A_r": {"5HT2A": 1.0, "D2": 1.0}
    },
    "Pimavanserin": {
        "pKi": {"5HT2A": 9.05, "D2": 5.00, "H1": 5.00, "alpha1": 5.00, "M1": 5.00},
        "A_r": {"5HT2A": 1.0, "D2": 0.0}
    },
    "Haloperidol": {
        "pKi": {"5HT2A": 7.10, "D2": 8.70, "H1": 6.00, "alpha1": 7.20, "M1": 5.00},
        "A_r": {"5HT2A": 1.0, "D2": -1.0}  # Full D2 antagonism opposes apathy goals
    }
}

# ---------------------------------------------------------
# 2. CALCULATION ENGINE
# ---------------------------------------------------------
def calculate_match_score(drug_name, drug_data, weights, lambda_risks, mmse_score):
    pk = drug_data["pKi"]
    ar = drug_data["Ar"]
    
    # 1. Therapeutic Utility Component: sum(w_r * pKi * A_r)
    u_thera = (weights["5HT2A"] * pk["5HT2A"] * ar["5HT2A"]) + \
              (weights["D2"] * pk["D2"] * ar["D2"])
    
    # 2. Risk Component: sum(lambda_r * pKi)
    u_risk = (lambda_risks["H1"] * pk["H1"]) + \
             (lambda_risks["alpha1"] * pk["alpha1"])
    
    # 3. Discontinuous Anticholinergic Cognitive Burden Penalty (PACB)
    # Scaled by MMSE: severe impairment (MMSE < 10) = 3.0, moderate = 2.0, mild = 1.0
    if mmse_score < 10:
        c_patient = 3.0
    elif mmse_score <= 20:
        c_patient = 2.0
    else:
        c_patient = 1.0
        
    pacb = c_patient * 1.0 if pk["M1"] >= 7.0 else 0.0
    
    # Final Net Score Formula
    m_j = u_thera - u_risk - pacb
    
    return {
        "Drug": drug_name,
        "Net Score (Mj)": round(mj, 2),
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

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Patient Baseline & Symptom Severity (Inputs)")
    
    # Target Symptom Weights (w_r) via Sliders
    st.write("**Target Symptom Severity (Normalized w_r)**")
    w_5ht2a = st.slider("Psychotic Agitation Severity (5-HT2A)", 0.0, 1.0, 0.9, 0.1)
    w_d2 = st.slider("Apathy / Executive Dysfunction Severity (D2)", 0.0, 1.0, 0.8, 0.1)
    
    weights = {"5HT2A": w_5ht2a, "D2": w_d2}

with col2:
    st.subheader("2. Patient Vulnerabilities & Risk Profiles (Inputs)")
    
    # Risk Penalty Weights (lambda_r) via Sliders
    st.write("**Patient Clinical Risk Weights (λ_r)**")
    l_h1 = st.slider("Fall / Sedation Vulnerability (λ_H1)", 0.0, 1.0, 0.8, 0.1)
    l_alpha1 = st.slider("Orthostatic Hypotension Risk (λ_α1)", 0.0, 1.0, 0.7, 0.1)
    
    mmse = st.number_input("Baseline MMSE Score (Cognitive Assessment)", min_value=0, max_value=30, value=15)
    
    lambda_risks = {"H1": l_h1, "alpha1": l_alpha1}

st.markdown("---")
st.subheader("3. Calculated Drug Match Dashboard")

# Calculate results for all candidate drugs
results = []
for drug_name, drug_data in DRUG_DATABASE.items():
    res = calculate_match_score(drug_name, drug_data, weights, lambda_risks, mmse)
    results.append(res)

df_results = pd.DataFrame(results).sort_values(by="Net Score (Mj)", ascending=False)

# Render Top Recommended Drug Card
top_drug = df_results.iloc[0]
st.success(f"**Top Recommended Option:** {top_drug['Drug']} (Net Score Mj = {top_drug['Net Score (Mj)']})")

# Render Interactive Results Table
def color_score(val):
    if val > 1.0:
        return 'background-color: #d4edda; color: #155724;' # Green
    elif val >= -2.0:
        return 'background-color: #fff3cd; color: #856404;' # Yellow
    else:
        return 'background-color: #f8d7da; color: #721c24;' # Red

st.dataframe(
    df_results.style.map(color_score, subset=['Net Score (Mj)']),
    use_container_width=True
)

st.info("**Traffic Light Guide:** Green = Optimal Match | Yellow = Proceed with Caution | Red = Flagged High Toxicity Risk")

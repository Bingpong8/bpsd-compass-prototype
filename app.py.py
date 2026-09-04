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
# 2. CALCULATION ENGINE (Clean 1-Decimal Output)
# ---------------------------------------------------------
def calculate_match_score(drug_name, drug_data, weights, lambda_risks, mmse_score):
    pk = drug_data["pKi"]
    ar = drug_data["Ar"]
    
    # 1. Target Utility Component: sum(w_r * pKi * A_r)
    u_thera = (weights["5HT2A"] * pk["5HT2A"] * ar["5HT2A"]) + \
              (weights["D2"] * pk["D2"] * ar["D2"])
    
    # 2. Off-Target Risk Component: sum(lambda_r * pKi)
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
    
    # Rounding to 1 decimal place to avoid overwhelming clinicians with long floats
    return {
        "Drug": drug_name,
        "Net Score (Mj)": round(m_j, 1),
        "Therapeutic Gain": round(u_thera, 1),
        "Risk Deductions": round(u_risk, 1),
        "ACB Penalty": round(pacb, 1),
        "M1 Potency (pKi)": round(pk["M1"], 1)
    }

# ---------------------------------------------------------
# 3. STREAMLIT FRONTEND / USER INTERFACE
# ---------------------------------------------------------
st.set_page_config(page_title="BPSD Prescribing Compass", layout="wide")

st.title("🧠 BPSD Pharmacologic Decision-Support Compass")
st.caption("Parameter-driven clinical matching algorithm based on Neurotransmitter-Receptor Affinities")

st.markdown("---")

# Anchor Scale Mappings (Translates Clinical Assessments to Weights)
NPI_MAPPING = {
    "0 - Absent / No Symptoms": 0.0,
    "1 - Mild (Slight distress, no impairment)": 0.3,
    "2 - Moderate (Significant distress, partial impairment)": 0.7,
    "3 - Severe (Major disruption, marked impairment)": 1.0
}

FALL_RISK_MAPPING = {
    "Low Risk (Morse Score 0-24)": 0.1,
    "Moderate Risk (Morse Score 25-44)": 0.5,
    "High Risk (Morse Score ≥ 45 or history of falls)": 0.9
}

ORTHO_BP_MAPPING = {
    "Normal (< 10 mmHg drop upon standing)": 0.1,
    "Subclinical Drop (10-19 mmHg drop)": 0.5,
    "Diagnostic Orthostasis (≥ 20 mmHg SBP drop)": 0.9
}

PARKINSONISM_MAPPING = {
    "None (Normal muscle tone and gait)": 0.0,
    "Mild (Pre-existing mild tremor or rigidity)": 0.5,
    "Severe (Diagnosed Parkinsonism / DLB / High SAS score)": 1.0
}

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Target Symptom Severity (Bedside Anchor Scales)")
    
    # Bedside Rating Scales replacing abstract floats
    npi_agit = st.selectbox(
        "Psychotic Agitation / Aggression Severity (NPI-Q Scale)",
        options=list(NPI_MAPPING.keys()),
        index=3,
        help="Mapped to target 5-HT2A inverse agonism requirement."
    )
    
    npi_apat = st.selectbox(
        "Apathy / Executive Dysfunction Severity (NPI-Q Scale)",
        options=list(NPI_MAPPING.keys()),
        index=2,
        help="Mapped to frontostriatal D2/D3 partial agonism requirement."
    )
    
    weights = {
        "5HT2A": NPI_MAPPING[npi_agit],
        "D2": NPI_MAPPING[npi_apat]
    }

with col2:
    st.subheader("Patient Clinical Vulnerabilities & Safety Profiles")
    
    fall_sel = st.selectbox(
        "Fall & Sedation Risk Assessment (Morse Scale)",
        options=list(FALL_RISK_MAPPING.keys()),
        index=2,
        help="Determines toxicity penalty weight for H1 receptor affinity."
    )
    
    ortho_sel = st.selectbox(
        "Orthostatic Hypotension Profile (Standing SBP Drop)",
        options=list(ORTHO_BP_MAPPING.keys()),
        index=1,
        help="Determines toxicity penalty weight for alpha-1 adrenergic receptor affinity."
    )
    
    park_sel = st.selectbox(
        "Parkinsonism / EPS Vulnerability (SAS / UPDRS Scale)",
        options=list(PARKINSONISM_MAPPING.keys()),
        index=0,
        help="Determines toxicity penalty weight for full D2 receptor antagonists."
    )
    
    mmse = st.number_input("Baseline MMSE Score (Cognitive Assessment)", min_value=0, max_value=30, value=15)
    
    lambda_risks = {
        "H1": FALL_RISK_MAPPING[fall_sel],
        "alpha1": ORTHO_BP_MAPPING[ortho_sel],
        "D2_full": PARKINSONISM_MAPPING[park_sel]
    }

st.markdown("---")
st.subheader("Calculated Drug Match Dashboard")

# Calculate results for candidate drugs
results = [calculate_match_score(d, data, weights, lambda_risks, mmse) for d, data in DRUG_DATABASE.items()]
df_results = pd.DataFrame(results).sort_values(by="Net Score (Mj)", ascending=False).reset_index(drop=True)

# Format numerical scores to 1 decimal place consistently
df_results["Net Score (Mj)"] = df_results["Net Score (Mj)"].map("{:.1f}".format)
df_results["Therapeutic Gain"] = df_results["Therapeutic Gain"].map("{:.1f}".format)
df_results["Risk Deductions"] = df_results["Risk Deductions"].map("{:.1f}".format)
df_results["ACB Penalty"] = df_results["ACB Penalty"].map("{:.1f}".format)

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
            &nbsp;|&nbsp; Risk Deductions: <strong>-{top_drug['Risk Deductions']}</strong>
            &nbsp;|&nbsp; ACB Penalty: <strong>-{top_drug['ACB Penalty']}</strong>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Apply Styling Matrix to Net Score
def apply_traffic_lights(val):
    val_float = float(val)
    if val_float > 1.0:
        return 'background-color: #d4edda; color: #155724; font-weight: bold;'
    elif val_float >= -2.0:
        return 'background-color: #fff3cd; color: #856404;'
    else:
        return 'background-color: #f8d7da; color: #721c24;'

# Render Interactive Table
st.dataframe(
    df_results.style.map(apply_traffic_lights, subset=['Net Score (Mj)']),
    use_container_width=True
)

st.info("**Traffic Light Guide:** Green = Optimal Match (Mj > 1.0) | Yellow = Proceed with Caution (-2.0 ≤ Mj ≤ 1.0) | Red = Flagged High Toxicity Risk (Mj < -2.0)")

# Explainable AI Component with Detailed Background Rationales
with st.expander("🔍 Background Rationale & Pharmacodynamic Calculation Details"):
    st.markdown(
        """
        ### Algorithmic Architecture & Pharmacodynamic Rationale
        
        The Net Therapeutic Match Score ($M_j$) evaluates psychotropic suitability by combining target receptor binding affinity, off-target toxicity penalties, and cognitive burden adjustments[cite: 1]:
        
        $$M_j = U_{\\text{thera}} - U_{\\text{risk}} - P_{\\text{ACB}}$$
        
        ---

        #### 1. Therapeutic Gain ($U_{\\text{thera}}$)
        * **$5\\text{-HT}_{2\\text{A}}$ Inverse Agonism / Antagonism:** Psychotic agitation in neurodegenerative illness is driven by cortical $5\\text{-HT}_{2\\text{A}}$ receptor upregulation[cite: 1]. Drugs with high $5\\text{-HT}_{2\\text{A}}$ binding potency ($pK_i$) offset serotonin hyperfunction[cite: 1].
        * **$D_2$ Receptor Modulation:** Frontostriatal dopamine depletion causes apathy and executive dysfunction[cite: 1]. Partial agonists (e.g., Brexpiprazole) stabilize dopamine transmission without causing motor block ($A_r = +1.0$)[cite: 1]. Full $D_2$ antagonists (e.g., Haloperidol) exacerbate apathy ($A_r = -1.0$)[cite: 1].

        #### 2. Risk Deductions ($U_{\\text{risk}}$)
        * **Histamine $H_1$ Blockade:** Off-target $H_1$ affinity correlates directly with central sedation, gait instability, and fall risk[cite: 1]. Scaled via patient's baseline **Morse Fall Scale** ($\lambda_{H1}$)[cite: 1].
        * **$\alpha_1$-Adrenergic Blockade:** $\alpha_1$ receptor antagonism impairs peripheral vasoconstriction, causing orthostatic hypotension and syncope[cite: 1]. Scaled via standing systolic BP drop ($\lambda_{\alpha1}$)[cite: 1].
        * **Full $D_2$ Blockade Penalty:** Applying potent full $D_2$ antagonism in patients with underlying extrapyramidal vulnerability triggers acute parkinsonism[cite: 1]. Scaled via baseline **Simpson-Angus Scale** ($\lambda_{D2\\_full}$)[cite: 1].

        #### 3. Discontinuous Cognitive Penalty ($P_{\\text{ACB}}$)
        Central $M_1$ muscarinic receptor blockade degrades cholinergic transmission essential for memory[cite: 1]. 
        * Applied as a step-function penalty when $M_1$ binding potency exceeds threshold ($pK_i \\ge 7.0$, or $K_i \\le 100\\text{ nM}$)[cite: 1].
        * Scaled dynamically according to cognitive impairment severity: $C_{\\text{patient}} = 3.0$ for severe dementia (MMSE < 10), $2.0$ for moderate (MMSE 10-20), and $1.0$ for mild/normal baseline cognitive scores[cite: 1].
        """
    )
    
    st.write("**Current Parameter Values Applied in Calculation:**")
    st.json({
        "Normalized Target Weights (w_r)": weights,
        "Normalized Risk Coefficients (lambda_r)": lambda_risks,
        "Cognitive Parameter (MMSE)": mmse
    })

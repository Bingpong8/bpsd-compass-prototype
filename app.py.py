import streamlit as st
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. EXPANDED DATABASE & PHARMACODYNAMIC DATA (pKi = -log10(K_i))
# ---------------------------------------------------------
DRUG_DATABASE = {
    "Brexpiprazole": {
        "pKi": {"5HT2A": 9.33, "D2": 9.52, "NET": 5.00, "α2a": 8.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 7.72, "α1": 8.42, "M1": 5.50},
        "Ar": {"5HT2A": 1.0, "D2": 1.0, "NET": 0.0, "α2a": 1.0, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Olanzapine": {
        "pKi": {"5HT2A": 8.40, "D2": 7.96, "NET": 5.00, "α2a": 6.20, "NMDA": 5.00, "GABA-A": 5.00, "H1": 8.15, "α1": 7.72, "M1": 7.59},
        "Ar": {"5HT2A": 1.0, "D2": 1.0, "NET": 0.0, "α2a": 0.5, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Quetiapine": {
        "pKi": {"5HT2A": 6.70, "D2": 5.80, "NET": 6.60, "α2a": 6.10, "NMDA": 5.00, "GABA-A": 5.00, "H1": 8.00, "α1": 7.00, "M1": 6.00},
        "Ar": {"5HT2A": 1.0, "D2": 1.0, "NET": 1.0, "α2a": 0.5, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Risperidone": {
        "pKi": {"5HT2A": 9.30, "D2": 8.43, "NET": 5.00, "α2a": 7.20, "NMDA": 5.00, "GABA-A": 5.00, "H1": 7.70, "α1": 8.15, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": 1.0, "NET": 0.0, "α2a": 0.5, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Haloperidol": {
        "pKi": {"5HT2A": 7.10, "D2": 8.70, "NET": 5.00, "α2a": 5.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 6.00, "α1": 7.20, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": -1.0, "NET": 0.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 0.0}  # Full D2 antagonism opposes apathy goals
    },
    "Mirtazapine": {
        "pKi": {"5HT2A": 8.10, "D2": 5.00, "NET": 5.00, "α2a": 7.80, "NMDA": 5.00, "GABA-A": 5.00, "H1": 9.00, "α1": 7.20, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "α2a": 1.0, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Memantine": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "NET": 5.00, "α2a": 5.00, "NMDA": 6.30, "GABA-A": 5.00, "H1": 5.00, "α1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2a": 0.0, "NMDA": 1.0, "GABA-A": 0.0}
    },
    "Clonidine": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "NET": 5.00, "α2a": 7.50, "NMDA": 5.00, "GABA-A": 5.00, "H1": 5.00, "α1": 6.20, "M1": 5.00},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2a": 1.0, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Lorazepam": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "NET": 5.00, "α2a": 5.00, "NMDA": 5.00, "GABA-A": 7.80, "H1": 5.00, "α1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 1.0}
    },
    "Escitalopram": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "NET": 5.00, "α2a": 5.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 5.00, "α1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 0.0}
    }
}

# ---------------------------------------------------------
# 2. CALCULATION ENGINE (Clean 1-Decimal Output)
# ---------------------------------------------------------
def calculate_match_score(drug_name, drug_data, weights, lambda_risks, mmse_score):
    pk = drug_data["pKi"]
    ar = drug_data["Ar"]
    
    # 1. Multi-Neurotransmitter Therapeutic Utility Sum: sum(ω_r * pKi * A_r)
    u_thera = (weights["5HT2A"] * pk["5HT2A"] * ar["5HT2A"]) + \
              (weights["D2"] * pk["D2"] * ar["D2"]) + \
              (weights["NET"] * pk["NET"] * ar["NET"]) + \
              (weights["α2a"] * pk["α2a"] * ar["α2a"]) + \
              (weights["NMDA"] * pk["NMDA"] * ar["NMDA"]) + \
              (weights["GABA-A"] * pk["GABA-A"] * ar["GABA-A"])
    
    # 2. Off-Target Risk Deductions
    d2_risk = (lambda_risks["D2_full"] * pk["D2"]) if ar["D2"] < 0 else 0.0
    u_risk = (lambda_risks["H1"] * pk["H1"]) + \
             (lambda_risks["α1"] * pk["α1"]) + \
             d2_risk
    
    # 3. Discontinuous Anticholinergic Cognitive Burden Penalty (PACB)
    if mmse_score < 10:
        c_patient = 3.0
    elif mmse_score <= 20:
        c_patient = 2.0
    else:
        c_patient = 1.0
        
    pacb = c_patient * 2.0 if pk["M1"] >= 7.0 else 0.0
    
    # Net Therapeutic Score
    m_j = u_thera - u_risk - pacb
    
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
st.set_page_config(page_title="BPSD Compass - Prototype", layout="wide")

st.title("🧠 Multi-Neurotransmitter BPSD Prescribing Compass")
st.caption("Parameter-driven clinical matching algorithm based on Neurotransmitter-Receptor Affinities")

st.markdown("---")

# Standardized Clinical Anchor Mappings (Translates Bedside Evaluations to Weights)
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
    
    npi_agit = st.selectbox(
        "Psychotic Agitation / Hallucinations (5-HT2A Target)",
        options=list(NPI_MAPPING.keys()),
        index=3,
        help="Maps to cortical 5-HT2A inverse agonism requirement."
    )
    
    npi_apat = st.selectbox(
        "Apathy / Executive Dysfunction (D2 Target)",
        options=list(NPI_MAPPING.keys()),
        index=1,
        help="Maps to frontostriatal D2 partial agonism requirement."
    )
    
    npi_arousal = st.selectbox(
        "Hyperadrenergic Agitation / Autonomic Storm (Alpha-2A Target)",
        options=list(NPI_MAPPING.keys()),
        index=2,
        help="Maps to central presynaptic alpha-2A autoreceptor agonism."
    )
    
    npi_anx = st.selectbox(
        "Acute Panic / Severe Hyperarousal (GABA-A Target)",
        options=list(NPI_MAPPING.keys()),
        index=0,
        help="Maps to central GABA-A receptor positive allosteric modulation."
    )
    
    npi_excit = st.selectbox(
        "Excitotoxic Agitation / Delirious Psychosis (NMDA Target)",
        options=list(NPI_MAPPING.keys()),
        index=0,
        help="Maps to uncompetitive NMDA receptor antagonism."
    )
    
    weights = {
        "5HT2A": NPI_MAPPING[npi_agit],
        "D2": NPI_MAPPING[npi_apat],
        "NET": NPI_MAPPING[npi_apat] * 0.5,  # NET inhibition co-targeted for frontostriatal deficit
        "α2a": NPI_MAPPING[npi_arousal],
        "NMDA": NPI_MAPPING[npi_excit],
        "GABA-A": NPI_MAPPING[npi_anx]
    }

with col2:
    st.subheader("Patient Risk Profile & Vulnerabilities")
    
    fall_sel = st.selectbox(
        "Fall & Sedation Risk Assessment (Morse Fall Scale)",
        options=list(FALL_RISK_MAPPING.keys()),
        index=2,
        help="Determines toxicity penalty weight for H1 histamine receptor blockade."
    )
    
    ortho_sel = st.selectbox(
        "Orthostatic Hypotension Profile (Standing SBP Drop)",
        options=list(ORTHO_BP_MAPPING.keys()),
        index=1,
        help="Determines toxicity penalty weight for alpha-1 adrenergic blockade."
    )
    
    park_sel = st.selectbox(
        "Parkinsonism / EPS Vulnerability (SAS / UPDRS Scale)",
        options=list(PARKINSONISM_MAPPING.keys()),
        index=0,
        help="Determines toxicity penalty weight for full D2 receptor antagonists."
    )
    
    mmse = st.number_input(
        "Baseline MMSE Score (Cognitive Assessment)",
        min_value=0,
        max_value=30,
        value=15,
        help="Determines scaling factor for discontinuous anticholinergic cognitive penalty."
    )
    
    lambda_risks = {
        "H1": FALL_RISK_MAPPING[fall_sel],
        "α1": ORTHO_BP_MAPPING[ortho_sel],
        "D2_full": PARKINSONISM_MAPPING[park_sel]
    }

st.markdown("---")
st.subheader("Calculated Multi-System Match Dashboard")

# Calculate results for all candidate drugs
results = [calculate_match_score(d, data, weights, lambda_risks, mmse) for d, data in DRUG_DATABASE.items()]
df_results = pd.DataFrame(results).sort_values(by="Net Score (Mj)", ascending=False).reset_index(drop=True)

# Strictly format values to 1 decimal place across all score columns
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
            &nbsp;|&nbsp; Gain: <strong>+{top_drug['Therapeutic Gain']}</strong> 
            &nbsp;|&nbsp; Deductions: <strong>-{top_drug['Risk Deductions']}</strong>
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

st.info("**Traffic Light Guide:** Green = Optimal Match (Mj > 1.0) | Yellow = Proceed with Caution (-2.0 ≤ Mj ≤ 1.0) | Red = High Risk Flag (Mj < -2.0)")

# Explainable Component with Extended Multi-System Rationale
with st.expander("🔍 Background Rationale & Expanded Pharmacodynamic Details"):
    st.markdown(
        """
        ### Multi-Neurotransmitter Algorithmic Architecture
        
        The Net Therapeutic Match Score ($M_j$) evaluates psychotropic suitability across 6 distinct neurochemical systems by balancing symptom-weighted target affinities against off-target safety penalties[cite: 1]:
        
        $$M_j = U_{\\text{thera}} - U_{\\text{risk}} - P_{\\text{ACB}}$$
        
        ---

        #### 1. Multi-Target Therapeutic Gain ($U_{\\text{thera}}$)
        The total therapeutic utility is calculated across six neurotransmitter pathways:
        
        $$U_{\\text{thera}} = \\sum_{r \\in \\{5HT2A, D2, NET, α2a, NMDA, GABA-A\\}} \\left( ω_r \\cdot pK_{i,r} \\cdot A_r \\right)$$
        
        * **$5\\text{-HT}_{2\\text{A}}$ Serotonergic Target:** $5\\text{-HT}_{2\\text{A}}$ inverse agonism ($A_r = +1.0$) attenuates cortical serotonergic hyperfunction driving psychotic agitation, visual hallucinations, and paranoia[cite: 1].
        * **$D_2$ Dopaminergic & $\\text{NET}$ Targets:** Frontostriatal dopamine and norepinephrine depletion drive apathy, motor slowing, and executive dysfunction[cite: 1]. Partial $D_2$ agonists (e.g., Brexpiprazole) stabilize dopamine neurotransmission ($A_r = +1.0$) without triggering motor block[cite: 1].
        * **$\\alpha_{2\\text{A}}$ Noradrenergic Target:** Central $\\alpha_{2\\text{A}}$ presynaptic agonism ($A_r = +1.0$) reduces hyperadrenergic outflow, calming autonomic arousal, tachycardia, and noradrenergic agitation[cite: 1].
        * **$\\text{GABA}_{A}$ Benzodiazepine Target:** $\\text{GABA}_{A}$ positive allosteric modulation ($A_r = +1.0$) enhances inhibitory neurotransmission to rapidly alleviate acute hyperarousal and panic[cite: 1].
        * **$\\text{NMDA}$ Glutamatergic Target:** Uncompetitive $\\text{NMDA}$ receptor antagonism ($A_r = +1.0$) protects against excess glutamatergic excitotoxicity and delirium.

        ---

        #### 2. Risk Deductions ($U_{\\text{risk}}$)
        Off-target risk penalties scale dynamically based on patient frailty and baseline clinical scores:
        
        * **Histamine $H_1$ Blockade:** Off-target $H_1$ affinity induces severe central sedation and gait ataxia. Penalty weight ($\lambda_{H1}$) is scaled using the bedside **Morse Fall Scale**[cite: 1].
        * **$\\alpha_1$-Adrenergic Blockade:** $\\alpha_1$ antagonism inhibits vascular vasoconstriction, causing postural hypotension and syncope. Penalty weight ($\lambda_{α1}$) is scaled using standing systolic blood pressure drop[cite: 1].
        * **Full $D_2$ Antagonism Penalty:** Potent full $D_2$ antagonists (e.g., Haloperidol, $A_{D2} = -1.0$) block extrapyramidal motor pathways, worsening apathy and inducing severe parkinsonism[cite: 1]. Scaled via baseline **Simpson-Angus Scale (SAS)** ($\lambda_{D2\\_full}$)[cite: 1].

        ---

        #### 3. Discontinuous Anticholinergic Penalty ($P_{\\text{ACB}}$)
        Central $M_1$ muscarinic receptor blockade impairs memory encoding and precipitates acute delirium[cite: 1].
        * **Threshold Penalty:** Applied as a step function when $M_1$ binding potency reaches or exceeds $pK_{i,M1} \\ge 7.0$ ($K_i \\le 100\\text{ nM}$).
        * **Cognitive Scaling:** Penalty magnitude is scaled dynamically using baseline MMSE score: $C_{\\text{patient}} = 3.0$ for severe dementia (MMSE < 10), $2.0$ for moderate impairment (MMSE 10-20), and $1.0$ for mild/normal baseline cognition[cite: 1].
        """
    )
    
    st.write("**Current Parameter Values Applied in Calculation:**")
    st.json({
        "Normalized Target Weights (ω_r)": weights,
        "Normalized Risk Coefficients (lambda_r)": lambda_risks,
        "Baseline Cognitive Score (MMSE)": mmse
    })

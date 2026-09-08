import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="BPSD Compass Prototype (P2)", layout="wide")
st.title("BPSD Compass Prototype (V2)")
st.caption("Parameter-driven neurotransmitters affinity based decision-support tool")

ascii_header = r"""
								THE DEATH OF PEACE OF MIND
													
								      _---~~(~~-_.
								    _{        )   )
								  ,   ) -~~- ( ,-' )_
								 (   `-,\_..`., )-- '_,)
								( `\_)  (  -\~( -\_`,  }
								(_-  _  ~_-~~~~`,  ,' )
								  `~ -^(    __;-,((()))
								        ~~~~ {_ -_(())
								               `\  }
								                 { }
																	 
							     Dolor et Astra, Nihil est Veritas
"""

st.code(ascii_header, language=None)

# 1. PHARMACODYNAMIC DATABASE
# ---------------------------------------------------------
DRUG_DATABASE = {
    "Brexpiprazole": {
        "pKi": {"5HT2A": 9.33, "D2": 9.52, "NET": 5.00, "α2a": 8.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 7.72, "α1": 8.42, "M1": 5.50},
        "Ar": {"5HT2A": 1.0, "D2": 1.0, "NET": 0.0, "α2a": 1.0, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Aripiprazole": {
        "pKi": {"5HT2A": 8.40, "D2": 8.80, "NET": 5.00, "α2a": 6.80, "NMDA": 5.00, "GABA-A": 5.00, "H1": 7.50, "α1": 7.30, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": 1.0, "NET": 0.0, "α2a": 0.5, "NMDA": 0.0, "GABA-A": 0.0}
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
        "Ar": {"5HT2A": 1.0, "D2": -1.0, "NET": 0.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 0.0}
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
    },
    "Sertraline": {
        "pKi": {"5HT2A": 5.00, "D2": 6.60, "NET": 5.00, "α2a": 5.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 5.00, "α1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 0.0, "D2": 0.5, "NET": 0.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Fluoxetine": {
        "pKi": {"5HT2A": 6.70, "D2": 5.00, "NET": 5.00, "α2a": 5.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 5.00, "α1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 0.5, "D2": 0.0, "NET": 0.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Nortriptyline": {
        "pKi": {"5HT2A": 7.50, "D2": 5.00, "NET": 7.90, "α2a": 6.30, "NMDA": 5.00, "GABA-A": 5.00, "H1": 8.00, "α1": 7.20, "M1": 6.70},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 1.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Amitriptyline": {
        "pKi": {"5HT2A": 7.80, "D2": 5.00, "NET": 7.40, "α2a": 6.10, "NMDA": 5.00, "GABA-A": 5.00, "H1": 8.90, "α1": 7.60, "M1": 8.00},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 1.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Venlafaxine": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "NET": 5.60, "α2a": 5.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 5.00, "α1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 1.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Duloxetine": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "NET": 7.70, "α2a": 5.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 5.00, "α1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 1.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Vortioxetine": {
        "pKi": {"5HT2A": 7.60, "D2": 5.00, "NET": 5.00, "α2a": 5.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 5.00, "α1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Bupropion": {
        "pKi": {"5HT2A": 5.00, "D2": 5.20, "NET": 5.20, "α2a": 5.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 5.00, "α1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 0.0, "D2": 0.5, "NET": 1.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Trazodone": {
        "pKi": {"5HT2A": 7.80, "D2": 5.00, "NET": 5.00, "α2a": 6.40, "NMDA": 5.00, "GABA-A": 5.00, "H1": 7.50, "α1": 7.80, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 0.0}
    },
    "Agomelatine": {
        "pKi": {"5HT2A": 6.20, "D2": 5.00, "NET": 5.00, "α2a": 5.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 5.00, "α1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "α2a": 0.0, "NMDA": 0.0, "GABA-A": 0.0}
    }
}

# 2. CAUTIONS DATABASE
# ---------------------------------------------------------
BLACK_BOX_WARNINGS = {
    "Brexpiprazole": "Exercise extreme caution for akathisia and impulse-control disorders.",
    "Aripiprazole": "High clinical risk of akathisia, restlessness, and compulsive behavior.",
    "Olanzapine": "High risk of severe metabolic syndrome, rapid weight gain, profound sedation, and anticholinergic cognitive impairment (M1 pKi ≥ 7.0).",
    "Quetiapine": "Risk of severe orthostatic hypotension, somnolence, and metabolic dysregulation.",
    "Risperidone": "Dose-dependent extrapyramidal symptoms and hyperprolactinemia with elevated risk of cerebrovascular adverse events (stroke/TIA).",
    "Haloperidol": "High risk of severe Extrapyramidal Symptoms (EPS), Tardive Dyskinesia, and QTc prolongation / Torsades de Pointes.",
    "Mirtazapine": "Increased risk of suicidal ideation in young adults. Causes marked somnolence, appetite stimulation, and potential agranulocytosis.",
    "Memantine": "Requires dose adjustment in severe renal impairment (CrCl < 30 mL/min). May cause mild dizziness, confusion, and headache.",
    "Clonidine": "High risk of severe rebound hypertension upon abrupt withdrawal. Causes sinus bradycardia, orthostatic hypotension, and central sedation.",
    "Lorazepam": "Concomitant use with opioids may result in severe sedation, respiratory depression, coma, and death. High risk of physical dependence, ataxia, paradoxical disinhibition in dementia, and falls.",
    "Escitalopram": "Dose-dependent QTc prolongation (maximum recommended dose 10 mg/day in elderly patients). Risk of hyponatremia / SIADH.",
    "Sertraline": "Increased risk of suicidal ideation; caution for severe hyponatremia/SIADH, serotonin syndrome and bleeding risk.",
    "Fluoxetine": "Long half-life with high risk of drug interactions (CYP2D6/3A4 inhibitor), and serotonin syndrome.",
    "Amitriptyline": "High risk of overdose, cardiac arrhythmias, anticholinergic toxicity.",
    "Venlafaxine": "Dose-dependent sustained hypertension, QTc prolongation, severe discontinuation syndrome.",
    "Duloxetine": "Contraindicated in Severe hepatic impairment or Chronic liver disease.",
    "Vortioxetine": "Abnormal bleeding, Hyponatremia/SIADH, and Serotonin syndrome.",
    "Bupropion": "Contraindicated in patients with seizure disorders, active eating disorders (bulimia/anorexia), or abrupt cessation of GABAergic agents or alcohol.",
    "Trazodone": "High risk of oversedation due to high α1 blockade, QTc prolongation, and rare risk of priapism.",
    "Agomelatine": "Contraindicated in hepatic impairment due to hepatotoxic profile."
}

# 3. CALCULATION MODEL
# ---------------------------------------------------------
def calculate_match_score(drug_name, drug_data, weights, lambda_risks, TMSE_score):
    pk = drug_data["pKi"]
    ar = drug_data["Ar"]
    
    # 1. Therapeutic Component
    u_thera = (weights.get("5HT2A", 0) * pk["5HT2A"] * ar["5HT2A"]) + \
              (weights.get("D2", 0) * pk["D2"] * ar["D2"]) + \
              (weights.get("NET", 0) * pk["NET"] * ar["NET"]) + \
              (weights.get("α2a", 0) * pk["α2a"] * ar["α2a"]) + \
              (weights.get("NMDA", 0) * pk["NMDA"] * ar["NMDA"]) + \
              (weights.get("GABA-A", 0) * pk["GABA-A"] * ar["GABA-A"])
    
    # 2. Risk Component
    d2_risk = (lambda_risks["D2_full"] * pk["D2"]) if ar["D2"] < 0 else 0.0
    u_risk = (lambda_risks["H1"] * pk["H1"]) + \
             (lambda_risks["α1"] * pk["α1"]) + \
             d2_risk
    
    # 3. Discontinuous Anticholinergic Cognitive Burden Penalty (PACB)
    if TMSE_score < 10:
        c_patient = 3.0
    elif TMSE_score <= 20:
        c_patient = 2.0
    else:
        c_patient = 1.0
        
    pacb = c_patient * 2.0 if pk["M1"] >= 7.0 else 0.0
    
    # Score Calculation
    m_j = u_thera - u_risk - pacb
    
    return {
        "Drug": drug_name,
        "Net Score (Mj)": round(m_j, 1),
        "Therapeutic Gain": round(u_thera, 1),
        "Risk Deductions": round(u_risk, 1),
        "ACB Penalty": round(pacb, 1),
        "M1 Potency": round(pk["M1"], 1)
    }


# 4. WEB-APP FRONTEND & UI
# ---------------------------------------------------------

st.markdown("---")

if "excluded_drugs" not in st.session_state:
    st.session_state.excluded_drugs = []

NPI_MAPPING = {
    "0 - Absent (No symptoms)": 0.0,
    "1 - Mild (Slight distress, no functional impairment)": 0.33,
    "2 - Moderate (Significant distress, partial impairment)": 0.67,
    "3 - Severe (Major disruption, marked impairment)": 1.00
}

FALL_RISK_MAPPING = {
    "Low Risk (Morse Fall Score 0-24)": 0.1,
    "Moderate Risk (Morse Fall Score 25-44)": 0.5,
    "High Risk (Morse Fall Score ≥ 45 or fall history)": 0.9
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

# PRIOR PSYCHOTROPIC MEDICATION EVALUATION
st.subheader("📋 Prior Psychotropic Medication Status")
st.caption("Assess ongoing/recent psychotropic regimens before new target optimization.")

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    prior_drug = st.selectbox(
        "Current / Prior Psychotropic Medication",
        options=["None (Treatment Naïve)"] + list(DRUG_DATABASE.keys()),
        index=0
    )

with col_p2:
    prior_response = st.selectbox(
        "Response to Current Regimen",
        options=[
            "N/A (Naïve)",
            "Adequate Response (Symptom Control)",
            "Partial Response (Subtherapeutic)",
            "No Response (Refractory)",
            "Intolerable Adverse Effects"
        ],
        index=0
    )

with col_p3:
    prior_dose_status = st.selectbox(
        "Current Dosage Level",
        options=["N/A", "Low / Starting Dose", "Moderate / Target Dose", "Maximum Tolerated Dose"],
        index=0
    )

st.markdown("---")

# 11-NPI SYMPTOM DOMAIN ASSESSMENT
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Target Symptom Severity (NPI Domains)")
    st.caption("💡 *Normalized Weight (ωr) = Bedside Score / Maximum Score (0.0 to 1.0)*")
    
    npi_delusions = st.selectbox("1. Delusions (5-HT2A / D2 Target)", options=list(NPI_MAPPING.keys()), index=0)
    npi_hallucinations = st.selectbox("2. Hallucinations (5-HT2A Target)", options=list(NPI_MAPPING.keys()), index=0)
    npi_agitation = st.selectbox("3. Agitation / Aggression (5-HT2A / α2a Target)", options=list(NPI_MAPPING.keys()), index=3)
    npi_depression = st.selectbox("4. Depression / Dysphoria (SERT / 5-HT1A Target)", options=list(NPI_MAPPING.keys()), index=2)
    npi_anxiety = st.selectbox("5. Anxiety / Panic (GABA-A / 5-HT1A Target)", options=list(NPI_MAPPING.keys()), index=0)
    npi_elation = st.selectbox("6. Elation / Euphoria (D2 Modulation Target)", options=list(NPI_MAPPING.keys()), index=0)
    npi_apathy = st.selectbox("7. Apathy / Indifference (D2 / NET Target)", options=list(NPI_MAPPING.keys()), index=1)
    npi_disinhibition = st.selectbox("8. Disinhibition (D2 / 5-HT2A Target)", options=list(NPI_MAPPING.keys()), index=0)
    npi_irritability = st.selectbox("9. Irritability / Lability (α2a / 5-HT2A Target)", options=list(NPI_MAPPING.keys()), index=2)
    npi_motor = st.selectbox("10. Aberrant Motor Behavior (D2 / 5-HT2A Target)", options=list(NPI_MAPPING.keys()), index=0)
    npi_sleep = st.selectbox("11. Sleep / Nighttime Behavior (H1 / α2a Target)", options=list(NPI_MAPPING.keys()), index=0)

# Derive receptor weights dynamically from 11 NPI domains
weights = {
    "5HT2A": max(NPI_MAPPING[npi_agitation], NPI_MAPPING[npi_delusions], NPI_MAPPING[npi_hallucinations], NPI_MAPPING[npi_irritability] * 0.7),
    "D2": max(NPI_MAPPING[npi_apathy], NPI_MAPPING[npi_delusions] * 0.8, NPI_MAPPING[npi_elation] * 0.8, NPI_MAPPING[npi_disinhibition] * 0.6),
    "NET": NPI_MAPPING[npi_apathy] * 0.5,
    "α2a": max(NPI_MAPPING[npi_agitation] * 0.8, NPI_MAPPING[npi_irritability], NPI_MAPPING[npi_sleep] * 0.5),
    "NMDA": NPI_MAPPING[npi_agitation] * 0.3,  # Delirious/excitotoxic component
    "GABA-A": NPI_MAPPING[npi_anxiety],
    "SERT": NPI_MAPPING[npi_depression],
    "5HT1A": max(NPI_MAPPING[npi_depression] * 0.8, NPI_MAPPING[npi_anxiety] * 0.7)
}

with col2:
    st.subheader("Patient Risk & Safety Profile")
    st.caption("💡 *Risk Coefficients (λr) of frailty matched to toxicity*")
    
    fall_sel = st.selectbox("Fall & Sedation Vulnerability (Morse Fall Scale)", options=list(FALL_RISK_MAPPING.keys()), index=2)
    ortho_sel = st.selectbox("Orthostatic Hypotension Profile (Standing SBP Drop)", options=list(ORTHO_BP_MAPPING.keys()), index=1)
    park_sel = st.selectbox("Parkinsonism / EPS Vulnerability (SAS / UPDRS Scale)", options=list(PARKINSONISM_MAPPING.keys()), index=0)
    TMSE = st.number_input("Baseline TMSE Score (Cognitive Assessment)", min_value=0, max_value=30, value=15)
    
    lambda_risks = {
        "H1": FALL_RISK_MAPPING[fall_sel],
        "α1": ORTHO_BP_MAPPING[ortho_sel],
        "D2_full": PARKINSONISM_MAPPING[park_sel]
    }

st.markdown("---")
st.subheader("Dashboard")

raw_results = [calculate_match_score(d, data, weights, lambda_risks, TMSE) for d, data in DRUG_DATABASE.items()]
df_results = pd.DataFrame(raw_results).sort_values(by="Net Score (Mj)", ascending=False).reset_index(drop=True)
df_filtered = df_results[~df_results["Drug"].isin(st.session_state.excluded_drugs)].reset_index(drop=True)

for col in ["Net Score (Mj)", "Therapeutic Gain", "Risk Deductions", "ACB Penalty", "M1 Potency"]:
    df_results[col] = df_results[col].map("{:.1f}".format)
    df_filtered[col] = df_filtered[col].map("{:.1f}".format)

if not df_filtered.empty:
    top_drug = df_filtered.iloc[0]
    top_name = top_drug['Drug']
    
    st.markdown(
        f"""
        <div style="background-color: #d1e7dd; border-left: 8px solid #0f5132; padding: 18px; border-radius: 6px; margin-bottom: 15px;">
            <span style="font-size: 14px; color: #0f5132; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;">Top Recommended Option</span>
            <h1 style="color: #0f5132; margin: 4px 0 0 0; font-size: 32px; font-weight: 800;">
                🏆 {top_name}
            </h1>
            <p style="color: #0f5132; font-size: 18px; margin: 6px 0 0 0;">
                Net Match Score (Mj): <strong>{top_drug['Net Score (Mj)']}</strong> 
                &nbsp;|&nbsp; Gain: <strong>+{top_drug['Therapeutic Gain']}</strong> 
                &nbsp;|&nbsp; Deductions: <strong>-{top_drug['Risk Deductions']}</strong>
                &nbsp;|&nbsp; ACB Penalty: <strong>-{top_drug['ACB Penalty']}</strong>
                &nbsp;|&nbsp; M1 Potency: <strong>{top_drug['M1 Potency']}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # NEXT-STEP MANAGEMENT DECISION ENGINE
    if prior_drug != "None (Treatment Naïve)":
        st.markdown("### 🔀 Prior Medication Management Strategy")
        if prior_drug == top_name:
            if prior_response == "Partial Response (Subtherapeutic)" and prior_dose_status != "Maximum Tolerated Dose":
                st.info(f"<b>Management Strategy: Dose Optimization.</b> {top_name} remains the optimal algorithmic choice. Continue {top_name} and titrate upward to target dose before switching.")
            elif prior_response == "Intolerable Adverse Effects":
                st.warning(f"<b>Management Strategy: Cross-Titration Indicated.</b> Although {top_name} ranks highest, but the patient experienced intolerable side effects. Consider checking the rule-out box for {top_name} to evaluate second-line alternatives.")
            else:
                st.info(f"<b>Management Strategy: Maintain Current Regimen.</b> {top_name} matches the target profile and is currently active.")
        else:
            if prior_response in ["Partial Response (Subtherapeutic)", "No Response (Refractory)", "Intolerable Adverse Effects"]:
                st.success(f"<b>Management Strategy: Cross-Titration Recommended.</b> Transition from <b>{prior_drug}</b> to <b>{top_name}</b> via slow cross-titration over 2–4 weeks to prevent receptor rebound or withdrawal.")
            elif prior_response == "Adequate Response (Symptom Control)":
                st.info(f"<b>Management Strategy: Maintain Current Regimen.</b> Patient has adequate response on <b>{prior_drug}</b>. Switching to {top_name} is not immediately required unless safety concerns arise.")

    st.warning(f"⚠️ **Clinical Cautions for {top_name}:**\n\n{BLACK_BOX_WARNINGS.get(top_name, 'No specific black box warning listed.')}")
    
    rule_out_check = st.checkbox(
        f"🚫 **Rule out {top_name} for this patient** (Check if patient has contraindications, high risks for exact medication, intolerance or allergy)",
        key=f"ruleout_{top_name}"
    )
    
    if rule_out_check:
        st.session_state.excluded_drugs.append(top_name)
        st.rerun()

else:
    st.error("All candidate medications have been ruled out. Please reset the rule-out filters.")

if st.session_state.excluded_drugs:
    st.markdown(" ")
    if st.button(f"🔄 Reset Ruled-Out Medications ({len(st.session_state.excluded_drugs)} Currently Excluded)"):
        st.session_state.excluded_drugs = []
        st.rerun()

st.markdown("### Comparative Drugs Table")

def apply_traffic_lights(val):
    val_float = float(val)
    if val_float > 1.0:
        return 'background-color: #d4edda; color: #155724; font-weight: bold;'
    elif val_float >= -2.0:
        return 'background-color: #fff3cd; color: #856404;'
    else:
        return 'background-color: #f8d7da; color: #721c24;'

df_results_display = df_results.copy()
df_results_display["Status"] = df_results_display["Drug"].apply(
    lambda x: "❌ Ruled Out" if x in st.session_state.excluded_drugs else "✅ Candidate"
)

st.dataframe(
    df_results_display.style.map(apply_traffic_lights, subset=['Net Score (Mj)']),
    use_container_width=True
)

st.info("**Traffic Light Guide:** Green = Optimal Match (Mj > 1.0) | Yellow = Proceed with Caution (-2.0 ≤ Mj ≤ 1.0) | Red = High Risk Flag (Mj < -2.0)")

st.markdown("---")
with st.expander("🔍 Background Rationale & Expanded Pharmacodynamic Details"):
    st.markdown(
        """
        ### Multi-Neurotransmitter Algorithmic Architecture
        
        The Net Therapeutic Match Score ($M_j$) evaluates psychotropic suitability across distinct neurochemical systems:
        
        $$M_j = U_{\\text{thera}} - U_{\\text{risk}} - P_{\\text{ACB}}$$
        
        #### 11-NPI Neuroreceptor Mapping
        * **Delusions / Hallucinations:** Mapped to $5\\text{-HT}_{2\\text{A}}$ inverse agonism and $D_2$ antagonism.
        * **Agitation / Aggression:** Mapped to $5\\text{-HT}_{2\\text{A}}$ blockade and $\\alpha_{2\\text{A}}$ autoreceptor agonism.
        * **Depression / Dysphoria:** Mapped to $\\text{SERT}$ inhibition and $5\\text{-HT}_{1\\text{A}}$ partial agonism.
        * **Apathy:** Mapped to frontostriatal $D_2$ partial agonism and $\\text{NET}$ inhibition.
        * **Sleep Disturbances:** Mapped to central $H_1$ blockade and $\\alpha_{2\\text{A}}$ modulation.
        """
    )

with st.expander("🔍 Core References & Algorithmic Citations"):
    st.markdown(
        """
        1. **Roth B. L., et al.** *PDSP Ki Database. Psychoactive Drug Screening Program (PDSP)*. UNC Chapel Hill / NIMH.
        2. **Magierski R., et al. (2020).** *Pharmacotherapy of Behavioral and Psychological Symptoms of Dementia: State of the Art and Future Progress*. Front. Psychiatry. PMID: 32848775.
        3. **Tampi R. R., et al. (2022).** *Brexpiprazole for the Treatment of Agitation in Dementia*. Drugs Aging. PMID: 35904712.
        4. **Lee D., et al. (2023).** *Brexpiprazole for the Treatment of Agitation Associated with Dementia Due to Alzheimer's Disease*. Am J Psychiatry. PMID: 37143168.
        5. **Davies S. J., et al. (2018).** *Sequential drug treatment algorithm for agitation and aggression in Alzheimer's and mixed dementia*. J Psychopharmacol. PMID: 29338602.
        6. **Kales H. C., et al. (2015).** *Assessment and management of behavioral and psychological symptoms of dementia*. BMJ. PMID: 25731898.
        7. **Cummings J., et al. (2022).** *Alzheimer's disease drug development pipeline: 2022*. Alzheimers Dement (NY). PMID: 35510134.
        8. **CCSMH (2024–2025).** *Canadian Clinical Practice Guidelines for Assessing and Managing BPSD*. ccsmh.ca.
        """
    )

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="BPSD Compass Prototype", layout="wide")
st.title("BPSD Compass Prototype")
st.caption("Parameter-driven neurotransmitters affinity based decision-support tool ")

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

# 1. PHARMACODYNAMIC DATABASE (pKi = -log10(K_i))
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


# 2. CLINICAL CAUTIONS DATABASE
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
	"Sertaline": "Increased risk of suicidal ideation; caution for severe hyponatramia/SIADH, serotonin syndrome and bleeding risk.",
	"Fluoxetine": "Long half-life with high risk of drug interactions (CYP2D6/3A4 inhibitor), and serotonin syndrome.",
	"Amitryptyline": "High risk of overdose, cardiac arrhythmias, anticholinergic toxicity.",
	"Venlafaxine": "Dose-dependent sustained hypertension, QTc prolongation, severe discontinuation syndrome.",
	"Duloxetine": "Contraindicated in Severe hepatic impairment or Chronic liver disease.",
	"Vortioxetine": "Abnormal bleeding, Hyponatremia/SIADH, and Serotonin syndrome.",
	"Bupropion": "Contraindicated in patients with seizure disorders, active eating disorders (bulimia/anorexia), or abrupt cessation of GABAnergic agents or alcohol.",
	"Trazodone": "High risk of Oversedation due to high α1 blockade), QTc prologation and rare risk of Priapism.",
	"Agomelatine": "Contraindicated in Hepatic impairment due to Hepatotoxic profile."
}


# 3. CALCULATION ENGINE
# ---------------------------------------------------------
def calculate_match_score(drug_name, drug_data, weights, lambda_risks, TMSE_score):
    pk = drug_data["pKi"]
    ar = drug_data["Ar"]
    
    # 1. Therapeutic Component
    u_thera = (weights["5HT2A"] * pk["5HT2A"] * ar["5HT2A"]) + \
              (weights["D2"] * pk["D2"] * ar["D2"]) + \
              (weights["NET"] * pk["NET"] * ar["NET"]) + \
              (weights["α2a"] * pk["α2a"] * ar["α2a"]) + \
              (weights["NMDA"] * pk["NMDA"] * ar["NMDA"]) + \
              (weights["GABA-A"] * pk["GABA-A"] * ar["GABA-A"])
    
    # 2. Risk Deductions
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
    
    # Net Score Calculation
    m_j = u_thera - u_risk - pacb
    
    return {
        "Drug": drug_name,
        "Net Score (Mj)": round(m_j, 1),
        "Therapeutic Gain": round(u_thera, 1),
        "Risk Deductions": round(u_risk, 1),
        "ACB Penalty": round(pacb, 1),
        "M1 Potency (pKi)": round(pk["M1"], 1)
    }


# 4. STREAMLIT FRONTEND & UI
# ---------------------------------------------------------

st.markdown("---")

# Session State for Rule-Out Exclusions
if "excluded_drugs" not in st.session_state:
    st.session_state.excluded_drugs = []

# Standardized Bedside Rating Scale Mappings
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

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Target Symptom Severity (Bedside Anchor Scales)")
    st.caption("💡 *Normalized Weight (ωr) = Bedside Score / Maximum Score (0.0 to 1.0)*")
    
    npi_agit = st.selectbox(
        "Psychotic Agitation / Aggression (5-HT2A Target)",
        options=list(NPI_MAPPING.keys()),
        index=3,
        help="Maps to cortical 5-HT2A inverse agonism requirement."
	)
    
    npi_depr = st.selectbox(
        "Depression / Mood Lability / Dysphoria (SERT & 5-HT1A Target)",
        options=list(NPI_MAPPING.keys()), index=2,
        help="Rationale: Depressive lability in dementia benefits from SERT inhibition and 5-HT1A partial agonism."
	)
    
    npi_cog = st.selectbox(
        "Cognitive Decline / Memory Deficits (NMDA & Cholinergic Target)",
        options=list(NPI_MAPPING.keys()), index=2,
        help="Rationale: Excitotoxic cognitive decline warrants NMDA uncompetitive antagonism and avoidance of M1 anticholinergic block."
    )
    
    npi_apat = st.selectbox(
        "Apathy / Executive Dysfunction (D2 Target)",
        options=list(NPI_MAPPING.keys()),
        index=1,
        help="Maps to frontostriatal D2 partial agonism requirement."
    )
    
    npi_arousal = st.selectbox(
        "Hyperadrenergic Agitation / Autonomic Arousal (Alpha-2A Target)",
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
        "NET": NPI_MAPPING[npi_apat] * 0.5,
        "α2a": NPI_MAPPING[npi_arousal],
        "NMDA": NPI_MAPPING[npi_cog],
        "GABA-A": NPI_MAPPING[npi_anx],
        "SERT": NPI_MAPPING[npi_depr],
        "5HT1A": NPI_MAPPING[npi_depr] * 0.8
    }

with col2:
    st.subheader("Patient Risk Profile & Safety Vulnerabilities")
    st.caption("💡 *Risk Coefficients (λr) map clinical frailty scores directly to toxicity penalties*")
    
    fall_sel = st.selectbox(
        "Fall & Sedation Vulnerability (Morse Fall Scale)",
        options=list(FALL_RISK_MAPPING.keys()),
        index=2,
        help="Determines penalty weight for H1 histamine receptor blockade."
    )
    
    ortho_sel = st.selectbox(
        "Orthostatic Hypotension Profile (Standing SBP Drop)",
        options=list(ORTHO_BP_MAPPING.keys()),
        index=1,
        help="Determines penalty weight for alpha-1 adrenergic blockade."
    )
    
    park_sel = st.selectbox(
        "Parkinsonism / EPS Vulnerability (SAS / UPDRS Scale)",
        options=list(PARKINSONISM_MAPPING.keys()),
        index=0,
        help="Determines penalty weight for full D2 receptor antagonists."
    )
    
    TMSE = st.number_input(
        "Baseline TMSE Score (Cognitive Assessment)",
        min_value=0,
        max_value=30,
        value=15,
        help="Determines scaling factor (C_patient) for anticholinergic burden penalty."
    )
    
    lambda_risks = {
        "H1": FALL_RISK_MAPPING[fall_sel],
        "α1": ORTHO_BP_MAPPING[ortho_sel],
        "D2_full": PARKINSONISM_MAPPING[park_sel]
    }

st.markdown("---")
st.subheader("Calculated Multi-System Match Dashboard")

# Calculate results for all candidate drugs
raw_results = [calculate_match_score(d, data, weights, lambda_risks, TMSE) for d, data in DRUG_DATABASE.items()]
df_results = pd.DataFrame(raw_results).sort_values(by="Net Score (Mj)", ascending=False).reset_index(drop=True)

# Ruled-out medications
df_filtered = df_results[~df_results["Drug"].isin(st.session_state.excluded_drugs)].reset_index(drop=True)

# Format
for col in ["Net Score (Mj)", "Therapeutic Gain", "Risk Deductions", "ACB Penalty", "M1 Potency (pKi)"]:
    df_results[col] = df_results[col].map("{:.1f}".format)
    df_filtered[col] = df_filtered[col].map("{:.1f}".format)

# Render Top Recommended Drug Card & Dynamic Rule-Out Engine
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
                &nbsp;|&nbsp; M1 Potency: <strong>{top_drug['M1 Potency (pKi)']}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Black Box Warning & Clinical Caution Box
    st.warning(f"⚠️ **Clinical Cautions & Warnings for {top_name}:**\n\n{BLACK_BOX_WARNINGS.get(top_name, 'No specific black box warning listed.')}")
    
    # Rule-Out Checkbox
    rule_out_check = st.checkbox(
        f"🚫 **Rule out {top_name} for this patient** (Check if patient has contraindications, high risks for exact medication, intolerance or allergy)",
        key=f"ruleout_{top_name}"
    )
    
    if rule_out_check:
        st.session_state.excluded_drugs.append(top_name)
        st.rerun()

else:
    st.error("All candidate medications have been ruled out. Please reset the rule-out filters.")

# Reset Button for Rule-Out Filters
if st.session_state.excluded_drugs:
    st.markdown(" ")
    if st.button(f"🔄 Reset Ruled-Out Medications ({len(st.session_state.excluded_drugs)} Currently Excluded)"):
        st.session_state.excluded_drugs = []
        st.rerun()

st.markdown("### Complete Comparative Drug Matrix")

# Appearances
def apply_traffic_lights(val):
    val_float = float(val)
    if val_float > 1.0:
        return 'background-color: #d4edda; color: #155724; font-weight: bold;'
    elif val_float >= -2.0:
        return 'background-color: #fff3cd; color: #856404;'
    else:
        return 'background-color: #f8d7da; color: #721c24;'

# Highlight ruled-out drugs in the complete table
df_results_display = df_results.copy()
df_results_display["Status"] = df_results_display["Drug"].apply(
    lambda x: "❌ Ruled Out" if x in st.session_state.excluded_drugs else "✅ Candidate"
)

# Interactive Table
st.dataframe(
    df_results_display.style.map(apply_traffic_lights, subset=['Net Score (Mj)']),
    use_container_width=True
)

st.info("**Traffic Light Guide:** Green = Optimal Match (Mj > 1.0) | Yellow = Proceed with Caution (-2.0 ≤ Mj ≤ 1.0) | Red = High Risk Flag (Mj < -2.0)")

st.markdown("---")
# Rationale
with st.expander("🔍 Background Rationale & Expanded Pharmacodynamic Details"):
    st.markdown(
        """
        ### Multi-Neurotransmitter Algorithmic Architecture
        
        The Net Therapeutic Match Score ($M_j$) evaluates psychotropic suitability across 6 distinct neurochemical systems by balancing symptom-weighted target affinities against off-target safety penalties:
        
        $$M_j = U_{\\text{thera}} - U_{\\text{risk}} - P_{\\text{ACB}}$$
        
        ---

        #### 1. Bedside Parameter Derivation & Weighting Logic
        * **Target Weights ($\omega_r$):** Derived directly from bedside rating scales like the NPI-Q. The clinician's 4-point input score ($0$ to $3$) is normalized via $\omega_r$ = $$\\frac{\\text{Score}}{3.0}$$, producing a bounded parameter $\omega_r \in [0.0, 1.0]$.
        * **Histamine Risk ($\lambda_{H1}$):** Mapped to the bedside **Morse Fall Scale** ($0\\text{-}125$), where scores $\ge 45$ set $\lambda_{H1} = 0.9$ to penalize central $H_1$ sedation and gait instability.
        * **Orthostatic Risk ($\lambda_{α1}$):** Mapped to standing systolic blood pressure drop, where diagnostic orthostasis ($\Delta\\text{SBP} \ge 20 \\text{ mmHg}$) sets $\lambda_{α1} = 0.9$ to penalize $\\alpha_1$-adrenergic blockade.
        * **Parkinsonism Risk ($\lambda_{D2}$):** Mapped to extrapyramidal motor signs (Simpson-Angus Scale), where pre-existing rigidity or Lewy body dementia sets $\lambda_{D2} = 1.0$, heavily penalizing full $D_2$ antagonists like Haloperidol.

        ---

        #### 2. Multi-Target Therapeutic Gain ($U_{\\text{thera}}$)
        $$U_{\\text{thera}} = \\sum_{r \\in \\{5HT2A, D2, NET, α2a, NMDA, GABA-A\\}} \\left( \\omega_r \\cdot pK_{i,r} \\cdot A_r \\right)$$
        
        * **$5\\text{-HT}_{2\\text{A}}$ Target:** $5\\text{-HT}_{2\\text{A}}$ inverse agonism ($A_r = +1.0$) attenuates cortical serotonergic hyperfunction driving psychotic agitation and hallucinations.
        * **$D_2$ & $\\text{NET}$ Targets:** Frontostriatal dopamine and norepinephrine hypofunction drive apathy. Partial $D_2$ agonists (e.g. Brexpiprazole, Aripiprazole) stabilize transmission ($A_r = +1.0$) without causing extrapyramidal motor block.
        * **$\\alpha_{2\\text{A}}$ Target:** Presynaptic $\\alpha_{2A}$ agonism ($A_r = +1.0$) suppresses central noradrenergic outflow, reducing hyperadrenergic autonomic arousal.
        * **$\\text{GABA}_{A}$ Target:** Positive allosteric modulation ($A_r = +1.0$) enhances central inhibition to rapidly calm severe panic.
        * **$\\text{NMDA}$ Target:** Uncompetitive $\\text{NMDA}$ antagonism ($A_r = +1.0$) protects against glutamatergic excitotoxicity and delirium.

        ---

        #### 3. Cognitive Burden Penalty ($P_{\\text{ACB}}$) & Rule-Out Guardrails
        * **$M_1$ Potency Threshold:** Central $M_1$ muscarinic blockade triggers a step-function penalty when binding potency reaches $pK_{i,M1} \\ge 7.0$ ($K_i \\le 100\\text{ nM}$).
        * **Cognitive Scaling ($C_{\\text{patient}}$):** Scaled via baseline TMSE score ($3.0$ for TMSE < 10, $2.0$ for TMSE 10-20, and $1.0$ for TMSE > 20).
        * **Dynamic Rule-Out Engine:** When a clinician checks the rule-out box for a top drug, the system dynamically filters out that agent and recalculates the matrix to present the safest second-line alternative.
        """
    )
with st.expander("🔍 Core References & Algorithmic Citations"):
	references_markdown = 
	"""
	1. **Roth BL, et al.** *PDSP Ki Database. Psychoactive Drug Screening Program (PDSP)*. University of North Carolina at Chapel Hill and the United States National Institute of Mental Health.

	2. **Magierski R, et al. (2020).** *Pharmacotherapy of Behavioral and Psychological Symptoms of Dementia: State of the Art and Future Progress*. Front. Pharmacol. 11:1168. doi: 10.3389/fphar.2020.01168.

	3. **Caraci F., et al. (2020).** *New antipsychotic drugs for the treatment of agitation and psychosis in Alzheimer's disease: focus on brexpiprazole and pimavanserin*. F1000Res:F1000 Faculty Rev-686. doi: 10.12688/f1000research.22662.1. PMID: 32695312; PMCID: PMC7344175.

	4. **Kim H, et al. (2026).** *Brexpiprazole for the Treatment of Agitation Associated with Dementia due to Alzheimer's Disease: Clinical Perspectives*. Clin Psychopharmacol Neurosci;24(1):15-29. doi: 10.9758/cpn.24.1252.

	5. **Davies, S. J., et al. (2018).** *Sequential drug treatment algorithm for agitation and aggression in Alzheimer's and mixed dementia*. Journal of psychopharmacology, 32(5), 509–523. https://doi.org/10.1177/0269881117744996

	6. **Kales H. C., et al. (2015).** *Assessment and management of behavioral and psychological symptoms of dementia*. BMJ, 350, h369. https://doi.org/10.1136/bmj.h369

	7. **Cummings J., et al. (2022).** *Alzheimer's disease drug development pipeline: 2022*. Alzheimer's & dementia, 8(1), e12295. https://doi.org/10.1002/trc2.12295

	8. **Hatch S., et al. (2025).** *The Canadian Coalition for Seniors' Mental Health Canadian Clinical Practice Guidelines for Assessing and Managing Behavioural and Psychological Symptoms of Dementia (BPSD)*. Canadian geriatrics journal, 28(1), 91–102. https://doi.org/10.5770/cgj.28.820

	9. **Siafis, S., et al. (2018).** *Antipsychotic Drugs: From Receptor-binding Profiles to Metabolic Side Effects*. Current neuropharmacology, 16(8), 1210–1223. https://doi.org/10.2174/1570159X15666170630163616
	"""

st.markdown(references_markdown)

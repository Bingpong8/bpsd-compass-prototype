import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="BPSD Compass Prototype (P3)", layout="wide")
st.title("BPSD Compass Prototype (P3)")
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

#-----------------------------------------------------------------------------
# 1. PHARMACODYNAMIC DATABASE
# -----------------------------------------------------------------------------
DRUG_DATABASE = {
    "Brexpiprazole": {
        "category": "Atypical Antipsychotic",
        "pKi": {"5HT2A": 8.7, "D2": 9.5, "NET": 5.0, "a2A": 7.4, "NMDA": 0.0, "GABA-A": 0.0, "H1": 7.1, "alpha1": 8.0, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": 1.0, "NET": 0.0, "a2A": 1.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.14, "Fr_hepatic": 0.86, "Risk_QTc": 0.20,
        "warnings": "Exercise extreme caution for akathisia and impulse-control disorders."
    },
    "Pimavanserin": {
        "category": "Atypical Antipsychotic",
        "pKi": {"5HT2A": 9.3, "D2": 5.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 5.0, "alpha1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.06, "Fr_hepatic": 0.94, "Risk_QTc": 0.40,
        "warnings": "QTc prolongation."
    },
    "Risperidone": {
        "category": "Atypical Antipsychotic",
        "pKi": {"5HT2A": 9.7, "D2": 8.9, "NET": 5.0, "a2A": 6.8, "NMDA": 0.0, "GABA-A": 0.0, "H1": 7.3, "alpha1": 9.0, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": -1.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.70, "Fr_hepatic": 0.30, "Risk_QTc": 0.50,
        "warnings": "QTc prolongation, Dose-dependent extrapyramidal symptoms and hyperprolactinemia with elevated risk of cerebrovascular adverse events (stroke/TIA)."
    },
    "Quetiapine": {
        "category": "Atypical Antipsychotic",
        "pKi": {"5HT2A": 6.8, "D2": 5.8, "NET": 5.0, "a2A": 5.5, "NMDA": 0.0, "GABA-A": 0.0, "H1": 8.0, "alpha1": 7.1, "M1": 6.0},
        "Ar": {"5HT2A": 1.0, "D2": -1.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.05, "Fr_hepatic": 0.95, "Risk_QTc": 0.40,
        "warnings": "QTc prolongation, Risk of severe orthostatic hypotension, somnolence, and metabolic dysregulation."
    },
    "Olanzapine": {
        "category": "Atypical Antipsychotic",
        "pKi": {"5HT2A": 8.5, "D2": 7.8, "NET": 5.0, "a2A": 6.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 8.8, "alpha1": 7.7, "M1": 7.7},
        "Ar": {"5HT2A": 1.0, "D2": -1.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.07, "Fr_hepatic": 0.93, "Risk_QTc": 0.30,
        "warnings": "High risk of severe metabolic syndrome, rapid weight gain, profound sedation, and anticholinergic cognitive impairment (M1 pKi ≥ 7.0)."
    },
    "Haloperidol": {
        "category": "Typical Antipsychotic",
        "pKi": {"5HT2A": 7.2, "D2": 9.2, "NET": 5.0, "a2A": 5.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 6.0, "alpha1": 7.3, "M1": 5.0},
        "Ar": {"5HT2A": 0.0, "D2": -1.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.15, "Fr_hepatic": 0.85, "Risk_QTc": 0.85,
        "warnings": "QTc prolongation / Torsades de Pointes, High risk of severe Extrapyramidal Symptoms (EPS), Tardive Dyskinesia"
    },
    "Escitalopram": {
        "category": "Antidepressant (SSRI)",
        "pKi": {"5HT2A": 5.2, "D2": 5.0, "NET": 5.0, "a2A": 5.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 6.3, "alpha1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 0.5, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.20, "Fr_hepatic": 0.80, "Risk_QTc": 0.75,
        "warnings": "Dose-dependent QTc prolongation risk; maximum 10 mg/day recommended in elderly."
    },
    "Sertraline": {
        "category": "Antidepressant (SSRI)",
        "pKi": {"5HT2A": 6.2, "D2": 6.6, "NET": 5.5, "a2A": 5.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 5.0, "alpha1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 0.5, "D2": 0.5, "NET": 0.5, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.12, "Fr_hepatic": 0.88, "Risk_QTc": 0.25,
        "warnings": "Severe hyponatremia/SIADH, serotonin syndrome and bleeding risk, Generally well-tolerated QTc profile in elderly."
    },
    "Valproic Acid": {
        "category": "Mood Stabilizer",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 7.2, "H1": 5.0, "alpha1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 1.0},
        "Fr_renal": 0.05, "Fr_hepatic": 0.95, "Risk_QTc": 0.10,
        "warnings": "Hepatotoxicity, pancreatitis, thrombocytopenia; require monitoring of LFTs and CBC."
    },
    "Gabapentin": {
        "category": "Mood Stabilizer / Anticonvulsant",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 6.8, "H1": 5.0, "alpha1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 1.0},
        "Fr_renal": 1.00, "Fr_hepatic": 0.00, "Risk_QTc": 0.05,
        "warnings": "Respiratory depression risk with opioids or CNS depressants; strict renal dose adjustment required."
    },
    "Memantine": {
        "category": "Cognitive Enhancer",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 7.5, "GABA-A": 0.0, "H1": 0.0, "alpha1": 0.0, "M1": 0.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 1.0, "GABA-A": 0.0},
        "Fr_renal": 0.80, "Fr_hepatic": 0.20, "Risk_QTc": 0.05,
        "warnings": "Dose reduction necessary in severe renal impairment (eGFR < 30 mL/min)."
    },
	"Amitriptyline": {
        "category": "Tricyclic Antidepressant (TCA)",
        "pKi": {"5HT2A": 8.1, "D2": 5.5, "NET": 7.7, "a2A": 6.8, "NMDA": 0.0, "GABA-A": 0.0, "H1": 8.9, "alpha1": 8.0, "M1": 8.8},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 1.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.05, "Fr_hepatic": 0.95, "Risk_QTc": 0.70,
        "warnings": "Severe anticholinergic, fall, and cardiotoxicity/arrhythmia risk in dementia; suicidality warning."
    },
    "Venlafaxine": {
        "category": "Antidepressant (SNRI)",
        "pKi": {"5HT2A": 5.0, "D2": 5.0, "NET": 6.4, "a2A": 5.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 5.0, "alpha1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 1.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.85, "Fr_hepatic": 0.15, "Risk_QTc": 0.35,
        "warnings": "Dose-dependent blood pressure elevation and withdrawal discontinuation syndrome."
    },
    "Vortioxetine": {
        "category": "Multimodal Antidepressant",
        "pKi": {"5HT2A": 7.60, "D2": 5.00, "NET": 5.00, "a2A": 5.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 5.00, "alpha1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.59, "Fr_hepatic": 0.41, "Risk_QTc": 0.15,
        "warnings": "Favorable cognitive and low anticholinergic profile."
    },
    "Bupropion": {
        "category": "Antidepressant (NDRI)",
        "pKi": {"5HT2A": 5.00, "D2": 5.20, "NET": 5.20, "a2A": 5.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 5.00, "alpha1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 0.0, "D2": 0.5, "NET": 1.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.87, "Fr_hepatic": 0.13, "Risk_QTc": 0.20,
        "warnings": "Dose-dependent seizure risk; contraindicated in eating disorders or abrupt withdrawal of alcohol/sedatives."
    },
    "Trazodone": {
        "category": "Antidepressant (SARI)",
        "pKi": {"5HT2A": 7.80, "D2": 5.00, "NET": 5.00, "a2A": 6.40, "NMDA": 5.00, "GABA-A": 5.00, "H1": 7.50, "alpha1": 7.80, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.70, "Fr_hepatic": 0.30, "Risk_QTc": 0.45,
        "warnings": "Orthostatic hypotension, priapism, and marked sedation; used off-label PRN for agitation/insomnia in dementia."
    },
    "Mirtazapine": {
        "category": "Antidepressant (NaSSA)",
        "pKi": {"5HT2A": 8.1, "D2": 5.0, "NET": 5.0, "a2A": 7.3, "NMDA": 0.0, "GABA-A": 0.0, "H1": 9.3, "alpha1": 6.0, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "a2A": 1.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.75, "Fr_hepatic": 0.25, "Risk_QTc": 0.30,
        "warnings": "Marked sedation at low doses and weight gain; limited efficacy demonstrated for agitation in dementia (SYMBAD trial)."
    },
    "Mianserin": {
        "category": "Tetracyclic Antidepressant (NaSSA)",
        "pKi": {"5HT2A": 8.0, "D2": 5.5, "NET": 6.0, "a2A": 7.2, "NMDA": 0.0, "GABA-A": 0.0, "H1": 9.0, "alpha1": 7.3, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.5, "a2A": 1.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.70, "Fr_hepatic": 0.30, "Risk_QTc": 0.35,
        "warnings": "Risk of agranulocytosis and bone marrow depression, High sedation and orthostasis profile."
    },
    "Lamotrigine": {
        "category": "Mood Stabilizer / Anticonvulsant",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 6.5, "GABA-A": 6.0, "H1": 0.0, "alpha1": 0.0, "M1": 0.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.5, "GABA-A": 0.5},
        "Fr_renal": 0.94, "Fr_hepatic": 0.06, "Risk_QTc": 0.10,
        "warnings": "Severe cutaneous adverse reactions including Stevens-Johnson syndrome (SJS) and Toxic Epidermal Necrolysis (TEN)."
    },
    "Carbamazepine": {
        "category": "Mood Stabilizer / Anticonvulsant",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 6.0, "GABA-A": 6.8, "H1": 0.0, "alpha1": 0.0, "M1": 0.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "a2A": 0.0, "NMDA": 0.0, "GABA-A": 1.0},
        "Fr_renal": 0.28, "Fr_hepatic": 0.72, "Risk_QTc": 0.30,
        "warnings": "Aplastic anemia, agranulocytosis, and severe dermatologic reactions (SJS/TEN); potent CYP3A4 enzyme auto-inducer."
    }

# -----------------------------------------------------------------------------
# 2. CONTINUOUS SIGMOIDAL SCALING FUNCTIONS
# -----------------------------------------------------------------------------
def sigmoid(x, k, x0):
    """Normalized Sigmoidal Function: 1 / (1 + exp(-k * (x - x0)))"""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

def inverted_sigmoid(x, k, x0):
    """Inverted Sigmoidal Function for parameters where risk increases as score decreases (e.g. eGFR)"""
    return 1.0 / (1.0 + np.exp(k * (x - x0)))

def calculate_sigmoidal_lambdas(morse, sbp_drop, sas, qtc, egfr, lft_factor, dementia_subtype):
    # Fall Risk (Morse Fall Scale 0-125, Midpoint = 35)
    lambda_H1 = sigmoid(morse, 0.08, 35.0)
    
    # Orthostatic Hypotension (SBP drop mmHg, Midpoint = 15)
    lambda_alpha1 = sigmoid(sbp_drop, 0.25, 15.0)
    
    # Parkinsonism / Motor Risk (SAS 0-40, Midpoint = 8)
    lambda_D2_full = min(1.0, sigmoid(sas, 0.30, 8.0))
    
    # Hard-Lock Override for Lewy Body / Parkinson's Dementia
    if dementia_subtype in ["Dementia with Lewy Bodies (DLB)", "Parkinson's Disease Dementia (PDD)"]:
        lambda_D2_full = 1.0
        
    # Cardiac QTc Risk (Baseline QTc ms, Midpoint = 450)
    lambda_QTc = sigmoid(qtc, 0.05, 450.0)
    
    # Organ Clearance Penalties
    lambda_renal = inverted_sigmoid(egfr, 0.08, 45.0)
    lambda_hepatic = lft_factor  # Direct scalar 0.0 to 1.0 from UI LFT slider
    
    return {
        "H1": lambda_H1,
        "alpha1": lambda_alpha1,
        "D2_full": lambda_D2_full,
        "QTc": lambda_QTc,
        "renal": lambda_renal,
        "hepatic": lambda_hepatic
    }

# -----------------------------------------------------------------------------
# 3. CORE CALCULATION ENGINE
# -----------------------------------------------------------------------------
def calculate_p3_match_score(drug_name, drug_data, weights, lambdas, mmse_score, dementia_subtype, qtc_ms):
    pk = drug_data["pKi"]
    ar = drug_data["Ar"]
    
    # Hard-Lock / Exclusion Check
    hard_locked = False
    hard_lock_reason = ""
    
    if dementia_subtype in ["Dementia with Lewy Bodies (DLB)", "Parkinson's Disease Dementia (PDD)"] and ar["D2"] < 0:
        hard_locked = True
        hard_lock_reason = "Contraindicated: Full D2 antagonist in DLB/PDD etiology"
    elif qtc_ms > 500.0 and drug_data["Risk_QTc"] > 0.60:
        hard_locked = True
        hard_lock_reason = "Contraindicated: Severe baseline QTc (>500ms) with high QTc-risk agent"

    # Multi-Neurotransmitter Therapeutic Utility Sum (U_thera)
    u_thera = (
        (weights["5HT2A"] * pk["5HT2A"] * ar["5HT2A"]) +
        (weights["D2"] * pk["D2"] * ar["D2"]) +
        (weights["NET"] * pk["NET"] * ar["NET"]) +
        (weights["a2A"] * pk["a2A"] * ar["a2A"]) +
        (weights["NMDA"] * pk["NMDA"] * ar["NMDA"]) +
        (weights["GABA-A"] * pk["GABA-A"] * ar["GABA-A"])
    )
    
    # Continuous Risk Deductions (U_risk)
    d2_risk = (lambdas["D2_full"] * pk["D2"]) if ar["D2"] < 0 else 0.0
    qtc_risk_deduction = lambdas["QTc"] * drug_data["Risk_QTc"] * 5.0
    
    u_risk = (
        (lambdas["H1"] * pk["H1"]) +
        (lambdas["alpha1"] * pk["alpha1"]) +
        d2_risk +
        qtc_risk_deduction
    )
    
    # Anticholinergic Cognitive Penalty (P_ACB)
    if mmse_score < 10:
        c_patient = 3.0
    elif mmse_score <= 20:
        c_patient = 2.0
    else:
        c_patient = 1.0
        
    p_acb = (c_patient * 2.0) if pk["M1"] >= 7.0 else 0.0
    
    # Organ Clearance Penalty (P_organ)
    p_organ = (lambdas["renal"] * drug_data["Fr_renal"] * 4.0) + (lambdas["hepatic"] * drug_data["Fr_hepatic"] * 4.0)
    
    # Net Match Score (Mj)
    m_j = u_thera - u_risk - p_acb - p_organ
    
    if hard_locked:
        m_j = -999.0  # Force demotion
        
    # Estimated Side-Effect Probabilities (%)
    p_sedation = min(95, int(sigmoid(pk["H1"] * lambdas["H1"], 0.5, 3.5) * 100))
    p_orthostasis = min(95, int(sigmoid(pk["alpha1"] * lambdas["alpha1"], 0.5, 3.5) * 100))
    p_eps = min(95, int(sigmoid(pk["D2"] * lambdas["D2_full"], 0.5, 4.0) * 100)) if ar["D2"] < 0 else 5
    
    return {
        "Drug": drug_name,
        "Category": drug_data["category"],
        "Net Score (Mj)": round(m_j, 1) if not hard_locked else "HARD-LOCKED",
        "Raw_Mj": m_j,
        "Therapeutic Gain": round(u_thera, 1),
        "Risk Deductions": round(u_risk, 1),
        "ACB Penalty": round(p_acb, 1),
        "Organ Penalty": round(p_organ, 1),
        "Est. Sedation %": f"{p_sedation}%",
        "Est. Orthostasis %": f"{p_orthostasis}%",
        "Est. EPS %": f"{p_eps}%",
        "Hard Locked": hard_locked,
        "Lock Reason": hard_lock_reason,
        "Warnings": drug_data["warnings"]
    }

# -----------------------------------------------------------------------------
# 4. USER INTERFACE (SIDEBAR & BEDSIDE INPUTS)
# -----------------------------------------------------------------------------
st.sidebar.header("📋 Patient Clinical Parameters")

# Etiology Subtype
dementia_subtype = st.sidebar.selectbox(
    "Dementia Etiology / Subtype",
    ["Alzheimer's Disease (AD)", "Dementia with Lewy Bodies (DLB)", "Parkinson's Disease Dementia (PDD)", "Vascular Dementia (VaD)", "Frontotemporal Dementia (FTD)"]
)

st.sidebar.subheader("Target Symptoms (NPI Severity)")
s_agitation = st.sidebar.slider("Agitation / Aggression", 0.0, 1.0, 0.6)
s_psychosis = st.sidebar.slider("Psychosis (Delusions/Hallucinations)", 0.0, 1.0, 0.4)
s_apathy = st.sidebar.slider("Apathy / Executive Dysfunction", 0.0, 1.0, 0.2)
s_affective = st.sidebar.slider("Affective Lability / Mood Instability", 0.0, 1.0, 0.5)

st.sidebar.subheader("Physiological Risk & Biomarkers")
morse_score = st.sidebar.number_input("Morse Fall Scale Score (0-125)", 0, 125, 40)
sbp_drop = st.sidebar.number_input("Standing SBP Drop (mmHg)", 0, 60, 12)
sas_score = st.sidebar.number_input("SAS Motor / EPS Score (0-40)", 0, 40, 4)
qtc_ms = st.sidebar.number_input("Baseline QTc Interval (ms)", 300, 600, 430)
mmse_score = st.sidebar.number_input("MMSE / MoCA Cognitive Score (0-30)", 0, 30, 14)
egfr_val = st.sidebar.number_input("eGFR", 5, 120, 55)
lft_val = st.sidebar.slider("Hepatic Impairment Level (0 = Normal, 1 = Severe)", 0.0, 1.0, 0.2)

st.sidebar.subheader("Caregiver Preference Weights (Proxy Input)")
pref_sedation = st.sidebar.slider("Caregiver Avoid-Sedation Weight", 0.5, 1.5, 1.0)
pref_falls = st.sidebar.slider("Caregiver Avoid-Fall Weight", 0.5, 1.5, 1.0)

# -----------------------------------------------------------------------------
# 5. RUN ALGORITHMIC COMPUTATIONS
# -----------------------------------------------------------------------------
weights = {
    "5HT2A": s_psychosis,
    "D2": s_agitation,
    "NET": s_apathy,
    "a2A": s_agitation * 0.5,
    "NMDA": s_apathy * 0.5,
    "GABA-A": s_affective
}

# Calculate Sigmoidal Lambdas
lambdas = calculate_sigmoidal_lambdas(
    morse=morse_score * pref_falls,
    sbp_drop=sbp_drop,
    sas=sas_score,
    qtc=qtc_ms,
    egfr=egfr_val,
    lft_factor=lft_val,
    dementia_subtype=dementia_subtype
)

# Compute match scores for all candidates
results = []
for drug_name, drug_data in DRUG_DATABASE.items():
    res = calculate_p3_match_score(
        drug_name=drug_name,
        drug_data=drug_data,
        weights=weights,
        lambdas=lambdas,
        mmse_score=mmse_score,
        dementia_subtype=dementia_subtype,
        qtc_ms=qtc_ms
    )
    results.append(res)

# Sort results by raw Mj
results = sorted(results, key=lambda x: x["Raw_Mj"], reverse=True)
top_drug = results[0]

# -----------------------------------------------------------------------------
# 6. DASHBOARD DISPLAY & VISUALIZATIONS
# -----------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Primary Recommended Candidate")
    if top_drug["Hard Locked"]:
        st.error("No suitable candidate found. All eligible agents triggered critical clinical safety hard-locks.")
    else:
        st.success(f"**Top Recommended Match: {top_drug['Drug']}** ({top_drug['Category']})")
        st.metric(label="Net Match Score (Mj)", value=top_drug["Net Score (Mj)"])
        st.caption(f"**Warnings / Alert:** {top_drug['Warnings']}")

with col2:
    st.subheader("⚙️ Sigmoidal Risk Scalars (λ)")
    st.write(f"- **Fall Risk (λ_H1):** `{lambdas['H1']:.2f}`")
    st.write(f"- **Orthostasis Risk (λ_α1):** `{lambdas['alpha1']:.2f}`")
    st.write(f"- **Motor EPS Risk (λ_D2):** `{lambdas['D2_full']:.2f}`")
    st.write(f"- **Cardiac QTc Risk (λ_QTc):** `{lambdas['QTc']:.2f}`")
    st.write(f"- **Renal Penalty (λ_renal):** `{lambdas['renal']:.2f}`")

st.markdown("---")
st.subheader("📊 Comparative Candidate Rankings")

df_results = pd.DataFrame(results)
display_df = df_results[[
    "Drug", "Category", "Net Score (Mj)", "Therapeutic Gain", 
    "Risk Deductions", "ACB Penalty", "Organ Penalty", 
    "Est. Sedation %", "Est. Orthostasis %", "Est. EPS %", "Lock Reason"
]]

st.dataframe(display_df, use_container_width=True)

# -----------------------------------------------------------------------------
# 7. LONGITUDINAL TAPERING & CROSS-TITRATION DRAWER
# -----------------------------------------------------------------------------
with st.expander("🔄 Cross-Titration & Switching Protocol Generator"):
    st.markdown("""
    **Patient Transition Protocol Generator**
    When transitioning from a high-affinity D2 antagonist (e.g. Risperidone) to a D2 partial agonist (e.g. Brexpiprazole) or non-dopaminergic agent (e.g. Pimavanserin):
    
    *   **Week 1:** Reduce prior agent dose by 50%. Initiate target agent at 0.5 mg/day baseline.
    *   **Week 2:** Maintain taper. Monitor for cholinergic rebound or withdrawal psychosis.
    *   **Week 3:** Discontinue prior agent completely. Titrate target agent to optimal therapeutic score dosage.
    """)

# -----------------------------------------------------------------------------
# 8. CITATIONS & ALGORITHMIC REFERENCES
# -----------------------------------------------------------------------------
with st.expander("🔍 References & Citations"):
    st.markdown(
        """
        1. **Roth, B. L., et al.** *PDSP Ki Database. Psychoactive Drug Screening Program (PDSP)*. UNC Chapel Hill / NIMH.
        2. **Magierski, R., et al. (2020).** *Pharmacotherapy of Behavioral and Psychological Symptoms of Dementia: State of the Art and Future Progress*. Front. Psychiatry. PMID: 32848775.
        3. **Tampi, R. R., et al. (2022).** *Brexpiprazole for the Treatment of Agitation in Dementia*. Drugs Aging. PMID: 35904712.
        4. **Lee, D., et al. (2023).** *Brexpiprazole for the Treatment of Agitation Associated with Dementia Due to Alzheimer's Disease*. Am J Psychiatry. PMID: 37143168.
        5. **Davies, S. J., et al. (2018).** *Sequential drug treatment algorithm for agitation and aggression in Alzheimer's and mixed dementia*. J Psychopharmacol. PMID: 29338602.
        6. **Kales, H. C., et al. (2015).** *Assessment and management of behavioral and psychological symptoms of dementia*. BMJ. PMID: 25731898.
        7. **Cummings, J., et al. (2022).** *Alzheimer's disease drug development pipeline: 2022*. Alzheimers Dement (NY). PMID: 35510134.
        8. **CCSMH (2024–2025).** *Canadian Clinical Practice Guidelines for Assessing and Managing BPSD*. ccsmh.ca.
        """

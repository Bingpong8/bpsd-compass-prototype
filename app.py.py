import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="BPSD Compass Prototype (P3.5)", layout="wide")
st.title("BPSD Compass Prototype (P3.5)")
st.caption("Parameter-driven neurotransmitter affinity decision-support tool")

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

# -----------------------------------------------------------------------------
# 1. PHARMACODYNAMIC DATABASE WITH DOSAGE SPECTRA
# -----------------------------------------------------------------------------
DRUG_DATABASE = {
    "Brexpiprazole": {
        "category": "Atypical Antipsychotic",
        "pKi": {"5HT2A": 8.7, "D2": 9.5, "NET": 5.0, "α2A": 7.4, "NMDA": 0.0, "GABA-A": 0.0, "H1": 7.1, "α1": 8.0, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": 1.0, "NET": 0.0, "α2A": 1.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.14, "Fr_hepatic": 0.86, "Risk_QTc": 0.20,
        "dosage": "Start 0.5 mg PO OD; max 2 mg/day for agitation in dementia.",
        "warnings": "Exercise extreme caution for akathisia and impulse-control disorders. Monitor elderly closely."
    },
    "Pimavanserin": {
        "category": "Atypical Antipsychotic",
        "pKi": {"5HT2A": 9.3, "D2": 5.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 5.0, "α1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.06, "Fr_hepatic": 0.94, "Risk_QTc": 0.40,
        "dosage": "Standard dose: 34 mg PO OD (or 10 mg PO OD in CYP3A4 inhibitor co-administration).",
        "warnings": "QTc prolongation risk. Indicated primarily for Parkinson's Disease Psychosis."
    },
    "Risperidone": {
        "category": "Atypical Antipsychotic",
        "pKi": {"5HT2A": 9.7, "D2": 8.9, "NET": 5.0, "α2A": 6.8, "NMDA": 0.0, "GABA-A": 0.0, "H1": 7.3, "α1": 9.0, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": -1.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.70, "Fr_hepatic": 0.30, "Risk_QTc": 0.50,
        "dosage": "Start 0.25 mg - 0.5 mg/day; target 0.5 mg - 1.5 mg/day (max 2.0 mg/day in elderly).",
        "warnings": "QTc prolongation, dose-dependent extrapyramidal symptoms (EPS), hyperprolactinemia, and cerebrovascular risk."
    },
    "Quetiapine": {
        "category": "Atypical Antipsychotic",
        "pKi": {"5HT2A": 6.8, "D2": 5.8, "NET": 5.0, "α2A": 5.5, "NMDA": 0.0, "GABA-A": 0.0, "H1": 8.0, "α1": 7.1, "M1": 6.0},
        "Ar": {"5HT2A": 1.0, "D2": -1.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.05, "Fr_hepatic": 0.95, "Risk_QTc": 0.40,
        "dosage": "Start 12.5 mg - 25 mg PO hs",
        "warnings": "QTc prolongation, severe orthostatic hypotension, sedation, and metabolic dysregulation."
    },
    "Olanzapine": {
        "category": "Atypical Antipsychotic",
        "pKi": {"5HT2A": 8.5, "D2": 7.8, "NET": 5.0, "α2A": 6.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 8.8, "α1": 7.7, "M1": 7.7},
        "Ar": {"5HT2A": 1.0, "D2": -1.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.07, "Fr_hepatic": 0.93, "Risk_QTc": 0.30,
        "dosage": "Start 2.5 mg PO hs; target 5.0 mg/day, max 10 mg/day.",
        "warnings": "High risk of severe metabolic syndrome, weight gain, sedation, and anticholinergic cognitive impairment."
    },
    "Haloperidol": {
        "category": "Typical Antipsychotic",
        "pKi": {"5HT2A": 7.2, "D2": 9.2, "NET": 5.0, "α2A": 5.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 6.0, "α1": 7.3, "M1": 5.0},
        "Ar": {"5HT2A": 0.0, "D2": -1.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.15, "Fr_hepatic": 0.85, "Risk_QTc": 0.85,
        "dosage": "Start as low as possible, 0.25-0.5 mg PO OD or PRN - 2.0 mg/day maximum.",
        "warnings": "HIGH RISK: Torsades de Pointes, severe Extrapyramidal Symptoms (EPS), and Tardive Dyskinesia."
    },
    "Escitalopram": {
        "category": "Antidepressant (SSRI)",
        "pKi": {"5HT2A": 5.2, "D2": 5.0, "NET": 5.0, "α2A": 5.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 6.3, "α1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 0.5, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.20, "Fr_hepatic": 0.80, "Risk_QTc": 0.75,
        "dosage": "Start 5 mg/day; max 10 mg/day.",
        "warnings": "Dose-dependent QTc prolongation risk; hyponatremia and bleeding precautions."
    },
    "Sertraline": {
        "category": "Antidepressant (SSRI)",
        "pKi": {"5HT2A": 6.2, "D2": 6.6, "NET": 5.5, "α2A": 5.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 5.0, "α1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 0.5, "D2": 0.5, "NET": 0.5, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.12, "Fr_hepatic": 0.88, "Risk_QTc": 0.25,
        "dosage": "Start 25 mg/day; target 50 mg - 100 mg/day.",
        "warnings": "Hyponatremia/SIADH, serotonin syndrome, and mild GI distress. Well-tolerated cardiac profile."
    },
    "Valproic Acid": {
        "category": "Mood Stabilizer Anticonvulsant",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 7.2, "H1": 5.0, "α1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 1.0},
        "Fr_renal": 0.05, "Fr_hepatic": 0.95, "Risk_QTc": 0.10,
        "dosage": "Start 125 mg - 250 mg PO bid; target serum concentration 50-80 mcg/ml.",
        "warnings": "Hepatotoxicity, pancreatitis, thrombocytopenia; monitor LFTs, CBC, and plasma levels."
    },
    "Gabapentin": {
        "category": "GABA analogue Anticonvulsant",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 6.8, "H1": 5.0, "α1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 1.0},
        "Fr_renal": 1.00, "Fr_hepatic": 0.00, "Risk_QTc": 0.05,
        "dosage": "Start 100 mg tid; slow titrations up to 300 mg - 600 mg tid based on renal clearance.",
        "warnings": "Respiratory depression risk with CNS depressants/opioids; strict renal dose reduction required."
    },
    "Memantine": {
        "category": "Cognitive Enhancer",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 7.5, "GABA-A": 0.0, "H1": 0.0, "α1": 0.0, "M1": 0.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 1.0, "GABA-A": 0.0},
        "Fr_renal": 0.80, "Fr_hepatic": 0.20, "Risk_QTc": 0.05,
        "dosage": "Start 5 mg daily; titrate by 5 mg weekly to target 10 mg bid (max 10 mg daily if eGFR < 30).",
        "warnings": "Dose adjustment necessary in severe renal impairment (eGFR < 30 mL/min)."
    },
    "Amitriptyline": {
        "category": "Tricyclic Antidepressant (TCA)",
        "pKi": {"5HT2A": 8.1, "D2": 5.5, "NET": 7.7, "α2A": 6.8, "NMDA": 0.0, "GABA-A": 0.0, "H1": 8.9, "α1": 8.0, "M1": 8.8},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 1.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.05, "Fr_hepatic": 0.95, "Risk_QTc": 0.70,
        "dosage": "Generally avoid in dementia. (10 mg hs start if strictly indicated).",
        "warnings": "CRITICAL RISK: Severe anticholinergic toxicity, fall risk, cognitive decline, and cardiotoxicity."
    },
    "Venlafaxine": {
        "category": "Antidepressant (SNRI)",
        "pKi": {"5HT2A": 5.0, "D2": 5.0, "NET": 6.4, "α2A": 5.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 5.0, "α1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 1.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.85, "Fr_hepatic": 0.15, "Risk_QTc": 0.35,
        "dosage": "Start 37.5 mg daily XR; target 75 mg - 150 mg daily.",
        "warnings": "Dose-dependent blood pressure elevation and sharp withdrawal discontinuation syndrome."
    },
    "Vortioxetine": {
        "category": "Multimodal Serotonin Modulator Antidepressant",
        "pKi": {"5HT2A": 7.60, "D2": 5.00, "NET": 5.00, "α2A": 5.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 5.00, "α1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.59, "Fr_hepatic": 0.41, "Risk_QTc": 0.15,
        "dosage": "Start 5.0 mg daily; target 5.0 mg - 10 mg daily in elderly.",
        "warnings": "Nausea risk; low anticholinergic burden and favorable cognitive safety profile."
    },
    "Bupropion": {
        "category": "Antidepressant (NDRI)",
        "pKi": {"5HT2A": 5.00, "D2": 5.20, "NET": 5.20, "α2A": 5.00, "NMDA": 5.00, "GABA-A": 5.00, "H1": 5.00, "α1": 5.00, "M1": 5.00},
        "Ar": {"5HT2A": 0.0, "D2": 0.5, "NET": 1.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.87, "Fr_hepatic": 0.13, "Risk_QTc": 0.20,
        "dosage": "Start 100 mg SR or 150 mg XL daily; max 150 mg daily in elderly.",
        "warnings": "Dose-dependent seizure risk; contraindications in seizure disorders, eating disorders, or sedative withdrawal."
    },
    "Trazodone": {
        "category": "Antidepressant (SARI)",
        "pKi": {"5HT2A": 7.80, "D2": 5.00, "NET": 5.00, "α2A": 6.40, "NMDA": 5.00, "GABA-A": 5.00, "H1": 7.50, "α1": 7.80, "M1": 5.00},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.70, "Fr_hepatic": 0.30, "Risk_QTc": 0.45,
        "dosage": "Start 12.5 mg - 25 mg hs / PRN; target 25 mg - 100 mg daily for nighttime agitation.",
        "warnings": "Orthostatic hypotension, priapism, and marked daytime sedation."
    },
    "Mirtazapine": {
        "category": "Pyridine Tetracyclic Antidepressant (NaSSA)",
        "pKi": {"5HT2A": 8.1, "D2": 5.0, "NET": 5.0, "α2A": 7.3, "NMDA": 0.0, "GABA-A": 0.0, "H1": 9.3, "α1": 6.0, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "α2A": 1.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.75, "Fr_hepatic": 0.25, "Risk_QTc": 0.30,
        "dosage": "Start 7.5 mg hs; target 15 mg - 30 mg hs (higher doses decrease sedating H1 effect).",
        "warnings": "Marked low-dose sedation and hyperphagia/weight gain; limited efficacy in primary agitation (SYMBAD trial)."
    },
    "Mianserin": {
        "category": "Benzene Tetracyclic Antidepressant (NaSSA)",
        "pKi": {"5HT2A": 8.0, "D2": 5.5, "NET": 6.0, "α2A": 7.2, "NMDA": 0.0, "GABA-A": 0.0, "H1": 9.0, "α1": 7.3, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.5, "α2A": 1.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.70, "Fr_hepatic": 0.30, "Risk_QTc": 0.35,
        "dosage": "Start 10 mg hs; titrate up to 30 mg - 60 mg hs.",
        "warnings": "Agranulocytosis/bone marrow depression (requires regular FBC), high sedation, and orthostasis."
    },
    "Lamotrigine": {
        "category": "Mood Stabilizer Anticonvulsant",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 6.5, "GABA-A": 6.0, "H1": 0.0, "α1": 0.0, "M1": 0.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.5, "GABA-A": 0.5},
        "Fr_renal": 0.94, "Fr_hepatic": 0.06, "Risk_QTc": 0.10,
        "dosage": "Start 25 mg daily; slow bi-weekly titration to target 100 mg - 200 mg daily.",
        "warnings": "CRITICAL: Stevens-Johnson Syndrome (SJS) and Toxic Epidermal Necrolysis (TEN). Discontinue at first sign of rash."
    },
    "Carbamazepine": {
        "category": "Mood Stabilizer Anticonvulsant",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 6.0, "GABA-A": 6.8, "H1": 0.0, "α1": 0.0, "M1": 0.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 1.0},
        "Fr_renal": 0.28, "Fr_hepatic": 0.72, "Risk_QTc": 0.30,
        "dosage": "Start 100 mg bid; target 200 mg - 600 mg daily in divided doses.",
        "warnings": "Aplastic anemia, agranulocytosis, severe dermatologic reactions, and potent CYP3A4 auto-induction."
    }
}

# -----------------------------------------------------------------------------
# 2. CONTINUOUS SIGMOIDAL MATH & ALGORITHMIC ENGINE
# -----------------------------------------------------------------------------
def sigmoid(x, k, x0):
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

def inverted_sigmoid(x, k, x0):
    return 1.0 / (1.0 + np.exp(k * (x - x0)))

def calculate_sigmoidal_lambdas(morse, sbp_drop, sas, qtc, egfr, lft_factor, dementia_subtype):
    lambda_H1 = sigmoid(morse, 0.08, 35.0)
    lambda_α1 = sigmoid(sbp_drop, 0.25, 15.0)
    lambda_D2_full = min(1.0, sigmoid(sas, 0.30, 8.0))
    
    if dementia_subtype in ["Dementia with Lewy Bodies (DLB)", "Parkinson's Disease Dementia (PDD)"]:
        lambda_D2_full = 1.0
        
    lambda_QTc = sigmoid(qtc, 0.05, 450.0)
    lambda_renal = inverted_sigmoid(egfr, 0.08, 45.0)
    lambda_hepatic = lft_factor
    
    return {
        "H1": lambda_H1,
        "α1": lambda_α1,
        "D2": lambda_D2_full,
        "QTc": lambda_QTc,
        "renal": lambda_renal,
        "hepatic": lambda_hepatic
    }

def calculate_p3_match_score(drug, drug_data, weights, lambdas, mmse_score, dementia_subtype, qtc_ms):
    pk = drug_data["pKi"]
    ar = drug_data["Ar"]
    
    hard_locked = False
    hard_lock_reason = ""
    
    if dementia_subtype in ["Dementia with Lewy Bodies (DLB)", "Parkinson's Disease Dementia (PDD)"] and ar["D2"] < 0:
        hard_locked = True
        hard_lock_reason = "Contraindicated: Full D2 antagonist in DLB/PDD etiology"
    elif qtc_ms > 500.0 and drug_data["Risk_QTc"] > 0.60:
        hard_locked = True
        hard_lock_reason = "Contraindicated: Severe baseline QTc (>500ms) with high QTc-risk agent"

    u_thera = (
        (weights["5HT2A"] * pk["5HT2A"] * ar["5HT2A"]) +
        (weights["D2"] * pk["D2"] * ar["D2"]) +
        (weights["NET"] * pk["NET"] * ar["NET"]) +
        (weights["α2A"] * pk["α2A"] * ar["α2A"]) +
        (weights["NMDA"] * pk["NMDA"] * ar["NMDA"]) +
        (weights["GABA-A"] * pk["GABA-A"] * ar["GABA-A"])
    )
    
    d2_risk = (lambdas["D2"] * pk["D2"]) if ar["D2"] < 0 else 0.0
    qtc_risk_deduction = lambdas["QTc"] * drug_data["Risk_QTc"] * 5.0
    
    u_risk = (
        (lambdas["H1"] * pk["H1"]) +
        (lambdas["α1"] * pk["α1"]) +
        d2_risk +
        qtc_risk_deduction
    )
    
    c_patient = 3.0 if mmse_score < 10 else (2.0 if mmse_score <= 20 else 1.0)
    p_acb = (c_patient * 2.0) if pk["M1"] >= 7.0 else 0.0
    p_organ = (lambdas["renal"] * drug_data["Fr_renal"] * 4.0) + (lambdas["hepatic"] * drug_data["Fr_hepatic"] * 4.0)
    
    m_j = u_thera - u_risk - p_acb - p_organ
    if hard_locked:
        m_j = -999.0
        
    p_sedation = min(95, int(sigmoid(pk["H1"] * lambdas["H1"], 0.5, 3.5) * 100))
    p_orthostasis = min(95, int(sigmoid(pk["α1"] * lambdas["α1"], 0.5, 3.5) * 100))
    p_eps = min(95, int(sigmoid(pk["D2"] * lambdas["D2"], 0.5, 4.0) * 100)) if ar["D2"] < 0 else 5
    
    return {
        "Drug": drug,
        "Category": drug_data["category"],
        "Net Score (Mj)": round(m_j, 1) if not hard_locked else -999.0,
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
        "Dosage": drug_data["dosage"],
        "Warnings": drug_data["warnings"]
    }

# -----------------------------------------------------------------------------
# 3. SINGLE-PAGE FRONTEND INPUTS (NO SIDEBAR)
# -----------------------------------------------------------------------------
NPI_MAPPING = {
    "0 - Absent (0)": 0.0,
    "1 - Mild (1-3)": 0.3,
    "2 - Moderate (4-7)": 0.7,
    "3 - Severe (8-12)": 1.0
}

HEPATIC_MAPPING = {
    "Normal / Unimpaired": 0.0,
    "Abnormal LFTs / Mild Impairment": 0.4,
    "Liver Cirrhosis / Moderate-Severe Impairment": 0.8
}

CAREGIVER_CONCERN_MAPPING = {
    "0 - None (No concern)": 0.5,
    "1 - Low Concern": 1.0,
    "2 - High Concern": 1.5
}

st.subheader("📋 Patient Clinical Parameters")

c_etiology, c_bio1, c_bio2 = st.columns(3)

with c_etiology:
    dementia_subtype = st.selectbox(
        "Dementia Subtype / Etiology",
        ["Alzheimer's Disease (AD)", "Dementia with Lewy Bodies (DLB)", "Parkinson's Disease Dementia (PDD)", "Vascular Dementia (VaD)", "Frontotemporal Dementia (FTD)"]
    )
    mmse_score = st.number_input("MMSE / MoCA Score (0-30)", 0, 30, 14)

with c_bio1:
    morse_score = st.number_input("Morse Fall Scale (0-125)", 0, 125, 40)
    sbp_drop = st.number_input("Standing SBP Drop (mmHg)", 0, 60, 12)
    sas_score = st.number_input("SAS Motor / EPS Score (0-40)", 0, 40, 4)

with c_bio2:
    qtc_ms = st.number_input("Baseline QTc Interval (ms)", 300, 600, 430)
    egfr_val = st.number_input("eGFR (mL/min)", 5, 120, 55)
    hepatic_status = st.selectbox("Hepatic Function Status", list(HEPATIC_MAPPING.keys()), index=0)
    lft_val = HEPATIC_MAPPING[hepatic_status]

st.markdown("---")
st.subheader("🎯 Target Symptom Severity (All 12 NPI Subscales)")

# 3-Column Layout for full 12 NPI Index Dropdowns
col_npi1, col_npi2, col_npi3 = st.columns(3)

with col_npi1:
    s_delusions_str = st.selectbox("Delusions (5-HT2A / D2 Target)", list(NPI_MAPPING.keys()), index=1)
    s_hallucinations_str = st.selectbox("Hallucinations (5-HT2A Target)", list(NPI_MAPPING.keys()), index=1)
    s_agitation_str = st.selectbox("Agitation / Aggression (D2 / α2A Target)", list(NPI_MAPPING.keys()), index=2)
    s_depression_str = st.selectbox("Depression / Dysphoria (NET / 5-HT Target)", list(NPI_MAPPING.keys()), index=1)

with col_npi2:
    s_anxiety_str = st.selectbox("Anxiety (GABA-A / 5-HT Target)", list(NPI_MAPPING.keys()), index=1)
    s_euphoria_str = st.selectbox("Euphoria / Elation (GABA-A Target)", list(NPI_MAPPING.keys()), index=0)
    s_apathy_str = st.selectbox("Apathy / Indifference (NET / NMDA Target)", list(NPI_MAPPING.keys()), index=1)
    s_disinhibition_str = st.selectbox("Disinhibition (GABA-A / 5-HT Target)", list(NPI_MAPPING.keys()), index=0)

with col_npi3:
    s_irritability_str = st.selectbox("Irritability / Lability (GABA-A / α2A Target)", list(NPI_MAPPING.keys()), index=1)
    s_motor_str = st.selectbox("Aberrant Motor Behavior (D2 / 5-HT2A Target)", list(NPI_MAPPING.keys()), index=0)
    s_sleep_str = st.selectbox("Sleep / Night-time Disturbances (5-HT2A Target)", list(NPI_MAPPING.keys()), index=1)
    s_appetite_str = st.selectbox("Appetite / Eating Changes (5-HT2A Target)", list(NPI_MAPPING.keys()), index=0)

st.markdown("---")
st.subheader("⚙️ Caregiver Priorities & Risk Concerns")
col_pref1, col_pref2 = st.columns(2)

with col_pref1:
    pref_sedation_str = st.selectbox("Caregiver Avoid-Sedation Concern Weight", list(CAREGIVER_CONCERN_MAPPING.keys()), index=1)
    pref_sedation = CAREGIVER_CONCERN_MAPPING[pref_sedation_str]

with col_pref2:
    pref_falls_str = st.selectbox("Caregiver Avoid-Fall Concern Weight", list(CAREGIVER_CONCERN_MAPPING.keys()), index=1)
    pref_falls = CAREGIVER_CONCERN_MAPPING[pref_falls_str]

# Map 12 NPI Symptom severity choices into receptor affinity target weights
v_delusions = NPI_MAPPING[s_delusions_str]
v_hallucinations = NPI_MAPPING[s_hallucinations_str]
v_agitation = NPI_MAPPING[s_agitation_str]
v_depression = NPI_MAPPING[s_depression_str]
v_anxiety = NPI_MAPPING[s_anxiety_str]
v_euphoria = NPI_MAPPING[s_euphoria_str]
v_apathy = NPI_MAPPING[s_apathy_str]
v_disinhibition = NPI_MAPPING[s_disinhibition_str]
v_irritability = NPI_MAPPING[s_irritability_str]
v_motor = NPI_MAPPING[s_motor_str]
v_sleep = NPI_MAPPING[s_sleep_str]
v_appetite = NPI_MAPPING[s_appetite_str]

weights = {
    "5HT2A": min(1.0, max(v_delusions * 0.7, v_hallucinations * 0.8, v_agitation * 0.5, v_disinhibition * 0.5, v_sleep * 0.6, v_motor * 0.5, v_appetite * 0.4)),
    "D2": min(1.0, max(v_agitation * 0.6, v_delusions * 0.5, v_hallucinations * 0.4, v_apathy * 0.3, v_motor * 0.5)),
    "NET": min(1.0, max(v_apathy * 0.8, v_depression * 0.7)),
    "α2A": min(1.0, max(v_agitation * 0.5, v_irritability * 0.4)),
    "NMDA": min(1.0, v_apathy * 0.4),
    "GABA-A": min(1.0, max(v_anxiety * 0.7, v_irritability * 0.6, v_euphoria * 0.5, v_disinhibition * 0.5))
}

# -----------------------------------------------------------------------------
# 4. RUN COMPUTATIONS & SPOTLIGHT DISPLAY
# -----------------------------------------------------------------------------
lambdas = calculate_sigmoidal_lambdas(
    morse=morse_score * pref_falls,
    sbp_drop=sbp_drop,
    sas=sas_score,
    qtc=qtc_ms,
    egfr=egfr_val,
    lft_factor=lft_val,
    dementia_subtype=dementia_subtype
)

# Apply caregiver sedation preference weight directly to H1 risk multiplier
lambdas["H1"] = min(1.0, lambdas["H1"] * pref_sedation)

results = [
    calculate_p3_match_score(drug, drug_data, weights, lambdas, mmse_score, dementia_subtype, qtc_ms)
    for drug, drug_data in DRUG_DATABASE.items()
]

results = sorted(results, key=lambda x: x["Raw_Mj"], reverse=True)
top_drug = results[0]

st.markdown("---")

# HIGH SPOTLIGHT HERO CARD FOR TOP CANDIDATE
if top_drug["Hard Locked"]:
    st.error("🚨 No suitable candidate found. All eligible agents triggered critical clinical safety hard-locks.")
else:
    st.markdown(
        f"""
        <div style="background-color: #d1e7dd; border-left: 8px solid #0f5132; padding: 22px; border-radius: 8px; margin-bottom: 15px;">
            <span style="font-size: 13px; color: #0f5132; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px;">
                🏆 Recommended Candidate Option
            </span>
            <h1 style="color: #0f5132; margin: 6px 0 0 0; font-size: 36px; font-weight: 800;">
                {top_drug['Drug']} <span style="font-size: 18px; font-weight: normal;">({top_drug['Category']})</span>
            </h1>
            <p style="color: #0f5132; font-size: 18px; margin: 10px 0 0 0;">
                Net Match Score (Mj): <strong>{top_drug['Net Score (Mj)']}</strong> &nbsp;|&nbsp; 
                Gain: <strong>+{top_drug['Therapeutic Gain']}</strong> &nbsp;|&nbsp; 
                Deductions: <strong>-{top_drug['Risk Deductions']}</strong> &nbsp;|&nbsp; 
                ACB Penalty: <strong>-{top_drug['ACB Penalty']}</strong> &nbsp;|&nbsp;
                Organ Penalty: <strong>-{top_drug['Organ Penalty']}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # DOSAGE SPECTRUM INFORMATION
    st.markdown(
        f"""
        <div style="background-color: #e2f0d9; border-left: 6px solid #385723; color: #274411; padding: 12px 18px; border-radius: 6px; margin-bottom: 15px; font-size: 15px;">
            <strong>💊 Recommended / Standard Dosage Spectrum:</strong> {top_drug['Dosage']}
        </div>
        """,
        unsafe_allow_html=True
    )

    # HIGH-CONTRAST EASY-TO-READ YELLOW CAUTION BOX
    st.markdown(
        f"""
        <div style="background-color: #fff3cd; border: 1px solid #ffebaa; border-left: 8px solid #ffc107; color: #856404; padding: 18px; border-radius: 6px; margin-bottom: 20px;">
            <h4 style="margin: 0 0 6px 0; color: #856404; font-weight: 800; font-size: 17px;">
                ⚠️ Clinical Warnings for {top_drug['Drug']}
            </h4>
            <p style="margin: 0; font-size: 15px; line-height: 1.5; color: #533f03;">
                {top_drug['Warnings']}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------------------------
# 5. EXPANDABLE TRAFFIC LIGHT DASHBOARD & METRICS
# -----------------------------------------------------------------------------
with st.expander("🚦 Candidate Dashboard", expanded=False):
    df_results = pd.DataFrame(results)
    
    df_results_display = df_results[[
        "Drug", "Category", "Net Score (Mj)", "Therapeutic Gain", 
        "Risk Deductions", "ACB Penalty", "Organ Penalty", 
        "Est. Sedation %", "Est. Orthostasis %", "Est. EPS %", "Dosage", "Lock Reason"
    ]].copy()

    def apply_traffic_lights(val):
        try:
            val_float = float(val)
            if val_float > 1.0:
                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif val_float >= -2.0:
                return 'background-color: #fff3cd; color: #856404;'
            else:
                return 'background-color: #f8d7da; color: #721c24;'
        except (ValueError, TypeError):
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'

    st.dataframe(
        df_results_display.style.map(apply_traffic_lights, subset=['Net Score (Mj)']),
        use_container_width=True
    )
    
    st.info("🚦 **Traffic Light:** Green = Optimal Match ($M_j > 1.0$) | Yellow = Proceed with Caution ($-2.0 \\le M_j \\le 1.0$) | Red = High Risk / Contraindicated ($M_j < -2.0$)")

with st.expander("⚙️ Calculated Sigmoidal Risk(λ)", expanded=False):
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.write(f"- **Fall Risk (λH1):** `{lambdas['H1']:.2f}`")
    col_s1.write(f"- **Orthostasis Risk (λα1):** `{lambdas['α1']:.2f}`")
    col_s2.write(f"- **Motor EPS Risk (λD2):** `{lambdas['D2']:.2f}`")
    col_s2.write(f"- **Cardiac QTc Risk (λQTc):** `{lambdas['QTc']:.2f}`")
    col_s3.write(f"- **Renal Penalty (λrenal):** `{lambdas['renal']:.2f}`")
    col_s3.write(f"- **Hepatic Penalty (λhepatic):** `{lambdas['hepatic']:.2f}`")

with st.expander("🔄 Cross-Titration & Switching Protocol", expanded=False):
    st.markdown("""
    **Patient Transition Protocol**
    When transitioning from a high-affinity D2 antagonist (e.g. Risperidone) to a D2 partial agonist or non-dopaminergic agent:
    
    * **Week 1:** Reduce prior agent dose by 50%. Initiate target agent at baseline low dose.
    * **Week 2:** Maintain taper. Monitor for cholinergic rebound or withdrawal psychosis.
    * **Week 3:** Discontinue prior agent completely. Titrate target agent to optimal therapeutic dosage.
    """)
# -----------------------------------------------------------------------------
# EXPANDABLE RATIONALE, FORMULAS & ALGORITHMIC THINKING
# -----------------------------------------------------------------------------
with st.expander("🧮 Formulas & Clinical Rationale", expanded=False):
    st.markdown("""
    This decision-support tool based on balancing therapeutic receptor targeting against patient-specific physiological vulnerability factors.

    ---

    ### 1. Neurochemical Pathogenetic Coupling ($v_s \rightarrow w_r$)
    **Algorithmic Concept:**
    Rather than treating symptoms as isolated clinical categories, the tool aiming all 12 NPI subscales ($v_s \in [0.0, 1.0]$) to their underlying neurochemical drivers. To prevent scaling distortion, the target weight ($w_r$) for any receptor ($r$) uses a non-linear maximum-affinity coupling function:

    $$w_r = \min\left(1.0, \max_{s}\left(v_s \cdot \kappa_{s,r}\right)\right)$$

    *   **Coupling Coefficients ($\kappa_{s,r}$):** Represent the relative pathogenetic contribution of receptor system $r$ to symptom $s$. For example, hallucinations rely heavily on cortical $5\text{-HT}_{2\text{A}}$ hyperfunction ($\kappa = 0.8$), whereas apathy is primarily mediated via noradrenergic ($\kappa = 0.8$) and glutamatergic pathways ($\kappa = 0.4$).

    ---

    ### 2. Sigmoidal Patient Vulnerability Scaling ($\lambda_r$)
    **Algorithmic Concept:**
    Static drug contraindications fail to capture continuous patient physiological decline. The tool transforms continuous clinical biomarkers ($x$) into normalized risk scalars ($\lambda_r \in [0.0, 1.0]$) using sigmoidal functions:

    $$\lambda(x) = \frac{1}{1 + e^{-k(x - x_0)}}$$

    *   **Fall & Sedation Risk ($\lambda_{\text{H1}}$):** Driven by Morse Fall Scale score ($x_0 = 35.0, k = 0.08$) and scaled by caregiver concerns.
    *   **Orthostasis Risk ($\lambda_{\alpha1}$):** Driven by standing Systolic BP drop in mmHg ($x_0 = 15.0, k = 0.25$).
    *   **Extrapyramidal Risk ($\lambda_{\text{D2}}$):** Driven by Simpson-Angus Scale (SAS) motor score ($x_0 = 8.0, k = 0.30$), with $\lambda_{\text{D2}} = 1.0$ hard-coded for DLB/PDD etiologies due to extreme neuroleptic sensitivity.
    *   **Cardiotoxicity Risk ($\lambda_{\text{QTc}}$):** Driven by baseline QTc interval in ms ($x_0 = 450.0, k = 0.05$).
    *   **Organ Clearance Penalties:** Inverted sigmoid for eGFR ($\lambda_{\text{renal}}$) and discrete clinical stratification for liver impairment ($\lambda_{\text{hepatic}}$).

    ---

    ### 3. Net Utility Match Score Computation ($M_j$)
    **Algorithmic Concept:**
    For each candidate drug ($j$), the overall match score ($M_j$) combines therapeutic gain ($U_{\text{thera}}$), dynamic risk deductions ($U_{\text{risk}}$), anticholinergic burden ($P_{\text{ACB}}$), and clearance organ impairment penalties ($P_{\text{organ}}$):

    $$M_j = U_{\text{thera}} - U_{\text{risk}} - P_{\text{ACB}} - P_{\text{organ}}$$

    #### Mathematical Breakdown:
    1. **Therapeutic Gain ($U_{\text{thera}}$):**
       $$U_{\text{thera}} = \sum_{r} \left( w_r \cdot pK_{i,r} \cdot A_r \right)$$
       *Where $pK_{i,r}$ is binding affinity ($-\log_{10} K_i$) and $A_r \in \{-1.0, 0.0, 0.5, 1.0\}$ represents intrinsic efficacy (antagonist, neutral, partial agonist, full agonist)*.

    2. **Risk Deductions ($U_{\text{risk}}$):**
       $$U_{\text{risk}} = (\lambda_{\text{H1}} \cdot pK_{i,\text{H1}}) + (\lambda_{\alpha1} \cdot pK_{i,\alpha1}) + (\lambda_{\text{D2}} \cdot pK_{i,\text{D2}} \cdot \mathbb{I}_{\text{Antagonist}}) + 5.0(\lambda_{\text{QTc}} \cdot \text{Risk}_{\text{QTc}})$$
       *Dopaminergic risk applies exclusively to full $D_2$ antagonists ($A_{\text{D2}} < 0$)*.

    3. **Anticholinergic Cognitive Burden Penalty ($P_{\text{ACB}}$):**
       $$P_{\text{ACB}} = C_{\text{patient}} \times 2.0 \quad \text{if } pK_{i,\text{M1}} \ge 7.0 \text{ else } 0.0$$
       *Where cognitive vulnerability coefficient $C_{\text{patient}} = 3.0$ if MMSE < 10, $2.0$ if MMSE 10–20, and $1.0$ if MMSE > 20*.

    4. **Organ Clearance Penalty ($P_{\text{organ}}$):**
       $$P_{\text{organ}} = 4.0 \left( \lambda_{\text{renal}} \cdot \text{Fr}_{\text{renal}} + \lambda_{\text{hepatic}} \cdot \text{Fr}_{\text{hepatic}} \right)$$
       *Penalizes drugs heavily reliant on impaired elimination pathways based on renal/hepatic elimination fractions ($\text{Fr}$)*.

    ---

    ### 4. Safety Hard-Lock Protocol
    **Algorithmic Concept:**
    Regardless of a drug's therapeutic score, absolute clinical contraindications trigger an unconditional lock ($M_j = -999.0$):
    *   **Etiology Hard-Lock:** Full $D_2$ antagonists in DLB or PDD patients.
    *   **Cardiac Hard-Lock:** High QTc-risk agents ($\text{Risk}_{\text{QTc}} > 0.60$) when baseline QTc $> 500\text{ ms}$.

    ---

    ### 5. Adverse Event Probability Heuristics
    **Algorithmic Concept:**
    Estimated side-effect probabilities ($P_{\text{event}}$) translate receptor occupancy and baseline vulnerability into clinically readable percentages using a bounded logistic function:

    $$P_{\text{event}} = \min\left(95\%, \text{int}\left( \frac{100}{1 + e^{-0.5(pK_i \cdot \lambda - 3.5)}} \right)\right)$$
    """)

# -----------------------------------------------------------------------------
# 6. CITATIONS & ALGORITHMIC REFERENCES
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
    )

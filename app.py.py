import streamlit as st
import numpy as np
import pandas as pd
import itertools

st.set_page_config(page_title="BPSD Compass Prototype (P5)", layout="wide")
st.title("BPSD Compass Prototype (P5)")
st.caption("Parameter-driven decision support tools")

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

# Initialize Session State for Interactive Rule-Out
if "ruled_out" not in st.session_state:
    st.session_state.ruled_out = set()

# -----------------------------------------------------------------------------
# 1. PHARMACODYNAMIC DATABASE WITH DOSAGE SPECTRUM
# -----------------------------------------------------------------------------
DRUG_DATABASE = {
    "Brexpiprazole": {
        "category": "Atypical Antipsychotic",
        "pKi": {"5HT2A": 8.7, "D2": 9.5, "NET": 5.0, "α2A": 7.4, "NMDA": 0.0, "GABA-A": 0.0, "H1": 7.1, "α1": 8.0, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": 1.0, "NET": 0.0, "α2A": 1.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.14, "Fr_hepatic": 0.86, "Risk_QTc": 0.20,
        "dosage": "Start 0.5 mg PO OD, max 2 mg/day for agitation.",
        "warnings": "Akathisia and impulse-control disorders. Monitor elderly closely."
    },
    "Pimavanserin": {
        "category": "Atypical Antipsychotic",
        "pKi": {"5HT2A": 9.3, "D2": 5.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 5.0, "α1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.06, "Fr_hepatic": 0.94, "Risk_QTc": 0.40,
        "dosage": "Standard dose: 34 mg PO OD or 10 mg PO OD in CYP3A4 inhibitor co-administration.",
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
        "warnings": "Severe metabolic syndrome, weight gain, sedation, and anticholinergic cognitive impairment."
    },
    "Haloperidol": {
        "category": "Typical Antipsychotic",
        "pKi": {"5HT2A": 7.2, "D2": 9.2, "NET": 5.0, "α2A": 5.0, "NMDA": 0.0, "GABA-A": 0.0, "H1": 6.0, "α1": 7.3, "M1": 5.0},
        "Ar": {"5HT2A": 0.0, "D2": -1.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.15, "Fr_hepatic": 0.85, "Risk_QTc": 0.85,
        "dosage": "Start as low as possible, 0.25-0.5 mg PO OD or PRN - max 2 mg/day.",
        "warnings": "Torsades de Pointes, Extrapyramidal Symptoms (EPS), and Tardive Dyskinesia."
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
        "warnings": "Hepatotoxicity, pancreatitis, thrombocytopenia; monitor LFTs, CBC required."
    },
    "Gabapentin": {
        "category": "GABA analogue Anticonvulsant",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 6.8, "H1": 5.0, "α1": 5.0, "M1": 5.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 1.0},
        "Fr_renal": 1.00, "Fr_hepatic": 0.00, "Risk_QTc": 0.05,
        "dosage": "Start 100 mg tid; slow titrations up to 300 mg - 600 mg tid based on renal clearance.",
        "warnings": "Respiratory depression risk with CNS depressants/opioids; renal dose adjustment required."
    },
    "Memantine": {
        "category": "Cognitive Enhancer",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 7.5, "GABA-A": 0.0, "H1": 0.0, "α1": 0.0, "M1": 0.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 1.0, "GABA-A": 0.0},
        "Fr_renal": 0.80, "Fr_hepatic": 0.20, "Risk_QTc": 0.05,
        "dosage": "Start 5 mg daily; titrate by 5 mg weekly to target 10 mg bid (max 10 mg daily if eGFR < 30).",
        "warnings": "Dose adjustment in severe renal impairment (eGFR < 30 ml/min)."
    },
    "Amitriptyline": {
        "category": "Tricyclic Antidepressant (TCA)",
        "pKi": {"5HT2A": 8.1, "D2": 5.5, "NET": 7.7, "α2A": 6.8, "NMDA": 0.0, "GABA-A": 0.0, "H1": 8.9, "α1": 8.0, "M1": 8.8},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 1.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.05, "Fr_hepatic": 0.95, "Risk_QTc": 0.70,
        "dosage": "Generally avoid in dementia. 5-10 mg PO hs if strictly indicated).",
        "warnings": "Severe anticholinergic toxicity, fall risk, cognitive decline, and cardiotoxicity."
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
        "warnings": "Orthostatic hypotension, priapism, and Daytime sedation."
    },
    "Mirtazapine": {
        "category": "Pyridine Tetracyclic Antidepressant (NaSSA)",
        "pKi": {"5HT2A": 8.1, "D2": 5.0, "NET": 5.0, "α2A": 7.3, "NMDA": 0.0, "GABA-A": 0.0, "H1": 9.3, "α1": 6.0, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.0, "α2A": 1.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.75, "Fr_hepatic": 0.25, "Risk_QTc": 0.30,
        "dosage": "Start 7.5 mg hs; target 15 mg - 30 mg hs (higher doses decrease sedating H1 effect).",
        "warnings": "Low-dose sedation and weight gain; limited efficacy in primary agitation (SYMBAD trial)."
    },
    "Mianserin": {
        "category": "Benzene Tetracyclic Antidepressant (NaSSA)",
        "pKi": {"5HT2A": 8.0, "D2": 5.5, "NET": 6.0, "α2A": 7.2, "NMDA": 0.0, "GABA-A": 0.0, "H1": 9.0, "α1": 7.3, "M1": 5.0},
        "Ar": {"5HT2A": 1.0, "D2": 0.0, "NET": 0.5, "α2A": 1.0, "NMDA": 0.0, "GABA-A": 0.0},
        "Fr_renal": 0.70, "Fr_hepatic": 0.30, "Risk_QTc": 0.35,
        "dosage": "Start 10 mg hs; titrate up to 30 mg - 60 mg hs.",
        "warnings": "Agranulocytosis/bone marrow depression, high sedation, and orthostasis."
    },
    "Lamotrigine": {
        "category": "Mood Stabilizer Anticonvulsant",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 6.5, "GABA-A": 6.0, "H1": 0.0, "α1": 0.0, "M1": 0.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.5, "GABA-A": 0.5},
        "Fr_renal": 0.94, "Fr_hepatic": 0.06, "Risk_QTc": 0.10,
        "dosage": "Start 25 mg/day; slow titration to target 100 mg - 200 mg daily.",
        "warnings": "Stevens-Johnson Syndrome (SJS) and Toxic Epidermal Necrolysis (TEN). Discontinue at first sign of rash."
    },
    "Carbamazepine": {
        "category": "Mood Stabilizer Anticonvulsant",
        "pKi": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 6.0, "GABA-A": 6.8, "H1": 0.0, "α1": 0.0, "M1": 0.0},
        "Ar": {"5HT2A": 0.0, "D2": 0.0, "NET": 0.0, "α2A": 0.0, "NMDA": 0.0, "GABA-A": 1.0},
        "Fr_renal": 0.28, "Fr_hepatic": 0.72, "Risk_QTc": 0.30,
        "dosage": "Start 100 mg bid; target 200 mg - 600 mg daily in divided doses.",
        "warnings": "Aplastic anemia, agranulocytosis, severe dermatologic reactions, potent CYP3A4 inducer."
    }
}

# -----------------------------------------------------------------------------
# 2. FUNCTIONS & SIGMOIDAL SCALING
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

def calculate_p3_match_score(drug, drug_data, weights, lambdas, mmse_score, dementia_subtype, qtc_ms, prior_history):
    pk = drug_data["pKi"]
    ar = drug_data["Ar"]
    
    hard_locked = False
    hard_lock_reason = ""
    history_penalty = 0.0
    clinical_note = ""
    
    # Check Manual Clinician Rule-Out Statement
    if drug in st.session_state.ruled_out:
        hard_locked = True
        hard_lock_reason = "Ruled Out by Clinician"

    # Prior History Correlation Logic
    if drug in prior_history:
        outcome = prior_history[drug]["outcome"]
        dose = prior_history[drug]["dosage"]
        if outcome == "Treatment Failure / Ineffective":
            history_penalty += 5.0
            clinical_note = f"Prior trial at {dose} resulted in failure. Higher dose or mechanism switch advised."
        elif outcome == "Severe Adverse Effects / Intolerant":
            hard_locked = True
            hard_lock_reason = f"Historical Intolerance: Discontinued due to adverse effects ({dose})."
        elif outcome == "Partial Response / Tolerated":
            clinical_note = f"Prior partial benefit noted at {dose}. Consider optimizing dose before class switch."

    # Etiology & QTc Safety Hard-Locks
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
    
    m_j = u_thera - u_risk - p_acb - p_organ - history_penalty
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
        "History Penalty": round(history_penalty, 1),
        "Sedation risk": p_sedation,
        "Orthostatic risk": p_orthostasis,
        "EPS risk": p_eps,
        "Hard Locked": hard_locked,
        "Lock Reason": hard_lock_reason,
        "Clinical Correlation Note": clinical_note,
        "Dosage": drug_data["dosage"],
        "Warnings": drug_data["warnings"]
    }

def generate_cross_titration_schedule(prior_drug, target_drug):
    if not prior_drug or prior_drug == target_drug:
        return None
    return pd.DataFrame([
        {"Phase": "Days 1–3", "Prior Agent Action": f"Reduce {prior_drug} to 75% dose", "New Agent Action": f"Initiate {target_drug} at starting dose", "Monitoring": "Vital signs, orthostasis"},
        {"Phase": "Days 4–7", "Prior Agent Action": f"Taper {prior_drug} to 50% dose", "New Agent Action": f"Maintain {target_drug} starting dose", "Monitoring": "Sedation & fall precautions"},
        {"Phase": "Days 8–11", "Prior Agent Action": f"Taper {prior_drug} to 25% dose", "New Agent Action": f"Titrate {target_drug} toward target dose", "Monitoring": "NPI symptom trajectory"},
        {"Phase": "Day 12+", "Prior Agent Action": f"Discontinue {prior_drug}", "New Agent Action": f"Optimize {target_drug} target dose", "Monitoring": "Full CGI-I / NPI-Q re-assessment"}
    ])
# =============================================================================
# HELPER FUNCTIONS & SIGMOIDAL SCALARS
# =============================================================================

def sigmoid(x: float, k: float, x0: float) -> float:
    """Standard continuous sigmoidal transfer function."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))

def inverted_sigmoid(x: float, k: float, x0: float) -> float:
    """Inverted sigmoidal function for metrics where higher values reduce risk."""
    return 1.0 / (1.0 + np.exp(k * (x - x0)))


# =============================================================================
# MODULE 1: SPECIAL CONDITIONS RISK WEIGHTING ENGINE
# =============================================================================

def calculate_special_condition_penalties(
    drug: dict,
    patient_profile: dict
) -> dict:
    """
    Computes dynamic risk penalties for Epilepsy, NCDs, and Extreme Age Groups.
    
    patient_profile keys expected:
      - 'seizure_freq_year': float (seizures/year)
      - 'active_aeds': list of str (e.g., ['carbamazepine', 'valproate'])
      - 'hba1c': float (%)
      - 'bmi': float (kg/m^2)
      - 'sbp_drop': float (mmHg)
      - 'has_thyroid_dysfunction': bool
      - 'age': float (years)
      - 'frailty_score': float (0.0 to 1.0 scale)
      - 'acb_score': float
    """
    
    # -------------------------------------------------------------------------
    # 1. Epilepsy & AED Interaction Penalty (P_epilepsy)
    # -------------------------------------------------------------------------
    seizure_freq = patient_profile.get('seizure_freq_year', 0.0)
    active_aeds = patient_profile.get('active_aeds', [])
    
    # Sigmoidal scalar centered at midpoint x0 = 1.0 seizure/year
    lambda_seizure = sigmoid(seizure_freq, k=0.8, x0=1.0)
    convulsant_index = drug.get('convulsant_index', 0.0)  # Range 0.0 to 1.0
    
    # DDI Penalty matrix lookup against active AEDs (e.g., enzymatic induction/inhibition)
    ddi_matrix = drug.get('aed_ddi_penalties', {})
    ddi_penalty_sum = sum([ddi_matrix.get(aed.lower(), 0.0) for aed in active_aeds])
    
    P_epilepsy = (lambda_seizure * convulsant_index * 8.0) + ddi_penalty_sum

    # -------------------------------------------------------------------------
    # 2. Non-Communicable Diseases (NCD) Penalty (P_NCD)
    # -------------------------------------------------------------------------
    hba1c = patient_profile.get('hba1c', 5.7)
    bmi = patient_profile.get('bmi', 22.0)
    sbp_drop = patient_profile.get('sbp_drop', 0.0)
    has_thyroid = patient_profile.get('has_thyroid_dysfunction', False)
    
    # Metabolic Risk Scalar
    lambda_hba1c = sigmoid(hba1c, k=1.0, x0=8.0)
    lambda_bmi = sigmoid(bmi, k=0.15, x0=30.0)
    lambda_metabolic = 0.5 * (lambda_hba1c + lambda_bmi)
    
    # Binding affinities (pK_i) for metabolic risk receptors
    pKi_H1 = drug.get('pK_i', {}).get('H1', 0.0)
    pKi_5HT2C = drug.get('pK_i', {}).get('5HT2C', 0.0)
    P_metabolic = lambda_metabolic * (pKi_H1 + pKi_5HT2C)
    
    # Vascular/Hypertension Risk Scalar (alpha-1 blockade mapping)
    lambda_HT = sigmoid(sbp_drop, k=0.25, x0=15.0)
    pKi_alpha1 = drug.get('pK_i', {}).get('alpha1', 0.0)
    P_HT = lambda_HT * pKi_alpha1
    
    # Endocrine/Thyroid Risk Scalar (QTc Risk amplification)
    lambda_thyroid = 1.5 if has_thyroid else 0.0
    qtc_risk = drug.get('qtc_risk_score', 0.0)  # Range 0.0 to 1.0
    P_thyroid = lambda_thyroid * qtc_risk * 3.0
    
    P_NCD = P_metabolic + P_HT + P_thyroid

    # -------------------------------------------------------------------------
    # 3. Extreme Age Group Penalty (P_age)
    # -------------------------------------------------------------------------
    age = patient_profile.get('age', 65.0)
    frailty = patient_profile.get('frailty_score', 0.2)
    acb = patient_profile.get('acb_score', 0.0)
    gamma = 1.4  # Exponential scaling for advanced age
    
    if age > 75.0:
        age_factor = (age / 80.0) ** gamma
        P_age = age_factor * ((frailty * acb * 2.0) + (frailty * pKi_H1 * 1.5))
    else:
        P_age = 0.0

    return {
        "P_epilepsy": P_epilepsy,
        "P_NCD": P_NCD,
        "P_age": P_age,
        "P_special_total": P_epilepsy + P_NCD + P_age
    }
	
# -----------------------------------------------------------------------------
# 3. CLINICAL INPUTS
# -----------------------------------------------------------------------------
NPI_MAPPING = {
    "0 - Absent": 0.0,
    "1 - Mild": 0.3,
    "2 - Moderate": 0.7,
    "3 - Severe": 1.0
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
    egfr_val = st.number_input("eGFR (ml/min)", 5, 120, 55)
    hepatic_status = st.selectbox("Hepatic Function Status", list(HEPATIC_MAPPING.keys()), index=0)
    lft_val = HEPATIC_MAPPING[hepatic_status]

st.markdown("---")
st.subheader("💊 Prior Psychotropic Exposure & Regimen")

col_prior1, col_prior2 = st.columns(2)

with col_prior1:
    prior_drugs = st.multiselect(
        "Select Active / Prior Psychotropic Medications",
        options=list(DRUG_DATABASE.keys())
    )

prior_history = {}
if prior_drugs:
    with col_prior2:
        for drug in prior_drugs:
            c_d1, c_d2 = st.columns(2)
            with c_d1:
                dosage = st.text_input(f"Dose for {drug}", value=" mg/day", key=f"dose_{drug}")
            with c_d2:
                outcome = st.selectbox(
                    f"Outcome for {drug}",
                    ["Partial Response / Tolerated", "Treatment Failure / Ineffective", "Severe Adverse Effects / Intolerant"],
                    key=f"outcome_{drug}"
                )
            prior_history[drug] = {"dosage": dosage, "outcome": outcome}

st.markdown("---")
st.subheader("🎯 Target Symptom Severity (12 NPI Subscales)")

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
st.subheader("⚙️ Caregiver Priorities & Shared Decision (PETRUSHKA Model)")
col_pref1, col_pref2 = st.columns(2)

with col_pref1:
    pref_sedation_str = st.selectbox("Caregiver Avoid-Sedation Weight", list(CAREGIVER_CONCERN_MAPPING.keys()), index=1)
    pref_sedation = CAREGIVER_CONCERN_MAPPING[pref_sedation_str]

with col_pref2:
    pref_falls_str = st.selectbox("Caregiver Avoid-Fall Concern Weight", list(CAREGIVER_CONCERN_MAPPING.keys()), index=1)
    pref_falls = CAREGIVER_CONCERN_MAPPING[pref_falls_str]

# Receptor Weight
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
# 4. HERO SPOTLIGHT
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

lambdas["H1"] = min(1.0, lambdas["H1"] * pref_sedation)

results = [
    calculate_p3_match_score(drug, drug_data, weights, lambdas, mmse_score, dementia_subtype, qtc_ms, prior_history)
    for drug, drug_data in DRUG_DATABASE.items()
]

results = sorted(results, key=lambda x: x["Raw_Mj"], reverse=True)
top_drug = results[0]

# Multi-Symptom Phenotype Clustering Display
st.markdown("---")
st.subheader("📊 Phenotype Cluster Analysis (CCSMH Model)")
p_agitation_psychosis = (v_agitation + v_delusions + v_hallucinations) / 3.0
p_affective = (v_depression + v_anxiety + v_irritability) / 3.0
p_apathy_executive = (v_apathy + v_disinhibition) / 2.0

col_ph1, col_ph2, col_ph3 = st.columns(3)
col_ph1.metric("Agitation-Psychosis Cluster", f"{p_agitation_psychosis*100:.0f}%")
col_ph2.metric("Affective-Lability Cluster", f"{p_affective*100:.0f}%")
col_ph3.metric("Apathy-Executive Cluster", f"{p_apathy_executive*100:.0f}%")

st.markdown("---")

if st.session_state.ruled_out:
    col_ro_btn1, col_ro_btn2 = st.columns([4, 1])
    with col_ro_btn1:
        st.info(f"🚫 **Ruled Out Agents ({len(st.session_state.ruled_out)}):** {', '.join(st.session_state.ruled_out)}")
    with col_ro_btn2:
        if st.button("Reset Rule-Outs"):
            st.session_state.ruled_out.clear()
            st.rerun()

if top_drug["Hard Locked"]:
    st.error("🚨 No suitable candidate found. All eligible agents triggered critical clinical safety hard-locks or manual rule-outs.")
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
                Organ Penalty: <strong>-{top_drug['Organ Penalty']}</strong> &nbsp;|&nbsp;
                History Penalty: <strong>-{top_drug['History Penalty']}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        f"""
        <div style="background-color: #e2f0d9; border-left: 6px solid #385723; color: #274411; padding: 12px 18px; border-radius: 6px; margin-bottom: 15px; font-size: 15px;">
            <strong>💊 Recommended Dosage Spectrum:</strong> {top_drug['Dosage']}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style="background-color: #fff3cd; border: 1px solid #ffebaa; border-left: 8px solid #ffc107; color: #856404; padding: 18px; border-radius: 6px; margin-bottom: 15px;">
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

    # Clinician Rule-Out Checkbox
    rule_out_flag = st.checkbox(
        f"🚫 Rule Out **{top_drug['Drug']}** (Clinical contraindication / Tailored made preference)",
        key=f"chk_ruleout_{top_drug['Drug']}"
    )
    if rule_out_flag:
        st.session_state.ruled_out.add(top_drug["Drug"])
        st.rerun()

# -----------------------------------------------------------------------------
# 5. EXPANDED CLINICAL DASHBOARD & CROSS-TITRATION ENGINE
# -----------------------------------------------------------------------------
with st.expander("📊 Estimated Side-Effect Probabilities (PETRUSHKA based model)", expanded=True):
    st.markdown("#### Patient-Specific Risk Likelihood Output")
    col_vis1, col_vis2, col_vis3 = st.columns(3)
    
    with col_vis1:
        st.write("**Estimated Sedation Risk:**")
        st.progress(top_drug["Sedation risk"])
        st.caption(f"Likelihood: {top_drug['Sedation risk']}%")
        
    with col_vis2:
        st.write("**Estimated Orthostatic Risk:**")
        st.progress(top_drug["Orthostatic risk"])
        st.caption(f"Likelihood: {top_drug['Orthostatic risk']}%")

    with col_vis3:
        st.write("**Estimated EPS Risk:**")
        st.progress(top_drug["EPS risk"])
        st.caption(f"Likelihood: {top_drug['EPS risk']}%")

with st.expander("🔄 Sequential Cross-Titration", expanded=True):
    if prior_drugs:
        distinct_priors = [d for d in prior_drugs if d != top_drug["Drug"]]
        
        # Check if top recommended drug is ALREADY in the active regimen
        if top_drug["Drug"] in prior_drugs:
            st.info(
                f"💡 **Dose Optimization Protocol:** **{top_drug['Drug']}** is already part of the patient's active regimen. "
                f"Rather than cross-tapering, evaluate optimizing current dosage toward targeted spectrum (**{top_drug['Dosage']}**)."
            )
        
        if distinct_priors:
            prior_selected = st.selectbox(
                "Select distinct prior agent to cross-taper from:",
                options=distinct_priors,
                key="cross_taper_selector"
            )
            st.markdown(f"### Cross-Titration: Taper **{prior_selected}** $\\rightarrow$ Initiate **{top_drug['Drug']}**")
            tt_df = generate_cross_titration_schedule(prior_selected, top_drug["Drug"])
            if tt_df is not None:
                st.table(tt_df)
        elif top_drug["Drug"] not in prior_drugs:
            st.info("Treatment Naive: Initiate top candidate at starting dose without cross-tapering.")
    else:
        st.info("Treatment Naive: Initiate top candidate at starting dose without cross-tapering.")

with st.expander("🚦 Dashboard & Ranking", expanded=False):
    df_results = pd.DataFrame(results)
    
    df_results_display = df_results[[
        "Drug", "Category", "Net Score (Mj)", "Therapeutic Gain", 
        "Risk Deductions", "ACB Penalty", "Organ Penalty", "History Penalty",
        "Dosage", "Lock Reason"
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
# -----------------------------------------------------------------------------
# EXPANDABLE RATIONALE, FORMULAS & ALGORITHMIC THINKING MODEL
# -----------------------------------------------------------------------------
with st.expander("🧮 Algorithmic Architecture & Clinical Rationale", expanded=False):
    st.markdown(r"""
    This decision-support tool employs a **Multi-Criteria Utility Optimization Model** balancing therapeutic receptor targeting against patient-specific vulnerabilities and prior treatment history.

    ---

    **1. Neurological-Treatment Coupling ($v_s \rightarrow w_r$)**
    
    Rather than treating symptoms as isolated clinical presentation, this tool aims mapping NPI subscales ($v_s \in [0.0, 1.0]$) to patient's underlying neurological underlying. To prevent scaling bias, target weighting ($w_r$) for each receptors focused ($r$) uses a non-linear coupling function:

    $$w_r = \min\left(1.0, \max_{s}\left(v_s \cdot \kappa_{s,r}\right)\right)$$

    * **Coupling Coefficients ($\kappa_{s,r}$):** Represent relativity of each contributions, e.g. Hallucinations rely heavily on cortical $5\text{-HT}_{2\text{A}}$ hyperfunction ($\kappa = 0.8$), whereas apathy is relied via noradrenergic ($\kappa = 0.8$) and glutamatergic pathways ($\kappa = 0.4$).

    ---

    **2. Sigmoidal Patient Vulnerability Scaling ($\lambda_r$)**
    
    Static contraindications fail to represent continuous physiological decline. Continuous biomarkers ($x$) are used to normalized risk scalars ($\lambda_r \in [0.0, 1.0]$) using sigmoidal transfer functions:

    $$\lambda(x) = \frac{1}{1 + e^{-k(x - x_0)}}$$

    * **Fall & Sedation Risk ($\lambda_{\text{H1}}$):** Analyed via Morse Fall Scale score ($x_0 = 35.0, k = 0.08$) and scaled by caregiver concern preference weights.
    * **Orthostatic Risk ($\lambda_{\alpha1}$):** Implied by standing Systolic BP drop in mmHg ($x_0 = 15.0, k = 0.25$).
    * **Extrapyramidal Risk ($\lambda_{\text{D2}}$):** Based on Simpson-Angus Scale (SAS) score ($x_0 = 8.0, k = 0.30$), with $\lambda_{\text{D2}} = 1.0$ hard-locked for DLB/PDD etiologies.
    * **Cardiotoxicity Risk ($\lambda_{\text{QTc}}$):** Evaluation of baseline QTc interval ($x_0 = 450.0\text{ ms}, k = 0.05$).
    * **Organ Clearance Penalties:** Inverted sigmoid for eGFR ($\lambda_{\text{renal}}$) and discrete mapping for hepatic impairment ($\lambda_{\text{hepatic}}$).

    ---

    **3. Net Utility Match Score Calculation ($M_j$)**
    
    For each candidate agent ($j$), the match score ($M_j$) combines therapeutic gain ($U_{\text{thera}}$), dynamic risk deductions ($U_{\text{risk}}$), anticholinergic burden ($P_{\text{ACB}}$), clearance organ penalties ($P_{\text{organ}}$), and prior treatment history penalties ($P_{\text{history}}$):

    $$M_j = U_{\text{thera}} - U_{\text{risk}} - P_{\text{ACB}} - P_{\text{organ}} - P_{\text{history}}$$

    * **Therapeutic Gain ($U_{\text{thera}}$):**
      $$U_{\text{thera}} = \sum_{r} \left( w_r \cdot pK_{i,r} \cdot A_r \right)$$
      *Where $pK_{i,r}$ is binding affinity ($-\log_{10} K_i$) and $A_r \in \{-1.0, 0.0, 0.5, 1.0\}$ represents intrinsic efficacy (antagonist, neutral, partial agonist, full agonist).*

    * **Dynamic Risk Deductions ($U_{\text{risk}}$):**
      $$U_{\text{risk}} = (\lambda_{\text{H1}} \cdot pK_{i,\text{H1}}) + (\lambda_{\alpha1} \cdot pK_{i,\alpha1}) + (\lambda_{\text{D2}} \cdot pK_{i,\text{D2}} \cdot \mathbb{I}_{\text{Antagonist}}) + 5.0(\lambda_{\text{QTc}} \cdot \text{Risk}_{\text{QTc}})$$
      *Dopaminergic risk deduction applies to full $D_2$ antagonists ($A_{\text{D2}} < 0$).*

    * **Anticholinergic Cognitive Burden Penalty ($P_{\text{ACB}}$):**
      $$P_{\text{ACB}} = C_{\text{patient}} \times 2.0 \quad \text{if } pK_{i,\text{M1}} \ge 7.0 \text{ else } 0.0$$
      *Where cognitive vulnerability constants $C_{\text{patient}} = 3.0$ if MMSE < 10, $2.0$ if MMSE 10–20, and $1.0$ if MMSE > 20.*

    * **Organ Clearance Penalty ($P_{\text{organ}}$):**
      $$P_{\text{organ}} = 4.0 \left( \lambda_{\text{renal}} \cdot \text{Fr}_{\text{renal}} + \lambda_{\text{hepatic}} \cdot \text{Fr}_{\text{hepatic}} \right)$$

    * **Prior Response Penalty ($P_{\text{history}}$):**
      Applies a $-5.0$ penalty for documented prior treatment failure at standard doses, forces a hard-lock for prior severe adverse reactions/intolerance, or highlights dose optimization when partial benefit was previously observed.

    ---

    **4. Clinical Safety Hard-Lock Protocol**
    
    Absolute clinical contraindications override scoring and force an unconditional lock ($M_j = -999.0$):
    * **Etiology Hard-Lock:** Full $D_2$ antagonists in DLB or PDD etiologies.
    * **Cardiac Hard-Lock:** High QTc-risk agents ($\text{Risk}_{\text{QTc}} > 0.60$) when baseline QTc $> 500\text{ ms}$.
    * **Clinician Rule-Out:** Direct UI override via session-state toggle.

    ---

    **5. Side-Effect Probability**
    
    Estimates receptor occupancy and baseline vulnerability into clinically risk likelihood percentages ($P_{\text{event}} \in [0\%, 95\%]$) via a bounded logistic function:

    $$P_{\text{event}} = \min\left(95\%, \text{int}\left( \frac{100}{1 + e^{-0.5(pK_i \cdot \lambda - 3.5)}} \right)\right)$$
    """)

# -----------------------------------------------------------------------------
# 6. CITATIONS & REFERENCES
# -----------------------------------------------------------------------------
with st.expander("🔍 References & Citations"):
    st.markdown(
        """
        1. **Roth, B. L., et al.** *PDSP Ki Database. Psychoactive Drug Screening Program (PDSP)*. UNC Chapel Hill / NIMH.
        2. **Magierski, R., et al. (2020).** *Pharmacotherapy of Behavioral and Psychological Symptoms of Dementia: State of the Art and Future Progress*. Front. Psychiatry. PMID: 32848775.
        3. **Tampi, R. R., et al. (2022).** *Brexpiprazole for the Treatment of Agitation in Dementia*. Drugs Aging. PMID: 35904712.
        4. **Lee, D., et al. (2023).** *Brexpiprazole for the Treatment of Agitation Associated with Dementia Due to Alzheimer's Disease*. Am J Psychiatry. PMID: 37143168.
        5. **Davies, S. J., et al. (2018).** *Sequential drug treatment algorithm for agitation and aggression in Alzheimer's and mixed dementia*. J Psychopharmacol. PMID: 29338602.
        6. **CCSMH (2024–2025).** *Canadian Clinical Practice Guidelines for Assessing and Managing BPSD*. ccsmh.ca.
        """
    )


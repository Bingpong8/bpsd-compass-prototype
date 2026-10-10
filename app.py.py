"""
BPSD Compass Prototype (P5), framework v3.0
Occupancy-based, two-stage decision-support model with an audited, evidence-annotated database.
**THIS PROTOYPE STILL NOT validated for clinical use.**

FILE ORGANISATION
  0. Imports, page configuration and header
  1. UI mappings and session state
  2. Knowledge base
       2.1 Drug database
       2.2 Model constants, receptor mechanisms, anchors, safety domains
  3. Evidence & calibration registry
       3.1 Evidence tables  3.2 Parameter register  3.3 Validation & improvement roadmaps
  4. Model engine
       4.1 Numerics and uncertainty sampler   4.2 Need weights and patient vulnerability
       4.3 Exposure and occupancy             4.4 Hazards and regimen scoring
       4.5 Stage A constraints and notes      4.6 Candidate evaluation, Monte Carlo, sensitivity
       4.7 Combination regimens and cross-titration
  5. Verification: self-checks
  6. User interface
       6.1 Settings   6.2 Patient parameters   6.3 Medication audit   6.4 Symptoms & preferences
       6.5 Decision engine   6.6 Dashboards   6.7 Phenotype   6.8 Calibration center
       6.9 Rationale   6.10 References
"""
# =============================================================================
# 0. IMPORTS, PAGE CONFIGURATION & HEADER
# =============================================================================
import streamlit as st
import numpy as np
import pandas as pd
import itertools

# Page Configuration & Header Setup
st.set_page_config(page_title="BPSD Compass Prototype (P5 - Framework v3)", layout="wide")
st.title("BPSD Compass Prototype (P5)")
st.caption("Parameter-driven decision support tools — occupancy-based framework v3")

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

# =============================================================================
# 1. UI MAPPINGS & SESSION STATE
# =============================================================================
# Initialize Session State for Interactive Rule-Out
if "ruled_out" not in st.session_state:
    st.session_state.ruled_out = set()

# Standardized AED Name Mappings (Including Dilantin / Phenytoin)
AED_NAME_MAP = {
    "Valproic Acid": "valproate",
    "Carbamazepine": "carbamazepine",
    "Lamotrigine": "lamotrigine",
    "Gabapentin": "gabapentin",
    "Phenytoin (Dilantin)": "phenytoin"
}

AED_TO_DRUG_NAME = {
    "carbamazepine": "Carbamazepine",
    "valproate": "Valproic Acid",
    "lamotrigine": "Lamotrigine",
    "gabapentin": "Gabapentin",
    "phenytoin": "Phenytoin (Dilantin)"
}

# Non-Database Prior Regimen Agents
MAOI_AGENTS = ["Phenelzine (MAOI)", "Tranylcypromine (MAOI)", "Selegiline (MAOI)"]
ANTICHOLINERGIC_AGENTS = ["Trihexyphenidyl (Anticholinergic)", "Benztropine (Anticholinergic)"]
ACHEI_AGENTS = ["Donepezil (AChEI)", "Galantamine (AChEI)", "Rivastigmine (AChEI)"]

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

FRAILTY_MAPPING = {
    "0 - Robust / Non-Frail (0.1)": 0.1,
    "1 - Mild Frailty / Pre-Frail (0.3)": 0.3,
    "2 - Moderate Frailty (0.5)": 0.5,
    "3 - Severe Frailty / Dependent (0.8)": 0.8
}

CAREGIVER_CONCERN_MAPPING = {
    "0 - None (No concern)": 0.5,
    "1 - Low Concern": 1.0,
    "2 - High Concern": 1.5
}

# =============================================================================
# 2. KNOWLEDGE BASE
# =============================================================================
# -----------------------------------------------------------------------------
# 2.1 DRUG DATABASE (audited; status codes explained in section 3)
# -----------------------------------------------------------------------------
# Fields: pKi (11 receptors; pKi<=5.0 = floor/censored, 0.0 = no binding), pKi_status / pKi_sigma, Fr_renal / Fr_hepatic,
# Risk_QTc (+ CredibleMeds category), convulsant_index, hypoNa_index, d2_full_antagonist, dosage, warnings.
DRUG_DATABASE = {
    'Brexpiprazole': {
        "category": 'Atypical Antipsychotic',
        "pKi": {'5HT2A': 9.33, 'D2': 9.5, 'NET': 5.0, 'SERT': 5.0, 'α2A': 7.4, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 7.72, 'α1': 8.0, 'M1': 5.0, '5HT2C': 7.47},
        "pKi_status": {'5HT2A': 'C', 'D2': 'V', 'NET': 'F', 'SERT': 'F', 'α2A': 'U', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'C', 'α1': 'V', 'M1': 'F', '5HT2C': 'C'},
        "Fr_renal": 0.14, "Fr_hepatic": 0.86, "pk_status": 'U', "pk_note": 'carried over (not re-checked)',
        "Risk_QTc": 0.1, "qt_category": 'NL', "qt_status": 'C',
        "convulsant_index": 0.1, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Start 0.5 mg PO OD, max 2 mg/day for agitation.',
        "warnings": 'Akathisia and impulse-control disorders. Low DDI risk profile.'
    },
    'Pimavanserin': {
        "category": 'Atypical Antipsychotic',
        "pKi": {'5HT2A': 9.3, 'D2': 5.0, 'NET': 0.0, 'SERT': 5.0, 'α2A': 0.0, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 5.0, 'α1': 5.0, 'M1': 5.0, '5HT2C': 9.36},
        "pKi_status": {'5HT2A': 'V', 'D2': 'F', 'NET': 'X', 'SERT': 'F', 'α2A': 'X', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'F', 'α1': 'F', 'M1': 'F', '5HT2C': 'C'},
        "Fr_renal": 0.06, "Fr_hepatic": 0.94, "pk_status": 'U', "pk_note": 'carried over (not re-checked)',
        "Risk_QTc": 0.3, "qt_category": 'PR', "qt_status": 'C',
        "convulsant_index": 0.0, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Standard dose: 34 mg PO OD or 10 mg PO OD in CYP3A4 inhibitor co-administration.',
        "warnings": "QTc prolongation risk. Indicated primarily for Parkinson's Disease Psychosis."
    },
    'Risperidone': {
        "category": 'Atypical Antipsychotic',
        "d2_full_antagonist": True,
        "pKi": {'5HT2A': 9.7, 'D2': 8.9, 'NET': 5.0, 'SERT': 5.0, 'α2A': 6.8, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 8.0, 'α1': 9.0, 'M1': 5.55, '5HT2C': 7.96},
        "pKi_status": {'5HT2A': 'V', 'D2': 'V', 'NET': 'F', 'SERT': 'F', 'α2A': 'V', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'C', 'α1': 'V', 'M1': 'C', '5HT2C': 'C'},
        "Fr_renal": 0.7, "Fr_hepatic": 0.3, "pk_status": 'V', "pk_note": '70% of dose in urine (active moiety); clearance of risperidone+metabolite reduced ~60% in renal disease',
        "Risk_QTc": 0.5, "qt_category": 'CR', "qt_status": 'V',
        "convulsant_index": 0.15, "seizure_status": 'C', "seizure_note": 'FDA-type incidence 0.3% on the same scale',
        "hypoNa_index": 0.0,
        "dosage": 'Start 0.25 mg - 0.5 mg/day; titrate up to 0.5 mg - 1.5 mg/day (max 2.0 mg/day in elderly).',
        "warnings": 'QTc prolongation, dose-dependent EPS, hyperprolactinemia, and cerebrovascular risk.'
    },
    'Quetiapine': {
        "category": 'Atypical Antipsychotic',
        "d2_full_antagonist": True,
        "pKi": {'5HT2A': 6.8, 'D2': 6.68, 'NET': 5.0, 'SERT': 5.0, 'α2A': 5.5, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 8.0, 'α1': 7.74, 'M1': 7.0, '5HT2C': 6.0},
        "pKi_status": {'5HT2A': 'V', 'D2': 'C', 'NET': 'F', 'SERT': 'F', 'α2A': 'V', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'V', 'α1': 'C', 'M1': 'C', '5HT2C': 'V'},
        "Fr_renal": 0.05, "Fr_hepatic": 0.95, "pk_status": 'V', "pk_note": '~99% hepatic metabolism',
        "Risk_QTc": 0.4, "qt_category": 'CR', "qt_status": 'V',
        "convulsant_index": 0.35, "seizure_status": 'C', "seizure_note": 'FDA-type incidence 0.8% scaled to olanzapine 0.9% = 0.40',
        "hypoNa_index": 0.0,
        "dosage": 'Start 12.5 mg - 25 mg PO hs; target 50 - 150 mg/day in divided doses.',
        "warnings": 'QTc prolongation, severe orthostatic hypotension, sedation, and metabolic dysregulation.'
    },
    'Olanzapine': {
        "category": 'Atypical Antipsychotic',
        "d2_full_antagonist": True,
        "pKi": {'5HT2A': 8.5, 'D2': 7.8, 'NET': 5.0, 'SERT': 5.0, 'α2A': 6.55, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 8.8, 'α1': 7.7, 'M1': 8.33, '5HT2C': 7.8},
        "pKi_status": {'5HT2A': 'V', 'D2': 'V', 'NET': 'F', 'SERT': 'F', 'α2A': 'C', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'V', 'α1': 'V', 'M1': 'C', '5HT2C': 'V'},
        "Fr_renal": 0.07, "Fr_hepatic": 0.93, "pk_status": 'V', "pk_note": 'Excreted mainly as changed drug; ~7% unchanged',
        "Risk_QTc": 0.3, "qt_category": 'CR', "qt_status": 'V',
        "convulsant_index": 0.4, "seizure_status": 'V', "seizure_note": 'FDA-type incidence 0.9% (highest of the atypicals listed)',
        "hypoNa_index": 0.0,
        "dosage": 'Start 2.5 mg PO hs; titrate up to 5.0 mg/day, max 10 mg/day.',
        "warnings": 'Severe metabolic syndrome, weight gain, sedation, and anticholinergic cognitive impairment.'
    },
    'Haloperidol': {
        "category": 'Typical Antipsychotic',
        "d2_full_antagonist": True,
        "pKi": {'5HT2A': 7.2, 'D2': 9.2, 'NET': 5.0, 'SERT': 5.49, 'α2A': 5.0, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 6.0, 'α1': 8.12, 'M1': 5.8, '5HT2C': 5.0},
        "pKi_status": {'5HT2A': 'V', 'D2': 'V', 'NET': 'F', 'SERT': 'V', 'α2A': 'F', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'V', 'α1': 'C', 'M1': 'C', '5HT2C': 'F'},
        "Fr_renal": 0.05, "Fr_hepatic": 0.95, "pk_status": 'C', "pk_note": 'Minimal renal excretion (psychotropic renal-dosing reviews)',
        "Risk_QTc": 0.85, "qt_category": 'KR', "qt_status": 'V',
        "convulsant_index": 0.3, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Start as low as possible, 0.25-0.5 mg PO OD or PRN - max 2 mg/day.',
        "warnings": 'Torsades de Pointes, Extrapyramidal Symptoms (EPS), and Tardive Dyskinesia.'
    },
    'Escitalopram': {
        "category": 'Antidepressant (SSRI)',
        "pKi": {'5HT2A': 5.2, 'D2': 5.0, 'NET': 5.0, 'SERT': 8.74, 'α2A': 5.0, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 6.3, 'α1': 5.0, 'M1': 5.0, '5HT2C': 5.0},
        "pKi_status": {'5HT2A': 'U', 'D2': 'F', 'NET': 'F', 'SERT': 'V', 'α2A': 'F', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'U', 'α1': 'F', 'M1': 'F', '5HT2C': 'F'},
        "Fr_renal": 0.2, "Fr_hepatic": 0.8, "pk_status": 'V', "pk_note": '~8-10% unchanged in urine (upper bound kept for active metabolites)',
        "Risk_QTc": 0.75, "qt_category": 'KR', "qt_status": 'V',
        "convulsant_index": 0.1, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Start 5 mg/day; max 10 mg/day.',
        "warnings": 'Dose-dependent QTc prolongation risk; hyponatremia and serotonin syndrome precautions.'
    },
    'Sertraline': {
        "category": 'Antidepressant (SSRI)',
        "pKi": {'5HT2A': 6.2, 'D2': 6.6, 'NET': 6.38, 'SERT': 9.77, 'α2A': 5.0, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 5.0, 'α1': 5.0, 'M1': 5.0, '5HT2C': 5.0},
        "pKi_status": {'5HT2A': 'U', 'D2': 'U', 'NET': 'C', 'SERT': 'V', 'α2A': 'F', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'F', 'α1': 'F', 'M1': 'F', '5HT2C': 'F'},
        "Fr_renal": 0.02, "Fr_hepatic": 0.98, "pk_status": 'C', "pk_note": 'Negligible unchanged urinary excretion; exposure unchanged in renal impairment (reviews)',
        "Risk_QTc": 0.25, "qt_category": 'CR', "qt_status": 'V',
        "convulsant_index": 0.1, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Start 25 mg/day; titrate up to 50 mg - 100 mg/day.',
        "warnings": 'Hyponatremia/SIADH, serotonin syndrome, and mild GI distress.'
    },
    'Valproic Acid': {
        "category": 'Mood Stabilizer Anticonvulsant',
        "pKi": {'5HT2A': 0.0, 'D2': 0.0, 'NET': 0.0, 'SERT': 0.0, 'α2A': 0.0, 'NMDA': 0.0, 'GABA-A': 7.2, 'H1': 5.0, 'α1': 5.0, 'M1': 5.0, '5HT2C': 0.0},
        "pKi_status": {'5HT2A': 'X', 'D2': 'X', 'NET': 'X', 'SERT': 'X', 'α2A': 'X', 'NMDA': 'X', 'GABA-A': 'P', 'H1': 'F', 'α1': 'F', 'M1': 'F', '5HT2C': 'X'},
        "Fr_renal": 0.05, "Fr_hepatic": 0.95, "pk_status": 'V', "pk_note": '<3% unchanged in urine (label knowledge, not re-searched)',
        "Risk_QTc": 0.1, "qt_category": 'NL', "qt_status": 'V',
        "convulsant_index": 0.0, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Start 125 mg - 250 mg PO bid; keep serum level 50-80 mcg/ml.',
        "warnings": 'Hepatotoxicity, pancreatitis, thrombocytopenia; monitor baseline LFTs, CBC, and plasma levels.'
    },
    'Gabapentin': {
        "category": 'GABA analogue Anticonvulsant',
        "pKi": {'5HT2A': 0.0, 'D2': 0.0, 'NET': 0.0, 'SERT': 0.0, 'α2A': 0.0, 'NMDA': 0.0, 'GABA-A': 6.8, 'H1': 5.0, 'α1': 5.0, 'M1': 5.0, '5HT2C': 0.0},
        "pKi_status": {'5HT2A': 'X', 'D2': 'X', 'NET': 'X', 'SERT': 'X', 'α2A': 'X', 'NMDA': 'X', 'GABA-A': 'P', 'H1': 'F', 'α1': 'F', 'M1': 'F', '5HT2C': 'X'},
        "Fr_renal": 1.0, "Fr_hepatic": 0.0, "pk_status": 'V', "pk_note": 'Eliminated unchanged by the kidney (well established)',
        "Risk_QTc": 0.05, "qt_category": 'NL', "qt_status": 'V',
        "convulsant_index": 0.0, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Start 100 mg tid; slow titrations up to 300 mg - 600 mg tid based on eGFR.',
        "warnings": 'Respiratory depression risk with CNS depressants/opioids; 100% renal elimination requires strict dose adjustment.'
    },
    'Memantine': {
        "category": 'Cognitive Enhancer',
        "pKi": {'5HT2A': 0.0, 'D2': 0.0, 'NET': 0.0, 'SERT': 0.0, 'α2A': 0.0, 'NMDA': 6.07, 'GABA-A': 0.0, 'H1': 0.0, 'α1': 0.0, 'M1': 0.0, '5HT2C': 0.0},
        "pKi_status": {'5HT2A': 'X', 'D2': 'X', 'NET': 'X', 'SERT': 'X', 'α2A': 'X', 'NMDA': 'C', 'GABA-A': 'X', 'H1': 'X', 'α1': 'X', 'M1': 'X', '5HT2C': 'X'},
        "Fr_renal": 0.8, "Fr_hepatic": 0.2, "pk_status": 'V', "pk_note": '48% unchanged in urine; AUC x1.62/1.97/2.33 in mild/moderate/severe impairment -> implied Fr_ren 0.70-0.86',
        "Risk_QTc": 0.05, "qt_category": 'NL', "qt_status": 'V',
        "convulsant_index": 0.0, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Start 5 mg daily; titrate by 5 mg weekly up to 10 mg bid (max 10 mg daily if eGFR < 30).',
        "warnings": 'Dose adjustment required in severe renal impairment (eGFR < 30 ml/min).'
    },
    'Amitriptyline': {
        "category": 'Tricyclic Antidepressant (TCA)',
        "pKi": {'5HT2A': 8.1, 'D2': 5.5, 'NET': 7.7, 'SERT': 8.41, 'α2A': 6.8, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 8.9, 'α1': 8.0, 'M1': 7.79, '5HT2C': 7.5},
        "pKi_status": {'5HT2A': 'U', 'D2': 'U', 'NET': 'V', 'SERT': 'V', 'α2A': 'U', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'V', 'α1': 'V', 'M1': 'C', '5HT2C': 'U'},
        "Fr_renal": 0.05, "Fr_hepatic": 0.95, "pk_status": 'U', "pk_note": 'carried over (not re-checked)',
        "Risk_QTc": 0.55, "qt_category": 'CR', "qt_status": 'C',
        "convulsant_index": 0.37, "seizure_status": 'C', "seizure_note": '1-1/SIR with SIR 1.58 (tricyclics)',
        "hypoNa_index": 0.0,
        "dosage": 'Generally avoid in dementia (5-10 mg PO hs strictly if indicated).',
        "warnings": 'Severe anticholinergic toxicity, fall risk, cognitive decline, and cardiotoxicity.'
    },
    'Venlafaxine': {
        "category": 'Antidepressant (SNRI)',
        "pKi": {'5HT2A': 5.0, 'D2': 5.0, 'NET': 5.79, 'SERT': 7.57, 'α2A': 5.0, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 5.0, 'α1': 5.0, 'M1': 5.0, '5HT2C': 5.0},
        "pKi_status": {'5HT2A': 'F', 'D2': 'F', 'NET': 'C', 'SERT': 'V', 'α2A': 'F', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'F', 'α1': 'F', 'M1': 'F', '5HT2C': 'F'},
        "pKi_sigma": {'SERT': np.float64(0.48)},
        "Fr_renal": 0.85, "Fr_hepatic": 0.15, "pk_status": 'V', "pk_note": 'Chiefly urinary elimination of venlafaxine + active metabolite; clearance reduced in renal insufficiency',
        "Risk_QTc": 0.35, "qt_category": 'PR', "qt_status": 'V',
        "convulsant_index": 0.2, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Start 37.5 mg daily XR; titrate up to 75 mg - 150 mg/day.',
        "warnings": 'Blood pressure elevation; sharp discontinuation syndrome.'
    },
    'Vortioxetine': {
        "category": 'Multimodal Serotonin Modulator Antidepressant',
        "pKi": {'5HT2A': 7.6, 'D2': 5.0, 'NET': 5.0, 'SERT': 8.53, 'α2A': 5.0, 'NMDA': 5.0, 'GABA-A': 5.0, 'H1': 5.0, 'α1': 5.0, 'M1': 5.0, '5HT2C': 5.0},
        "pKi_status": {'5HT2A': 'U', 'D2': 'F', 'NET': 'F', 'SERT': 'V', 'α2A': 'F', 'NMDA': 'F', 'GABA-A': 'F', 'H1': 'F', 'α1': 'F', 'M1': 'F', '5HT2C': 'F'},
        "Fr_renal": 0.59, "Fr_hepatic": 0.41, "pk_status": 'U', "pk_note": 'carried over (not re-checked)',
        "Risk_QTc": 0.15, "qt_category": 'NL', "qt_status": 'V',
        "convulsant_index": 0.1, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Start 5.0 mg daily; titrate up to 5.0 mg - 10 mg/day in elderly.',
        "warnings": 'Nausea risk; low anticholinergic burden and favorable cognitive safety profile.'
    },
    'Bupropion': {
        "category": 'Antidepressant (NDRI)',
        "pKi": {'5HT2A': 5.0, 'D2': 5.2, 'NET': 5.0, 'SERT': 5.04, 'α2A': 5.0, 'NMDA': 5.0, 'GABA-A': 5.0, 'H1': 5.0, 'α1': 5.0, 'M1': 5.0, '5HT2C': 5.0},
        "pKi_status": {'5HT2A': 'F', 'D2': 'P', 'NET': 'C', 'SERT': 'V', 'α2A': 'F', 'NMDA': 'F', 'GABA-A': 'F', 'H1': 'F', 'α1': 'F', 'M1': 'F', '5HT2C': 'F'},
        "Fr_renal": 0.87, "Fr_hepatic": 0.13, "pk_status": 'U', "pk_note": 'carried over (not re-checked)',
        "Risk_QTc": 0.1, "qt_category": 'NL', "qt_status": 'C',
        "convulsant_index": 0.76, "seizure_status": 'C', "seizure_note": '1-1/SIR with SIR 4.08 (bupropion IR); label 0.4% at <=450 mg/day',
        "hypoNa_index": 0.0,
        "dosage": 'Start 100 mg SR or 150 mg XL daily; max 150 mg daily in elderly.',
        "warnings": 'Dose-dependent seizure risk; strictly contraindicated in seizure disorders.'
    },
    'Trazodone': {
        "category": 'Antidepressant (SARI)',
        "pKi": {'5HT2A': 7.8, 'D2': 5.0, 'NET': 5.0, 'SERT': 6.62, 'α2A': 6.4, 'NMDA': 5.0, 'GABA-A': 5.0, 'H1': 7.5, 'α1': 7.31, 'M1': 5.0, '5HT2C': 6.5},
        "pKi_status": {'5HT2A': 'V', 'D2': 'F', 'NET': 'F', 'SERT': 'V', 'α2A': 'U', 'NMDA': 'F', 'GABA-A': 'F', 'H1': 'U', 'α1': 'C', 'M1': 'F', '5HT2C': 'V'},
        "pKi_sigma": {'α1': np.float64(0.49)},
        "Fr_renal": 0.7, "Fr_hepatic": 0.3, "pk_status": 'U', "pk_note": 'carried over (not re-checked)',
        "Risk_QTc": 0.45, "qt_category": 'CR', "qt_status": 'V',
        "convulsant_index": 0.1, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Start 12.5 mg - 25 mg hs/PRN; titrate up to 25 mg - 100 mg/day for nighttime agitation.',
        "warnings": 'Orthostatic hypotension, priapism, daytime sedation.'
    },
    'Mirtazapine': {
        "category": 'Pyridine Tetracyclic Antidepressant (NaSSA)',
        "pKi": {'5HT2A': 8.1, 'D2': 5.0, 'NET': 5.0, 'SERT': 5.0, 'α2A': 7.3, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 9.3, 'α1': 6.0, 'M1': 5.0, '5HT2C': 7.6},
        "pKi_status": {'5HT2A': 'V', 'D2': 'F', 'NET': 'F', 'SERT': 'F', 'α2A': 'V', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'V', 'α1': 'U', 'M1': 'F', '5HT2C': 'V'},
        "Fr_renal": 0.5, "Fr_hepatic": 0.5, "pk_status": 'C', "pk_note": 'Label clearance falls ~30-50% in moderate-severe impairment -> implied Fr_ren 0.4-0.55 (urinary recovery 75% is mostly metabolites)',
        "Risk_QTc": 0.3, "qt_category": 'PR', "qt_status": 'V',
        "convulsant_index": 0.05, "seizure_status": 'C', "seizure_note": 'Label-type incidence 0.04%',
        "hypoNa_index": 0.0,
        "dosage": 'Start 7.5 mg hs; 15 mg - 30 mg hs (higher doses decrease sedating H1 effect).',
        "warnings": 'Low-dose sedation and weight gain.'
    },
    'Mianserin': {
        "category": 'Benzene Tetracyclic Antidepressant (NaSSA)',
        "pKi": {'5HT2A': 8.0, 'D2': 5.5, 'NET': 6.0, 'SERT': 5.4, 'α2A': 7.2, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 9.0, 'α1': 7.3, 'M1': 5.0, '5HT2C': 7.5},
        "pKi_status": {'5HT2A': 'U', 'D2': 'U', 'NET': 'U', 'SERT': 'V', 'α2A': 'U', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'U', 'α1': 'U', 'M1': 'F', '5HT2C': 'U'},
        "Fr_renal": 0.7, "Fr_hepatic": 0.3, "pk_status": 'U', "pk_note": 'carried over (not re-checked)',
        "Risk_QTc": 0.35, "qt_category": 'UN', "qt_status": 'U',
        "convulsant_index": 0.2, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Start 10 mg hs; titrate up to 30 mg - 60 mg hs.',
        "warnings": 'Agranulocytosis risk, high sedation, and orthostasis.'
    },
    'Lamotrigine': {
        "category": 'Mood Stabilizer Anticonvulsant',
        "pKi": {'5HT2A': 0.0, 'D2': 0.0, 'NET': 0.0, 'SERT': 0.0, 'α2A': 0.0, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 0.0, 'α1': 0.0, 'M1': 0.0, '5HT2C': 0.0},
        "pKi_status": {'5HT2A': 'X', 'D2': 'X', 'NET': 'X', 'SERT': 'X', 'α2A': 'X', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'X', 'α1': 'X', 'M1': 'X', '5HT2C': 'X'},
        "Fr_renal": 0.1, "Fr_hepatic": 0.9, "pk_status": 'C', "pk_note": '~10% unchanged in urine; glucuronidation (label knowledge, not re-searched)',
        "Risk_QTc": 0.1, "qt_category": 'NL', "qt_status": 'V',
        "convulsant_index": 0.0, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Start 25 mg/day; slow titration up to 100 mg - 200 mg/day.',
        "warnings": 'Stevens-Johnson Syndrome (SJS). Discontinue immediately at first sign of rash. Reduce dose 50% with Valproate.'
    },
    'Carbamazepine': {
        "category": 'Mood Stabilizer Anticonvulsant',
        "pKi": {'5HT2A': 0.0, 'D2': 0.0, 'NET': 0.0, 'SERT': 0.0, 'α2A': 0.0, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 0.0, 'α1': 0.0, 'M1': 0.0, '5HT2C': 0.0},
        "pKi_status": {'5HT2A': 'X', 'D2': 'X', 'NET': 'X', 'SERT': 'X', 'α2A': 'X', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'X', 'α1': 'X', 'M1': 'X', '5HT2C': 'X'},
        "Fr_renal": 0.03, "Fr_hepatic": 0.97, "pk_status": 'C', "pk_note": '<3% unchanged in urine; hepatic CYP3A4 (label knowledge, not re-searched)',
        "Risk_QTc": 0.3, "qt_category": 'UN', "qt_status": 'U',
        "convulsant_index": 0.0, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.4,
        "dosage": 'Start 100 mg bid; titrate up to 200 mg - 600 mg/day in divided doses.',
        "warnings": 'Potent CYP3A4 inducer. Aplastic anemia, agranulocytosis, hyponatremia, and severe dermatologic reactions.'
    },
}

# -----------------------------------------------------------------------------
# 2.2 MODEL CONSTANTS, RECEPTOR MECHANISMS, ANCHORS & SAFETY DOMAINS
# -----------------------------------------------------------------------------
# Values marked PLACEHOLDER are uncalibrated modelling choices; section 3 lists the status of
# every parameter, how to calibrate it and how to validate it.

RECEPTORS_ALL = ["5HT2A", "D2", "NET", "SERT", "α2A", "NMDA", "GABA-A", "H1", "α1", "M1", "5HT2C"]
THERAPEUTIC_RECEPTORS = ["5HT2A", "D2", "NET", "SERT", "α2A", "NMDA", "GABA-A"]

PHI_MIN = 0.10          # floor on fraction of normal clearance remaining
PHI_MAX = 3.0           # cap on induction-driven clearance gain
E_MAX_STANDARD = 3.0     # PLACEHOLDER: max tolerated accumulation (x reference) at the standard dose (Stage A)
INDUCER_HEPATIC_CL_FACTOR = 2.0   # PLACEHOLDER: carbamazepine/phenytoin hepatic clearance multiplier
INDUCER_AEDS = {"carbamazepine", "phenytoin"}

CENSOR_PKI = 5.0        # pKi <= 5.0 in the database = floor/"no meaningful binding" -> occupancy 0
SIGMA_PKI = 0.40        # inter-assay noise (log10 units); re-estimated from 6 multi-source SERT pairs (σ≈0.38)

THETA_STAR = 0.80       # D2 net-antagonism window threshold (PET-derived; EPS rises above this)
TAU_EPS = 0.05          # width of the EPS window
QTC_MAX = 500.0         # hard QTc limit (ms)
QT_EFFECT_MS = 40.0     # PLACEHOLDER: dQTc (ms) at Risk_QTc = 1.0 and reference exposure
QT_MIN_MEANINGFUL_MS = 10.0   # a QTc rise below this does not trigger the QTc hard constraint
U_MIN = 0.02            # candidates with utility <= U_MIN have no modelled therapeutic engagement

# Receptor mechanism table: alpha = intrinsic activity relative to full agonist,
# tone = endogenous tone. Net signalling change per full occupancy: Delta = alpha - tone.
RECEPTOR_PROFILE = {
    "5HT2A": {"alpha": 0.0, "tone": 1.0},   # antagonism / inverse agonism  -> Delta = -1
    "D2":    {"alpha": 0.0, "tone": 1.0},   # antagonism                    -> Delta = -1
    "NET":   {"alpha": 1.0, "tone": 0.0},   # reuptake inhibition (enhancer)-> Delta = +1
    "SERT":  {"alpha": 1.0, "tone": 0.0},   # reuptake inhibition (enhancer)-> Delta = +1
    "α2A":   {"alpha": 0.0, "tone": 1.0},   # antagonism                    -> Delta = -1
    "NMDA":  {"alpha": 0.0, "tone": 1.0},   # channel blockade              -> Delta = -1
    "GABA-A": {"alpha": 1.0, "tone": 0.0},  # positive modulation           -> Delta = +1
}
ALPHA_OVERRIDES = {("Brexpiprazole", "D2"): 0.43}   # D2 partial agonist (PLACEHOLDER value)

# Signed symptom -> receptor coupling kappa (sign = desired direction of net signalling change).
# Magnitudes carried over from the previous version; D2/apathy is now explicitly opposite in sign.
KAPPA = {
    "5HT2A": {"delusions": -0.7, "hallucinations": -0.8, "agitation": -0.5, "disinhibition": -0.5,
              "sleep": -0.6, "motor": -0.5, "appetite": -0.4},
    "D2":    {"agitation": -0.6, "delusions": -0.5, "hallucinations": -0.4, "apathy": +0.3, "motor": -0.5},
    "NET":   {"apathy": +0.8, "depression": +0.7},
    "SERT":  {"depression": +0.7, "anxiety": +0.6, "irritability": +0.4, "agitation": +0.3},
    "α2A":   {"agitation": -0.5, "irritability": -0.4},
    "NMDA":  {"apathy": -0.4},
    "GABA-A": {"anxiety": +0.7, "irritability": +0.6, "euphoria": +0.5, "disinhibition": +0.5},
}

# Reference anchors: fix the unbound target-site concentration at the standard geriatric dose in
# normal organ function. ("occ", receptor, theta) back-calculates C_ref = Ki * theta / (1 - theta);
# ("conc", nM) gives the unbound concentration directly.
# PLACEHOLDER / ORDER-OF-MAGNITUDE ESTIMATES: replace with PET occupancy or measured unbound
# concentrations. These anchors are the single most influential uncalibrated inputs of the model.
REFERENCE_ANCHORS = {
    "Brexpiprazole": ("occ", "D2", 0.65),
    "Pimavanserin":  ("occ", "5HT2A", 0.80),
    "Risperidone":   ("occ", "D2", 0.65),
    "Quetiapine":    ("occ", "D2", 0.30),
    "Olanzapine":    ("occ", "D2", 0.55),
    "Haloperidol":   ("occ", "D2", 0.60),
    "Valproic Acid": ("occ", "GABA-A", 0.80),
    "Gabapentin":    ("occ", "GABA-A", 0.70),
    "Memantine":     ("occ", "NMDA", 0.50),   # PLACEHOLDER: NMDA Ki is in the µM range (~0.85 µM)
    "Escitalopram":  ("occ", "SERT", 0.80),   # PET: ~80% SERT occupancy at usual minimum dose (10 mg)
    "Sertraline":    ("occ", "SERT", 0.80),   # PET: ~80% at ~50 mg
    "Amitriptyline": ("conc", 1.0),
    "Venlafaxine":   ("occ", "SERT", 0.80),   # PET: ~80% at 75 mg
    "Vortioxetine":  ("occ", "SERT", 0.60),   # PLACEHOLDER: verify against vortioxetine PET dose-occupancy data (5-10 mg)
    "Bupropion":     ("conc", 80.0),
    "Trazodone":     ("conc", 50.0),
    "Mirtazapine":   ("conc", 17.0),
    "Mianserin":     ("conc", 5.5),
    "Lamotrigine":   ("conc", 1000.0),
    "Carbamazepine": ("conc", 1000.0),
}
DEFAULT_ANCHOR = ("conc", 10.0)

# Risk domains: severity weight zeta, receptor liabilities beta (probability that full occupancy
# produces the adverse effect), and which patient-vulnerability function applies.
DOMAINS = {
    "sedation":    {"label": "Sedation / falls",                  "zeta": 0.20, "beta": {"H1": 0.9}},
    "orthostasis": {"label": "Orthostatic hypotension",           "zeta": 0.15, "beta": {"α1": 0.9}},
    "eps":         {"label": "Extrapyramidal (D2 window)",        "zeta": 0.25, "beta": {}},
    "qtc":         {"label": "QTc prolongation",                  "zeta": 0.25, "beta": {}},
    "cognition":   {"label": "Anticholinergic cognitive burden",  "zeta": 0.20, "beta": {"M1": 0.8}},
    "seizure":     {"label": "Seizure threshold",                 "zeta": 0.30, "beta": {}},
    "metabolic":   {"label": "Metabolic",                         "zeta": 0.10, "beta": {"H1": 0.5, "5HT2C": 0.7}},
    "hyponatremia": {"label": "Hyponatremia (SIADH)",             "zeta": 0.20, "beta": {"SERT": 0.5}},
}
ZETA_HISTORY = 0.20     # severity weight of the soft history / interaction burden

# Rescaled-logistic vulnerability parameters: (x0, k, x_min, x_max). x0 and k carried over from the
VULN_PARAMS = {
    "sedation_falls": (35.0, 0.08, 0.0, 125.0),   # Morse Fall Scale
    "orthostasis":    (15.0, 0.25, 0.0, 60.0),    # standing SBP drop (mmHg)
    "eps":            (8.0, 0.30, 0.0, 40.0),     # SAS score
    "qtc":            (450.0, 0.05, 400.0, 550.0),# baseline QTc (ms)
    "cognition":      (20.0, 0.25, 0.0, 30.0),    # MMSE (inverted: low = vulnerable)
    "seizure":        (1.0, 0.8, 0.0, 10.0),      # seizures / year
    "hba1c":          (8.0, 1.0, 4.0, 15.0),
    "bmi":            (30.0, 0.15, 12.0, 50.0),
    "sodium":         (136.0, 0.5, 120.0, 145.0), # baseline serum Na (mmol/L), inverted: low = vulnerable
}
THYROID_QTC_AMPLIFIER = 0.30     # PLACEHOLDER: extra QTc vulnerability with thyroid dysfunction
EPS_COMBO_AMPLIFIER = 0.25       # PLACEHOLDER: AChEI + SSRI + antipsychotic EPS potentiation

# Soft history / interaction burden components (each in [0, 1], combined by noisy-OR)
RHO_FAILURE = 0.60               # prior documented treatment failure
RHO_ACHEI_MAX = 0.80             # anticholinergic opposition to AChEI (scaled by M1 occupancy)
RHO_AED_DUPLICATE = 0.80         # candidate AED already on board for epilepsy
RHO_POLYPHARMACY = 0.15          # extra burden for adding a second agent

# Combination-regimen decision constants (index units: M is on a [-1-Z, 1] scale)
COMBO_SUPERIORITY_DELTA = 0.05
COMBO_STRONG_MARGIN = 0.15
COMBO_U_TARGET = 0.50

# Non-database agents in the audit: hazards they contribute to the patient's *background* risk
NONDB_BACKGROUND_HAZARD = {
    "Trihexyphenidyl (Anticholinergic)": {"cognition": 0.60, "sedation": 0.15},
    "Benztropine (Anticholinergic)":     {"cognition": 0.60, "sedation": 0.15},
}
# Hyponatremia (SIADH) model constants (all PLACEHOLDER unless noted)
NA_HARD_LIMIT = 130.0                          # baseline Na (mmol/L) below which SERT-active/SIADH agents are excluded
THIAZIDE_AMPLIFIER = 0.40                      # extra hyponatremia vulnerability with thiazide-type diuretic
AGE_NA_AMPLIFIER_MAX = 0.20                    # extra vulnerability reached at age >= 85 (ramp from 65)
SERT_ACTIVE_PKI = 7.0                          # pKi(SERT) at/above which an agent counts as SERT-active

SEROTONERGIC_DRUGS = ["Escitalopram", "Sertraline", "Venlafaxine", "Vortioxetine", "Amitriptyline"]
LEWY_SUBTYPES = ["Dementia with Lewy Bodies (DLB)", "Parkinson's Disease Dementia (PDD)"]
ANTIPSYCHOTIC_CATEGORIES = ["Atypical Antipsychotic", "Typical Antipsychotic"]



# Per-entry affinity uncertainty from the database (wider where sources disagree)
PKI_SIGMA = {(d, r): s for d, dd in DRUG_DATABASE.items() for r, s in dd.get("pKi_sigma", {}).items()}

# =============================================================================
# 3. EVIDENCE & CALIBRATION REGISTRY
#    3.1 Evidence tables · 3.2 Parameter register · 3.3 Validation & improvement roadmaps
# =============================================================================
# Status legend used across the database (pKi_status / pk_status / qt_status / seizure_status):
#   V = verified: consistent with >=1 independent literature source (|ΔpKi| <= 0.5)
#   C = corrected in this audit to the literature value
#   U = unverified: carried over from the previous version (not re-checked)
#   F = floor/censored (pKi <= 5.0: reported weak/none; occupancy set to 0)
#   X = no binding assumed (pKi 0.0)
#   P = proxy: functional stand-in, not a true binding affinity (e.g., GABA-A rows for valproate/gabapentin)
STATUS_LEGEND = {
    "V": "Verified (consistent with literature)", "C": "Corrected to literature", "U": "Unverified (carried over)",
    "F": "Floor / censored (weak or none reported)", "X": "No binding assumed", "P": "Proxy (not a true affinity)",
}

# -----------------------------------------------------------------------------
# 3.1a Multi-source affinity evidence (human Ki, nM). DB pKi = 9 - mean(log10 Ki) where corrected.
# -----------------------------------------------------------------------------
PKI_EVIDENCE = {   # (drug, receptor): (reported human Ki in nM, sources)
    ('Escitalopram', 'SERT'): ([1.1, 3.0], 'Vanderbilt protocol (1.1); Kristensen 2011 review (3)'),
    ('Sertraline', 'SERT'): ([0.29, 0.1], 'Tatsumi 1997 (0.29); Kristensen 2011 (0.1)'),
    ('Venlafaxine', 'SERT'): ([8.9, 82.0], 'Tatsumi 1997 (8.9); Owens 1997 (82)'),
    ('Amitriptyline', 'SERT'): ([3.45, 4.3], 'MCE/probes (3.45); Tatsumi 1997 (4.3)'),
    ('Vortioxetine', 'SERT'): ([1.6, 5.4], 'Bang-Andersen 2011 (1.6); transporter compilation (5.4)'),
    ('Trazodone', 'SERT'): ([160.0, 367.0], 'Tatsumi 1997 (160); US label (367)'),
    ('Bupropion', 'SERT'): ([9100.0], 'Tatsumi 1997'),
    ('Mianserin', 'SERT'): ([4000.0], 'PDSP-based compilation'),
    ('Haloperidol', 'SERT'): ([3256.0], 'Psychopharmacology review table (PMC5112764)'),
    ('Brexpiprazole', 'D2'): ([0.3], 'Maeda 2014'),
    ('Brexpiprazole', '5HT2A'): ([0.47], 'Maeda 2014'),
    ('Brexpiprazole', '5HT2C'): ([34.0], 'NPR2021'),
    ('Brexpiprazole', 'α1'): ([3.8], 'Maeda 2014 (α1A)'),
    ('Brexpiprazole', 'H1'): ([19.0], 'Maeda 2014'),
    ('Olanzapine', 'D2'): ([20.0], 'NPR2021'),
    ('Olanzapine', '5HT2A'): ([3.3], 'NPR2021'),
    ('Olanzapine', '5HT2C'): ([10.0], 'NPR2021'),
    ('Olanzapine', 'α1'): ([54.0], 'NPR2021'),
    ('Olanzapine', 'H1'): ([2.8], 'NPR2021'),
    ('Olanzapine', 'M1'): ([4.7], 'NPR2021'),
    ('Olanzapine', 'α2A'): ([280.0], 'TCRM 2017 (α2)'),
    ('Quetiapine', 'D2'): ([180.0, 245.0], 'NPR2021; PMC3770982'),
    ('Quetiapine', '5HT2A'): ([220.0, 100.0], 'NPR2021; PMC3770982'),
    ('Quetiapine', '5HT2C'): ([1400.0, 2502.0], 'NPR2021; PMC3770982'),
    ('Quetiapine', 'α1'): ([15.0, 22.0], 'NPR2021; PMC3770982'),
    ('Quetiapine', 'H1'): ([8.7, 11.0], 'NPR2021; PMC3770982'),
    ('Quetiapine', 'M1'): ([100.0], 'NPR2021'),
    ('Quetiapine', 'α2A'): ([3630.0], 'PMC3770982 (α2)'),
    ('Risperidone', 'D2'): ([2.2, 4.9], 'NPR2021; PMC3770982'),
    ('Risperidone', '5HT2A'): ([0.29, 0.17], 'NPR2021; PMC3770982'),
    ('Risperidone', '5HT2C'): ([10.0, 12.0], 'NPR2021; PMC3770982'),
    ('Risperidone', 'α1'): ([1.4, 5.0], 'NPR2021; PMC3770982'),
    ('Risperidone', 'H1'): ([19.0, 5.2], 'NPR2021; PMC3770982'),
    ('Risperidone', 'M1'): ([2800.0], 'NPR2021'),
    ('Risperidone', 'α2A'): ([151.0], 'PMC3770982 (α2)'),
    ('Haloperidol', 'D2'): ([1.4, 2.0, 2.6], 'NPR2021; PMC5112764; TCRM 2017'),
    ('Haloperidol', '5HT2A'): ([120.0, 70.0, 61.0], 'NPR2021; PMC5112764; TCRM 2017'),
    ('Haloperidol', 'α1'): ([4.7, 12.0, 8.0], 'NPR2021; PMC5112764 (α1A/B)'),
    ('Haloperidol', 'H1'): ([440.0, 260.0], 'NPR2021; TCRM 2017'),
    ('Haloperidol', 'M1'): ([1600.0], 'NPR2021'),
    ('Haloperidol', '5HT2C'): ([4700.0, 5000.0], 'NPR2021; PMC5112764'),
    ('Pimavanserin', '5HT2A'): ([0.087, 0.4], 'US label (0.087); R-SAT (0.4)'),
    ('Pimavanserin', '5HT2C'): ([0.44], 'US label'),
    ('Trazodone', '5HT2A'): ([35.6], 'US label'),
    ('Trazodone', '5HT2C'): ([224.0], 'US label'),
    ('Trazodone', 'α1'): ([16.0, 153.0], 'prior DB value (16, source not recorded); US label α1A (153)'),
    ('Mirtazapine', '5HT2A'): ([8.9, 6.3], 'IUPHAR-type pKi 8.05; 8.2 (literature)'),
    ('Mirtazapine', 'H1'): ([0.5, 1.6], 'IUPHAR-type pKi 9.3; 8.8'),
    ('Mirtazapine', '5HT2C'): ([12.6], 'literature pKi 7.9'),
    ('Mirtazapine', 'α2A'): ([112.0], 'IUPHAR-type pKi 6.95'),
    ('Mirtazapine', 'NET'): ([4600.0], 'Tatsumi 1997'),
    ('Amitriptyline', 'H1'): ([0.5, 1.1], 'MCE/probes'),
    ('Amitriptyline', 'α1'): ([4.4], 'MCE/probes'),
    ('Amitriptyline', 'M1'): ([11.0, 24.0], 'MCE/probes'),
    ('Amitriptyline', 'NET'): ([13.3, 35.0], 'MCE/probes (13.3); Tatsumi 1997 (35)'),
    ('Sertraline', 'NET'): ([420.0], 'Tatsumi 1997 (a review lists NET/DAT swapped; Tatsumi used)'),
    ('Venlafaxine', 'NET'): ([1060.0, 2480.0], 'Tatsumi 1997; Owens 1997'),
    ('Escitalopram', 'NET'): ([6514.0], 'Kristensen 2011 review'),
    ('Bupropion', 'NET'): ([52000.0], 'Tatsumi 1997'),
    ('Memantine', 'NMDA'): ([400.0, 700.0, 2200.0], '[3H]MK-801 Ki 0.4-0.7 µM; current IC50 2.2 µM'),
}

# -----------------------------------------------------------------------------
# 3.1b CredibleMeds categories -> hazard bands; label PK calibration points
# -----------------------------------------------------------------------------
QT_BANDS = {"KR": (0.70, 1.00), "CR": (0.25, 0.55), "PR": (0.15, 0.35), "NL": (0.00, 0.15)}
QT_CATEGORY_NAMES = {"KR": "Known risk of TdP", "CR": "Conditional risk", "PR": "Possible risk",
                     "NL": "Not found in lists reviewed", "UN": "Unverified"}

PK_CALIBRATION_POINTS = [   # (drug, setting, g_ren = fraction of normal GFR (approx.), AUC ratio, source)
    ("Memantine", "mild impairment (CrCl 50-80)", 0.72, 1.62, "PK review of label data (Front Psychiatry 2022)"),
    ("Memantine", "moderate impairment (CrCl 30-49)", 0.44, 1.97, "PK review of label data (Front Psychiatry 2022)"),
    ("Memantine", "severe impairment (CrCl 5-29)", 0.19, 2.33, "PK review of label data (Front Psychiatry 2022)"),
    ("Mirtazapine", "severe impairment (clearance -30%)", 0.15, 1.43, "secondary source (psychotropic renal-dosing guide)"),
    ("Mirtazapine", "severe impairment (clearance -50%)", 0.15, 2.00, "secondary source (psychotropic renal-dosing guide)"),
    ("Risperidone", "renal disease (clearance -60%; g assumed 0.30)", 0.30, 2.50, "label as cited in a psychotropic renal-disease review"),
]


def implied_fr_ren(g_ren, auc_ratio):
    """Fr_ren implied by an observed AUC ratio: AUC ratio = 1 / (1 - Fr_ren (1 - g))."""
    return (1.0 - 1.0 / auc_ratio) / (1.0 - g_ren)


def _mean_pki(kis):
    return 9.0 - float(np.mean([np.log10(k) for k in kis]))


def pki_audit_df():
    rows = []
    for (drug, rec), (kis, src) in PKI_EVIDENCE.items():
        dd = DRUG_DATABASE[drug]
        lit = _mean_pki(kis)
        rows.append({"Drug": drug, "Receptor": rec, "DB pKi": dd["pKi"][rec], "Evidence pKi (geo-mean)": round(lit, 2),
                     "Δ (DB − evidence)": round(dd["pKi"][rec] - lit, 2), "n sources": len(kis),
                     "Status": dd["pKi_status"][rec], "Sources": src})
    return pd.DataFrame(rows)


def db_status_df():
    rows = []
    for drug, dd in DRUG_DATABASE.items():
        st_ = list(dd["pKi_status"].values())
        rows.append({"Drug": drug, **{k: st_.count(k) for k in "VCUFXP"},
                     "PK (Fr)": dd["pk_status"], "QT": f'{dd["qt_category"]}/{dd["qt_status"]}', "Seizure": dd["seizure_status"]})
    return pd.DataFrame(rows)


def pk_calibration_df():
    rows = []
    for drug, setting, g, auc, src in PK_CALIBRATION_POINTS:
        fr = implied_fr_ren(g, auc)
        rows.append({"Drug": drug, "Setting": setting, "AUC ratio": auc, "Implied Fr_ren": round(fr, 2),
                     "Physical (Fr ≤ 1)": bool(fr <= 1.0 + 1e-9),   # >1 means GFR alone cannot explain the AUC rise (e.g., tubular secretion)
                     "DB Fr_ren": DRUG_DATABASE[drug]["Fr_renal"], "Δ (DB − implied)": round(DRUG_DATABASE[drug]["Fr_renal"] - fr, 2),
                     "Source": src})
    return pd.DataFrame(rows)


def qt_band_df():
    rows = []
    for drug, dd in DRUG_DATABASE.items():
        cat = dd["qt_category"]
        lo, hi = QT_BANDS.get(cat, (None, None))
        inb = None if lo is None else (lo - 1e-9 <= dd["Risk_QTc"] <= hi + 1e-9)
        rows.append({"Drug": drug, "Category": cat, "Meaning": QT_CATEGORY_NAMES[cat], "Risk_QTc": dd["Risk_QTc"],
                     "Band": "—" if lo is None else f"{lo:.2f}–{hi:.2f}", "In band": "n/a" if inb is None else ("yes" if inb else "NO"),
                     "Status": dd["qt_status"]})
    return pd.DataFrame(rows)


def seizure_df():
    return pd.DataFrame([{"Drug": d, "convulsant_index": dd["convulsant_index"], "Status": dd["seizure_status"], "Basis": dd["seizure_note"]}
                         for d, dd in DRUG_DATABASE.items()])


def delphi_template_df(kind):
    """Blank expert-rating sheets for the judgment-based parameters (export, circulate, re-import)."""
    cols = {f"Expert {i}": "" for i in range(1, 11)}
    if kind == "kappa":
        return pd.DataFrame([{"Receptor": r, "Symptom": s, "Current signed κ": k, **cols} for r, m in KAPPA.items() for s, k in m.items()])
    if kind == "zeta_beta":
        rows = [{"Item": f"ζ severity weight: {c['label']}", "Current value": c["zeta"], **cols} for c in DOMAINS.values()]
        rows += [{"Item": f"β liability: {c['label']} / {r}", "Current value": b, **cols} for c in DOMAINS.values() for r, b in c["beta"].items()]
        rows.append({"Item": "ζ history/interaction burden", "Current value": ZETA_HISTORY, **cols})
        return pd.DataFrame(rows)
    return pd.DataFrame([{"Curve": n, "Current x0": p[0], "Current k": p[1], "Range": f"{p[2]}–{p[3]}",
                          "Expert x0 (risk doubles at)": "", "Expert k (slope)": "", "Cohort evidence / OR": ""} for n, p in VULN_PARAMS.items()])


# -----------------------------------------------------------------------------
# 3.2 Parameter register: every placeholder, its calibration status in this version, and next steps
# -----------------------------------------------------------------------------
REGISTER_COLUMNS = ["ID", "Group", "Parameter (current value)", "Status", "Applied in this version", "Why uncertain",
                    "Suggested source / value", "How to calibrate", "How to validate", "Priority"]
PLACEHOLDER_REGISTER = [
    ("P01", "Exposure", f"Fr_renal / Fr_hepatic per drug; PHI_MIN = {PHI_MIN}; PHI_MAX = {PHI_MAX}; E_MAX_STANDARD = {E_MAX_STANDARD}",
     "Partially calibrated",
     "Fr re-checked for 13 drugs (5 corrected: sertraline, haloperidol, mirtazapine, carbamazepine, lamotrigine); memantine and mirtazapine checked against label AUC ratios (table in Calibration Center).",
     "Limits are global; drug-specific tolerance (therapeutic index) differs. 7 drugs' Fr remain unverified.",
     "Label PK in renal/hepatic impairment (AUC ratio vs normal) and labelled dose adjustments.",
     "Set Fr_ren from AUC ratios via Fr = (1 − 1/AUC)/(1 − g); set E_MAX per drug to the largest label-accepted AUC ratio; φ_min = lowest clearance fraction with a labelled regimen.",
     "Model lock/unlock decisions equal labelled 'avoid' thresholds for every drug; implied-Fr vs DB within ±0.2.", "High"),
    ("P02", "Exposure", f"INDUCER_HEPATIC_CL_FACTOR = {INDUCER_HEPATIC_CL_FACTOR} (all substrates)", "Placeholder", "Not changed.",
     "Induction magnitude is substrate- and inducer-specific.", "Drug-specific AUC changes with carbamazepine/phenytoin (labels, DDI studies).",
     "Per-drug-pair multiplier table; keep Fr_hepatic scaling.", "Predicted exposure change within ±30% of published DDI AUC ratios.", "High"),
    ("P03", "Affinity", f"SIGMA_PKI = {SIGMA_PKI}; CENSOR_PKI = {CENSOR_PKI}", "Calibrated (provisional)",
     "σ re-estimated from 6 multi-source SERT pairs (σ ≈ 0.38 → 0.40); per-entry σ widened to half the log-spread when sources disagree.",
     "Only 6 pairs; spread is assay- and species-dependent (venlafaxine SERT 8.9 vs 82 nM).",
     "All replicate Ki entries from PDSP, ChEMBL, IUPHAR per drug-receptor pair.",
     "Hierarchical log-normal model (σ by receptor class); treat '>' values as left-censored rather than fixing pKi = 5.",
     "Leave-one-source-out: held-out Ki inside the 90% interval in ≥85% of cases.", "High"),
    ("P04", "Anchors", "REFERENCE_ANCHORS (occupancy or unbound concentration at the standard dose)", "Partially calibrated",
     "SERT anchors set from PET (≈80% at usual minimum dose: escitalopram, sertraline, venlafaxine); memantine anchor lowered to 0.50 to match its µM-range NMDA Ki; D2 anchors still placeholders.",
     "Anchors fix the absolute exposure scale; most are order-of-magnitude estimates.",
     "PET occupancy at geriatric doses (D2: Kapur 2000; SERT: Meyer 2004 and the 2021 dose-occupancy review) or Css × fu × Kp,uu from label PK.",
     "Replace each anchor by an Emax dose-occupancy curve at the DB dose; add age-related clearance reduction; use occupancy SD instead of fixed logit-SD 0.3.",
     "Predicted occupancy within ±10 percentage points of PET at the standard dose for each drug with PET data.", "Critical"),
    ("P05", "D2 window", f"THETA_STAR = {THETA_STAR}; TAU_EPS = {TAU_EPS}", "Literature-based (θ*); placeholder (τ)",
     "θ* = upper bound of the 65–80% D2 PET window (Kapur 2000).", "Derived in younger adults; older and DLB/PDD patients may be harmed at lower occupancy.",
     "PET-EPS studies, geriatric PET data.", "Fit logistic EPS probability vs D2 occupancy by age stratum; θ* at 50% EPS probability, τ from slope.",
     "Calibration slope 0.8–1.2 for predicted vs observed EPS by occupancy tercile.", "High"),
    ("P06", "QTc", f"Risk_QTc per drug; QT_EFFECT_MS = {QT_EFFECT_MS}; QTC_MAX = {QTC_MAX}; QT_MIN_MEANINGFUL_MS = {QT_MIN_MEANINGFUL_MS}",
     "Partially calibrated",
     "Risk_QTc values now sit in CredibleMeds-category bands (KR/CR/PR/not-listed); 4 values corrected (amitriptyline, pimavanserin, brexpiprazole, bupropion); carbamazepine and mianserin unverified.",
     "Risk_QTc is an index, not a concentration–QTc slope; QT_EFFECT_MS is a single global value.",
     "Thorough-QT data and label ΔQTc at therapeutic exposure; current CredibleMeds lists.",
     "Per-drug linear concentration–QTc slope m_j; calibrate hazards to TdP reporting rates.",
     "Predicted ΔQTc within ±5 ms of label; Spearman with CredibleMeds category ≥0.7 (checked by band test).", "High"),
    ("P07", "Engagement", f"U_MIN = {U_MIN}", "Placeholder", "Not changed.", "Arbitrary cut-off for 'no modelled engagement'.",
     "Utility equal to 10–20% occupancy of the primary target.", "Sensitivity sweep 0.01–0.10; list drugs that flip.",
     "No drug with trial evidence for the rated symptom is excluded across the sweep.", "Medium"),
    ("P08", "Pharmacology", "RECEPTOR_PROFILE (α, tone); ALPHA_OVERRIDES: brexpiprazole D2 α = 0.43", "Placeholder", "Not changed.",
     "Intrinsic activity is assay-specific; tone is not measured per patient.", "Functional assays (cAMP/GTPγS, % Emax), e.g., Maeda 2014.",
     "α = Emax relative to full agonist; propagate tone range 0.5–1.0.", "Top-3 ranking robust (≤1 position change) across tone 0.5–1.0.", "Medium"),
    ("P09", "Need map", "KAPPA (symptom → receptor, signed), incl. SERT: depression 0.7, anxiety 0.6, irritability 0.4, agitation 0.3", "Placeholder",
     "Template exported for expert rating (Calibration Center).", "Magnitudes assigned by judgment.",
     "Meta-analytic effect sizes by symptom × mechanism (BPSD RCTs; antidepressants in dementia).",
     "Delphi panel (≥10 clinicians, 2 rounds, IQR ≤ 0.2) or regression of effect size on mechanism.",
     "Spearman between model utility and published effect sizes ≥ 0.6.", "Critical"),
    ("P10", "Safety weights", "DOMAINS ζ (0.10–0.30); caregiver scaling of sedation/orthostasis", "Placeholder", "Template exported for expert rating.",
     "Value judgments, not elicited.", "Swing-weighting or discrete-choice experiment with clinicians and caregivers.",
     "Normalize elicited weights; Dirichlet prior for Monte Carlo.", "Top-3 set unchanged for ±30% change of any single ζ in ≥80% of vignettes.", "Critical"),
    ("P11", "Liabilities", "DOMAINS β: H1 0.9, α1 0.9, M1 0.8, metabolic 0.5/0.7, hyponatremia SERT 0.5", "Placeholder", "Template exported for expert rating.",
     "Probability that full occupancy yields the event is assumed.", "Incidence by occupancy/class from cohorts and trials.",
     "Fit β so mean(λ·h) equals observed incidence.", "Predicted vs observed incidence within a factor of 1.5–2.", "High"),
    ("P12", "Vulnerability", "VULN_PARAMS (x0, k, range) for Morse, SBP drop, SAS, QTc, MMSE, seizure, HbA1c, BMI, sodium", "Placeholder",
     "Template exported for expert/cohort input.", "Centres/slopes judgment-based; several ranges illustrative.",
     "Risk-factor odds ratios and external cohorts.", "x0 where risk doubles vs reference; k from OR slope; test alternative link functions.",
     "Discrimination AUC ≥ 0.65 per curve in an external cohort.", "High"),
    ("P13", "Hyponatremia", f"hypoNa_index (DB field; carbamazepine 0.40); NA_HARD_LIMIT = {NA_HARD_LIMIT}; THIAZIDE_AMPLIFIER = {THIAZIDE_AMPLIFIER}; AGE_NA_AMPLIFIER_MAX = {AGE_NA_AMPLIFIER_MAX}; SERT_ACTIVE_PKI = {SERT_ACTIVE_PKI}",
     "Placeholder", "hypoNa_index moved into the database as a per-drug field.", "Only carbamazepine has a non-SERT index; amplifiers and Na limit are judgment-based.",
     "Pharmacovigilance/cohort data for antipsychotics, mirtazapine, trazodone, valproate; AGS Beers use-with-caution list.",
     "Convert ORs (thiazide + SSRI, age) into amplifiers a with 1−(1−λ)(1−a) = OR-implied risk.", "Observed Na < 130 incidence by risk stratum follows predicted ΔR ordering (Spearman ≥ 0.7).", "High"),
    ("P14", "Amplifiers", f"THYROID_QTC_AMPLIFIER = {THYROID_QTC_AMPLIFIER}; EPS_COMBO_AMPLIFIER = {EPS_COMBO_AMPLIFIER}; frailty × age ramp (75→90)", "Placeholder", "Not changed.",
     "Interaction strengths assumed.", "Cohort ORs (thyroid & QT; AChEI + SSRI + antipsychotic & EPS; frailty & falls).", "Convert ORs to amplifiers as in P13.", "As P12.", "Medium"),
    ("P15", "Soft burden", f"RHO_FAILURE = {RHO_FAILURE}; RHO_ACHEI_MAX = {RHO_ACHEI_MAX}; RHO_AED_DUPLICATE = {RHO_AED_DUPLICATE}; RHO_POLYPHARMACY = {RHO_POLYPHARMACY}; ZETA_HISTORY = {ZETA_HISTORY}",
     "Placeholder", "Not changed.", "Judgment-based.", "Re-challenge outcome data; anticholinergic + AChEI pharmacoepidemiology; polypharmacy event rates.",
     "Elicit as in P10 or estimate from claims data.", "±50% on each ρ leaves the top-3 set unchanged in ≥80% of vignettes.", "Medium"),
    ("P16", "Combinations", f"COMBO_SUPERIORITY_DELTA = {COMBO_SUPERIORITY_DELTA}; COMBO_STRONG_MARGIN = {COMBO_STRONG_MARGIN}; COMBO_U_TARGET = {COMBO_U_TARGET}",
     "Placeholder", "Not changed.", "Heuristic thresholds.", "Minimal clinically important index difference from expert-rated vignette pairs.",
     "δ = smallest difference experts distinguish with ≥75% agreement.", "Combination suggestions concordant with experts in ≥80% of paired vignettes.", "Medium"),
    ("P17", "Background", f"NONDB_BACKGROUND_HAZARD = {NONDB_BACKGROUND_HAZARD}", "Placeholder", "Not changed.", "Fixed anticholinergic hazard.",
     "ACB score → cognitive-decline/delirium risk from cohorts.", "Fit h = 1 − exp(−γ·ACB).", "Predicted vs observed risk by ACB stratum within a factor of 1.5.", "Medium"),
    ("P18", "Database", "pKi values (220 entries), Fr, Risk_QTc, convulsant_index, dosage text", "Partially calibrated",
     "Audit of 60 affinity entries against multi-source evidence: 37 consistent (V), 20 corrected (C; 18 by |Δ|>0.5 plus trazodone α1 and SERT), 20 unverified (U), 3 proxies (P). QT and seizure indices re-based (see P06, P19).",
     "Many remaining entries are floors (F) or unverified (U); dosage text not re-checked.",
     "PDSP, ChEMBL, IUPHAR, labels.", "Scripted extraction of Ki replicates with assay metadata; keep source columns in the DB.",
     "100% of inputs cited; |ΔpKi| ≤ 0.5 vs median of independent sources.", "High"),
    ("P19", "Seizure", "convulsant_index per drug", "Partially calibrated",
     "Bupropion (SIR 4.08 → 0.76), amitriptyline (SIR 1.58 → 0.37), olanzapine/quetiapine/risperidone (label incidence 0.9/0.8/0.3% on one scale), mirtazapine re-based; others unverified.",
     "Index mixes incidence-based and SIR-based sources.", "Label seizure incidences and SIRs from pharmacoepidemiology.",
     "Express every index as 1 − 1/RR against untreated baseline.", "Ordering agrees with published relative risks (Spearman ≥ 0.7).", "Medium"),
    ("P20", "Uncertainty", "Sampler widths: anchor logit-SD 0.3 (conc. 0.5 log-SD), ζ log-SD 0.3, κ/β concentration ν = 20, vulnerability jitter", "Placeholder", "Not changed.",
     "Chosen for plausibility, not estimated.", "Posterior distributions from the calibration steps above.", "Replace fixed widths by calibrated posteriors.",
     "90% intervals cover held-out expert-score variation or bootstrap replicates in ≥85% of cases.", "Medium"),
]

# -----------------------------------------------------------------------------
# 3.3 VALIDATION & IMPROVEMENT ROADMAPS
# -----------------------------------------------------------------------------
VALIDATION_COLUMNS = ["Tier", "What is tested", "Acceptance criterion (suggested)", "Status in this app"]
VALIDATION_PLAN = [
    ("V1 Software verification", "Bounds, monotonicity, competitive binding, exact vulnerability endpoints, database integrity", "All checks pass on every code change", "Implemented (self-checks)"),
    ("V2 Face validity", "Hard constraints and mechanism-appropriate rankings in canonical vignettes", "≥90% agreement with panel on hard constraints; ≥80% top-3 concordance on ≥30 vignettes", "Partly implemented; vignettes to be expanded"),
    ("V3 Input verification", "pKi, Fr, QT and seizure inputs against independent sources", "|ΔpKi| ≤ 0.5 vs median of sources; every input cited", "Partly implemented (evidence tables, band and AUC checks)"),
    ("V4 Anchor verification", "Predicted occupancy at standard dose vs PET", "Within ±10 percentage points", "Not done (anchors partly placeholders)"),
    ("V5 Hazard calibration", "Predicted vs observed adverse-event incidence", "Calibration slope 0.8–1.2; within factor 1.5–2", "Not done"),
    ("V6 Retrospective concordance", "De-identified cases vs expert/guideline choice", "Kendall τ ≥ 0.6; top-3 agreement ≥ 80%; no hard-constraint violations among recommendations", "Not done"),
    ("V7 Uncertainty coverage", "Monte Carlo intervals vs held-out variability", "Coverage of 90% intervals ≥ 85%", "Not done"),
    ("V8 Prospective shadow study", "Silent-mode comparison with clinician decisions and outcomes", "Pre-registered protocol; regulatory (decision-support software) review before any use", "Not done"),
]

IMPROVEMENT_COLUMNS = ["Phase", "Workstream", "Action", "Deliverable / success metric", "Depends on"]
IMPROVEMENT_PLAN = [
    ("1 · 0–3 months", "Data", "Script PDSP/ChEMBL/IUPHAR extraction for all 20 drugs × 11 receptors; store replicate Ki with assay metadata; replace all U/F entries", "100% of affinity inputs V or C; provenance columns in the DB", "—"),
    ("1 · 0–3 months", "Data", "Verify unresolved items: SERT floors for quetiapine/olanzapine/brexpiprazole/pimavanserin, QT category of carbamazepine and mianserin, Fr of 7 unverified drugs, dosage text", "Zero 'unverified' flags on safety-critical fields", "Extraction script"),
    ("1 · 0–3 months", "Model scope", "Replace proxy rows (GABA-A for valproate/gabapentin) by mechanism-appropriate targets (α2δ, GABA-transaminase/Na-channel potency); add DAT, 5-HT1A, α2C, D3 and run an ablation study to keep only targets that change rankings", "No proxy (P) entries; panel justified by ablation", "Data extraction"),
    ("1 · 0–3 months", "Elicitation", "Run Delphi (κ, ζ, β) and vulnerability-curve rating with ≥10 clinicians and caregiver input using the exported templates", "Calibrated κ/ζ/β with inter-rater agreement (IQR ≤ 0.2)", "Templates (included)"),
    ("2 · 3–6 months", "Pharmacology", "Replace anchors by dose-occupancy curves (PET/PK); add geriatric clearance; per-drug E_MAX; per-pair DDI multipliers", "Predicted occupancy within ±10 pp of PET", "Data, PET literature"),
    ("2 · 3–6 months", "Safety", "Concentration–QTc slopes per drug; calibrate hazards (β) to incidence; fit vulnerability curves and hyponatremia amplifiers to cohort ORs", "Calibration slope 0.8–1.2", "Cohort data access"),
    ("2 · 3–6 months", "Statistics", "Hierarchical Bayesian calibration replacing hand-set σ; automated Sobol sensitivity with stability criteria", "90% interval coverage ≥ 85%", "Calibrated inputs"),
    ("2 · 3–6 months", "Engineering", "Move constants/DB to versioned JSON/YAML with checksums; unit tests in CI; audit log of inputs, outputs and DB version; separate model package from UI", "Reproducible runs; CI green", "—"),
    ("3 · 6–12 months", "Validation", "Expand vignette suite to ≥30 expert-authored cases; retrospective concordance on ≥200 de-identified cases", "Kendall τ ≥ 0.6; top-3 agreement ≥ 80%", "Phase 1–2"),
    ("3 · 6–12 months", "Usability", "Human-factors testing of the mobile workflow; uncertainty and 'indeterminate' messaging; clinician override capture", "Task-success and comprehension targets met", "—"),
    ("3 · 6–12 months", "Model extensions", "Time-course PK/PD (peak vs trough occupancy, e.g., quetiapine), prolactin domain, SERT-mediated bleeding domain, delirium/anticholinergic domain across M1–M5, CYP2D6/2C19 phenotype", "Extensions pass ablation and calibration gates", "Phase 2"),
    ("4 · 12+ months", "Deployment readiness", "Prospective shadow-mode study; clinical safety case; software-as-medical-device classification and regulatory review; post-market monitoring plan", "Approved protocol and regulatory pathway before any clinical use", "Phase 3"),
]

# =============================================================================
# 4. MODEL ENGINE
# =============================================================================
# -----------------------------------------------------------------------------
# 4.1 NUMERICS & UNCERTAINTY SAMPLER
# -----------------------------------------------------------------------------
def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def rescaled_logistic(x, x0, k, xmin, xmax, invert=False):
    """Logistic vulnerability rescaled so that lambda(xmin)=0 and lambda(xmax)=1 exactly."""
    x = np.clip(x, xmin, xmax)
    lo, hi = _sig(k * (xmin - x0)), _sig(k * (xmax - x0))
    lam = np.clip((_sig(k * (x - x0)) - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - lam if invert else lam


class Sampler:
    """Draws uncertain model inputs. n == 1 returns nominal (median) values.
    `vary` selects which uncertainty groups are randomised (used for sensitivity analysis)."""
    GROUPS = ("affinity", "anchor", "need", "liability", "severity", "vulnerability")
    GROUP_LABELS = {
        "affinity": "Ki / pKi (inter-assay noise)",
        "anchor": "Reference anchors (occupancy / concentration)",
        "need": "Symptom → receptor coupling (κ)",
        "liability": "Risk liabilities (β)",
        "severity": "Severity weights (ζ)",
        "vulnerability": "Vulnerability curves (x0, k)",
    }

    def __init__(self, n=1, seed=0, vary=None):
        self.n = int(n)
        self.rng = np.random.default_rng(seed)
        self.vary = set(self.GROUPS) if vary is None else set(vary)
        if self.n == 1:
            self.vary = set()
        self.cache = {}

    def _memo(self, key, make):
        if key not in self.cache:
            self.cache[key] = make()
        return self.cache[key]

    def pki(self, drug, rec, mu):
        def make():
            if "affinity" in self.vary and mu > CENSOR_PKI:
                return mu + PKI_SIGMA.get((drug, rec), SIGMA_PKI) * self.rng.standard_normal(self.n)
            return np.full(self.n, float(mu))
        return self._memo(("pki", drug, rec), make)

    def anchor_occ(self, drug, occ):
        def make():
            if "anchor" in self.vary:
                z = np.log(occ / (1.0 - occ)) + 0.3 * self.rng.standard_normal(self.n)
                return _sig(z)
            return np.full(self.n, float(occ))
        return self._memo(("anchor", drug), make)

    def anchor_conc(self, drug, conc):
        def make():
            if "anchor" in self.vary:
                return conc * np.exp(0.5 * self.rng.standard_normal(self.n))
            return np.full(self.n, float(conc))
        return self._memo(("anchorc", drug), make)

    def zeta(self, key, base):
        def make():
            if "severity" in self.vary:
                return base * np.exp(0.3 * self.rng.standard_normal(self.n))
            return np.full(self.n, float(base))
        return self._memo(("zeta", key), make)

    def beta(self, key, mean, nu=20.0):
        def make():
            if "liability" in self.vary:
                return self.rng.beta(nu * mean, nu * (1.0 - mean), self.n)
            return np.full(self.n, float(mean))
        return self._memo(("beta", key), make)

    def kappa(self, rec, sym, mean_abs, nu=20.0):
        def make():
            if "need" in self.vary:
                return self.rng.beta(nu * mean_abs, nu * (1.0 - mean_abs), self.n)
            return np.full(self.n, float(mean_abs))
        return self._memo(("kappa", rec, sym), make)

    def vuln_params(self, name):
        x0, k, xmin, xmax = VULN_PARAMS[name]
        def make():
            if "vulnerability" in self.vary:
                return (x0 + 0.05 * (xmax - xmin) * self.rng.standard_normal(self.n),
                        k * np.exp(0.2 * self.rng.standard_normal(self.n)))
            return np.full(self.n, x0), np.full(self.n, k)
        return self._memo(("vuln", name), make)


# -----------------------------------------------------------------------------
# 4.2 NEED WEIGHTS & PATIENT VULNERABILITY
# -----------------------------------------------------------------------------
def compute_need(v, S):
    """Noisy-OR need weights w_r in [0,1] and desired direction s_r in {-1,0,+1}."""
    def make():
        w, s = {}, {}
        for rec, mapping in KAPPA.items():
            surv, net = np.ones(S.n), np.zeros(S.n)
            for sym, kap in mapping.items():
                vs = v.get(sym, 0.0)
                if vs <= 0.0:
                    continue
                mag = S.kappa(rec, sym, abs(kap))
                surv = surv * (1.0 - vs * mag)
                net = net + vs * mag * np.sign(kap)
            w[rec] = 1.0 - surv
            s[rec] = np.sign(net)
        return w, s
    return S._memo(("need",), make)


def patient_vulnerability(ctx, S):
    """lambda_d in [0,1] for each risk domain (rescaled logistics + frailty/thyroid amplifiers)."""
    def make():
        def rl(name, x, invert=False):
            x0, k = S.vuln_params(name)
            _, _, xmin, xmax = VULN_PARAMS[name]
            return rescaled_logistic(x, x0, k, xmin, xmax, invert)

        def amplify(lam, a):
            return 1.0 - (1.0 - lam) * (1.0 - a)

        na_amp = 1.0 - (1.0 - (THIAZIDE_AMPLIFIER if ctx["thiazide"] else 0.0)) * \
                       (1.0 - AGE_NA_AMPLIFIER_MAX * float(np.clip((ctx["age"] - 65.0) / 20.0, 0.0, 1.0)))
        frail_amp = ctx["frailty_score"] * float(np.clip((ctx["age"] - 75.0) / 15.0, 0.0, 1.0))
        lam = {
            "sedation": amplify(rl("sedation_falls", ctx["morse"]), frail_amp),
            "orthostasis": rl("orthostasis", ctx["sbp_drop"]),
            "eps": np.ones(S.n) if ctx["dementia_subtype"] in LEWY_SUBTYPES else rl("eps", ctx["sas"]),
            "qtc": amplify(rl("qtc", ctx["qtc_ms"]), THYROID_QTC_AMPLIFIER if ctx["has_thyroid_dysfunction"] else 0.0),
            "cognition": amplify(rl("cognition", ctx["mmse"], invert=True), frail_amp),
            "seizure": rl("seizure", ctx["seizure_freq_year"]),
            "metabolic": 0.5 * (rl("hba1c", ctx["hba1c"]) + rl("bmi", ctx["bmi"])),
            "hyponatremia": amplify(rl("sodium", ctx["sodium"], invert=True), na_amp),
        }
        return lam
    return S._memo(("vulnerability_all",), make)


# -----------------------------------------------------------------------------
# 4.3 EXPOSURE & OCCUPANCY
# -----------------------------------------------------------------------------
def clearance_fraction(drug, dd, ctx):
    """phi = fraction of normal total clearance remaining (renal, hepatic, AED CYP induction)."""
    g_ren = float(np.clip(ctx["egfr"] / 90.0, 0.0, 1.0))
    g_hep = float(np.clip(1.0 - ctx["lft_impairment"], 0.0, 1.0))
    own_aed = AED_NAME_MAP.get(drug)
    induced = any(a in INDUCER_AEDS and a != own_aed for a in ctx["active_aeds"])
    f_hep = INDUCER_HEPATIC_CL_FACTOR if induced else 1.0
    return 1.0 - dd["Fr_renal"] * (1.0 - g_ren) - dd["Fr_hepatic"] * (1.0 - g_hep * f_hep)


def dose_factor(phi_raw, ctx):
    if ctx["dose_policy"] == "auto":
        return float(min(1.0, np.clip(phi_raw, PHI_MIN, PHI_MAX)))
    return 1.0


def drug_pharmacology(drug, dd, ctx, S, delta=None):
    """Exposure, unbound target concentration and occupancy ratios x_r = C / Ki_r."""
    phi_raw = clearance_fraction(drug, dd, ctx)
    phi = float(np.clip(phi_raw, PHI_MIN, PHI_MAX))
    if delta is None:
        delta = dose_factor(phi_raw, ctx)
    m = delta / phi                                   # exposure multiplier vs reference
    pk = {r: S.pki(drug, r, dd["pKi"].get(r, 0.0)) for r in RECEPTORS_ALL}
    anchor = REFERENCE_ANCHORS.get(drug, DEFAULT_ANCHOR)
    if anchor[0] == "occ":
        _, arec, occ = anchor
        occ_s = S.anchor_occ(drug, occ)
        c_ref = (10.0 ** (9.0 - pk[arec])) * occ_s / (1.0 - occ_s)
    else:
        c_ref = S.anchor_conc(drug, anchor[1])
    conc = m * c_ref
    x, eff = {}, {}
    for r in RECEPTORS_ALL:
        if dd["pKi"].get(r, 0.0) <= CENSOR_PKI:
            x[r] = np.zeros(S.n)
        else:
            x[r] = conc / (10.0 ** (9.0 - pk[r]))
    for r in THERAPEUTIC_RECEPTORS:
        a = ALPHA_OVERRIDES.get((drug, r), RECEPTOR_PROFILE[r]["alpha"])
        eff[r] = a - RECEPTOR_PROFILE[r]["tone"]
    return {"name": drug, "dd": dd, "phi_raw": phi_raw, "phi": phi, "E": 1.0 / phi, "delta": delta,
            "m": m, "x": x, "eff": eff, "conc": conc}


# -----------------------------------------------------------------------------
# 4.4 HAZARDS & REGIMEN SCORING
# -----------------------------------------------------------------------------
def compute_hazards(items, S):
    """Drug hazards h_d in [0,1] for a regimen of 1-2 competitively binding agents."""
    X = {r: sum(it["x"][r] for it in items) for r in RECEPTORS_ALL}
    th_tot = {r: X[r] / (1.0 + X[r]) for r in RECEPTORS_ALL}

    def th_i(it, r):
        return it["x"][r] / (1.0 + X[r])

    h = {}
    h["sedation"] = S.beta(("sed", "H1"), 0.9) * th_tot["H1"]
    h["orthostasis"] = S.beta(("ortho", "α1"), 0.9) * th_tot["α1"]
    d2_net = sum(th_i(it, "D2") * np.clip(-it["eff"]["D2"], 0.0, 1.0) for it in items)
    h["eps"] = _sig((d2_net - THETA_STAR) / TAU_EPS)
    surv_q, surv_s, dqt = np.ones(S.n), np.ones(S.n), np.zeros(S.n)
    for it in items:
        rq = min(it["dd"]["Risk_QTc"], 0.999)
        rc = min(it["dd"]["convulsant_index"], 0.999)
        surv_q = surv_q * (1.0 - rq) ** it["m"]
        surv_s = surv_s * (1.0 - rc) ** it["m"]
        dqt = dqt + QT_EFFECT_MS * it["dd"]["Risk_QTc"] * it["m"]
    h["qtc"], h["seizure"] = 1.0 - surv_q, 1.0 - surv_s
    h["cognition"] = S.beta(("cog", "M1"), 0.8) * th_tot["M1"]
    h["metabolic"] = 1.0 - (1.0 - S.beta(("met", "H1"), 0.5) * th_tot["H1"]) * \
                           (1.0 - S.beta(("met", "5HT2C"), 0.7) * th_tot["5HT2C"])
    surv_h = np.ones(S.n)
    for it in items:
        surv_h = surv_h * (1.0 - min(it["dd"].get("hypoNa_index", 0.0), 0.999)) ** it["m"]
    h["hyponatremia"] = 1.0 - (1.0 - S.beta(("hypona", "SERT"), 0.5) * th_tot["SERT"]) * surv_h
    return h, dqt, th_tot, th_i


def background_items(ctx, S, exclude):
    """Pharmacology of active database agents that stay on board (add-on mode only)."""
    if ctx["background_mode"] != "addon":
        return []
    out = []
    for dn in ctx["active_db_drugs"]:
        if dn in exclude:
            continue
        out.append(S._memo(("ph_active", dn),
                           lambda dn=dn: drug_pharmacology(dn, DRUG_DATABASE[dn], ctx, S, delta=1.0)))
    return out


def nondb_background(ctx, S):
    """Independent background hazards from non-database agents (e.g., anticholinergics)."""
    surv = {d: np.ones(S.n) for d in DOMAINS}
    for nd in ctx["active_nondb"]:
        for d, val in NONDB_BACKGROUND_HAZARD.get(nd, {}).items():
            surv[d] = surv[d] * (1.0 - val)
    return {d: 1.0 - sv for d, sv in surv.items()}


def score_regimen(items, ctx, S, v, prior_history):
    """Utility U, marginal risks dR_d, soft burden rho and net score M for 1-2 candidate agents.
    Add-on mode: active database agents are evaluated jointly with the candidate (competitive binding),
    and utility / hazards are the increments over the current regimen (not an independence discount)."""
    names = [it["name"] for it in items]
    bg_items = background_items(ctx, S, set(names))
    all_items = bg_items + items
    h_all, _, _, th_i_all = compute_hazards(all_items, S)
    if bg_items:
        h_bg, _, _, th_i_bg = compute_hazards(bg_items, S)
    else:
        h_bg, th_i_bg = {d: np.zeros(S.n) for d in DOMAINS}, None
    dqt = np.zeros(S.n)
    for it in items:
        dqt = dqt + QT_EFFECT_MS * it["dd"]["Risk_QTc"] * it["m"]

    # marginal therapeutic utility of the candidate regimen over the current one
    w, s = compute_need(v, S)
    sum_w = sum(w[r] for r in THERAPEUTIC_RECEPTORS)
    num = np.zeros(S.n)
    contrib = {}
    for r in THERAPEUTIC_RECEPTORS:
        eff_all = sum(th_i_all(it, r) * it["eff"][r] for it in all_items)
        eff_bg = sum(th_i_bg(it, r) * it["eff"][r] for it in bg_items) if bg_items else 0.0
        contrib[r] = w[r] * s[r] * (eff_all - eff_bg)
        num = num + contrib[r]
    U = np.where(sum_w > 1e-9, num / np.maximum(sum_w, 1e-9), 0.0)
    theta_c = {r: sum(th_i_all(it, r) for it in items) for r in RECEPTORS_ALL}

    # marginal risk per domain: lambda_d * (1 - h_nonDB) * (h_regimen - h_background)
    lam = dict(patient_vulnerability(ctx, S))
    if ctx["has_achei"] and ctx["has_ssri"] and any(it["dd"]["category"] in ANTIPSYCHOTIC_CATEGORIES for it in items):
        lam["eps"] = 1.0 - (1.0 - lam["eps"]) * (1.0 - EPS_COMBO_AMPLIFIER)
    nb = nondb_background(ctx, S)
    zeta_scale = {"sedation": ctx["pref_sedation"], "orthostasis": ctx["pref_falls"]}
    dR, weighted = {}, {}
    for d, cfg in DOMAINS.items():
        dR[d] = lam[d] * (1.0 - nb[d]) * np.maximum(h_all[d] - h_bg[d], 0.0)
        weighted[d] = S.zeta(d, cfg["zeta"] * zeta_scale.get(d, 1.0)) * dR[d]
    risk_total = sum(weighted.values())

    # soft history / interaction burden (noisy-OR)
    surv = np.ones(S.n)
    for it in items:
        nm = it["name"]
        hist = prior_history.get(nm)
        if hist and hist["outcome"] == "Treatment Failure / Ineffective":
            surv = surv * (1.0 - RHO_FAILURE)
        if ctx["has_achei"]:
            surv = surv * (1.0 - RHO_ACHEI_MAX * th_i_all(it, "M1"))
        if AED_NAME_MAP.get(nm) in ctx["active_aeds"]:
            surv = surv * (1.0 - RHO_AED_DUPLICATE)
    if len(items) > 1:
        surv = surv * (1.0 - RHO_POLYPHARMACY)
    rho = 1.0 - surv
    zeta_h = S.zeta("history", ZETA_HISTORY)
    M = U - risk_total - zeta_h * rho

    qtc_ok = ~((ctx["qtc_ms"] + dqt > QTC_MAX) & (dqt >= QT_MIN_MEANINGFUL_MS))
    return {"U": U, "dR": dR, "weighted": weighted, "risk_total": risk_total, "rho": rho,
            "hist_pen": zeta_h * rho, "M": M, "qtc_ok": qtc_ok, "dqt": dqt,
            "indicated": U > U_MIN, "theta": theta_c, "contrib": contrib, "w": w, "s": s}


# -----------------------------------------------------------------------------
# 4.5 STAGE A CONSTRAINTS & CLINICAL NOTES
# -----------------------------------------------------------------------------
def static_lock_reasons(drug, dd, ctx, prior_history):
    reasons = []
    if drug in st.session_state.ruled_out:
        reasons.append("Ruled Out by Clinician")
    if drug == "Bupropion" and (ctx["seizure_freq_year"] > 0.0 or len(ctx["active_aeds"]) > 0):
        reasons.append("Contraindicated: Bupropion strictly prohibited in patients with active seizure history/epilepsy")
    if ctx["has_maoi"] and drug in SEROTONERGIC_DRUGS:
        reasons.append("Contraindicated: Concomitant MAOI exposure. Mandatory 14-day washout required (5 weeks for Fluoxetine)")
    if ctx["sodium"] < NA_HARD_LIMIT and (dd["pKi"].get("SERT", 0.0) >= SERT_ACTIVE_PKI or dd.get("hypoNa_index", 0.0) >= 0.30):
        reasons.append(f"Contraindicated: baseline serum Na {ctx['sodium']:.0f} mmol/L is below {NA_HARD_LIMIT:.0f} (SERT-active / SIADH-associated agent)")
    hist = prior_history.get(drug)
    if hist and hist["outcome"] == "Severe Adverse Effects / Intolerant":
        reasons.append(f"Historical Intolerance: Discontinued due to adverse effects ({hist['dosage']}).")
    if ctx["dementia_subtype"] in LEWY_SUBTYPES and dd.get("d2_full_antagonist"):
        reasons.append("Contraindicated: Full D2 antagonist in DLB/PDD etiology")
    return reasons


def clinical_notes(drug, dd, ctx, prior_history, theta_m1):
    note = ""
    hist = prior_history.get(drug)
    if hist:
        if hist["outcome"] == "Treatment Failure / Ineffective":
            note += f"Prior trial at {hist['dosage']} resulted in failure. Higher dose or mechanism switch advised. "
        elif hist["outcome"] in ["Partial Response / Tolerated", "Active Regimen - Ongoing"]:
            note += f"Currently active/prior tolerated agent at {hist['dosage']}. Evaluate dose optimization. "
    if ctx["has_achei"] and theta_m1 >= 0.30:
        note += "⚠️ Appreciable muscarinic occupancy directly opposes baseline AChEI efficacy and escalates delirium/fall risks. "
    if any(a in INDUCER_AEDS for a in ctx["active_aeds"]) and dd["category"] in [
            "Atypical Antipsychotic", "Typical Antipsychotic", "Antidepressant (SSRI)", "Antidepressant (SNRI)"]:
        note += "⚠️ Potent CYP induction (Phenytoin/Carbamazepine) lowers exposure (modelled via clearance factor); dosage adjustment required. "
    if ctx["has_achei"] and ctx["has_ssri"] and dd["category"] in ANTIPSYCHOTIC_CATEGORIES:
        note += " ⚠️ Potentiated EPS Risk: Co-administered AChEI + SSRI elevates EPS incidence via 5-HT2 modulation (modelled as raised EPS vulnerability)."
    if dd["pKi"].get("SERT", 0.0) >= SERT_ACTIVE_PKI or dd.get("hypoNa_index", 0.0) >= 0.30:
        note += "🧂 Hyponatremia surveillance: check serum Na at baseline and at ~2 and ~4 weeks (higher risk with age, thiazides, low baseline Na). "
    return note


# -----------------------------------------------------------------------------
# 4.6 CANDIDATE EVALUATION, MONTE CARLO & SENSITIVITY
# -----------------------------------------------------------------------------
def evaluate_single(drug, dd, ctx, S, v, prior_history):
    """Full v2 evaluation of one candidate monotherapy. Arrays have length S.n."""
    ph = drug_pharmacology(drug, dd, ctx, S)
    sc = score_regimen([ph], ctx, S, v, prior_history)
    locks = static_lock_reasons(drug, dd, ctx, prior_history)
    if ctx["dose_policy"] == "standard" and ph["E"] > E_MAX_STANDARD:
        locks.append(f"Standard dose infeasible: expected exposure ×{ph['E']:.1f} of reference (limit ×{E_MAX_STANDARD:.0f}); "
                     f"dose reduction ≈ ×{min(1.0, ph['phi']):.2f} required (use the auto-reduce dose policy to rank at adjusted dose)")
    static_ok = len(locks) == 0
    feasible = static_ok & sc["qtc_ok"] & sc["indicated"]
    M = np.where(feasible, sc["M"], -np.inf)
    return {"drug": drug, "ph": ph, "sc": sc, "locks": locks, "feasible": feasible, "M": M}


def result_row(res, ctx, prior_history):
    """Scalarise a nominal (n=1) evaluation into the dictionary used by the UI."""
    drug, ph, sc = res["drug"], res["ph"], res["sc"]
    dd = ph["dd"]
    locks = list(res["locks"])
    if not bool(sc["qtc_ok"][0]):
        locks.append(f"Contraindicated: projected QTc {ctx['qtc_ms'] + float(sc['dqt'][0]):.0f} ms exceeds {QTC_MAX:.0f} ms limit")
    if not bool(sc["indicated"][0]):
        locks.append("Not indicated: no modelled receptor engagement for the current symptom profile (U ≤ 0.02)")
    hard_locked = len(locks) > 0
    M = float(sc["M"][0])
    theta = {r: float(sc["theta"][r][0]) for r in RECEPTORS_ALL}
    row = {
        "Drug": drug, "Category": dd["category"],
        "Net Score (Mj)": round(100 * M, 1) if not hard_locked else -999.0,
        "Raw_Mj": M if not hard_locked else -999.0,
        "Therapeutic Gain": round(100 * float(sc["U"][0]), 1),
        "Risk Deductions": round(100 * float(sc["risk_total"][0]), 1),
        "History Burden": round(100 * float(sc["hist_pen"][0]), 1),
        "Domain Breakdown": {DOMAINS[d]["label"]: round(100 * float(sc["weighted"][d][0]), 1) for d in DOMAINS},
        "Exposure Factor (E)": round(ph["E"], 2),
        "Suggested Dose Scaling": round(min(1.0, ph["phi"]), 2),
        "Sedation risk": int(round(100 * float(sc["dR"]["sedation"][0]))),
        "Orthostatic risk": int(round(100 * float(sc["dR"]["orthostasis"][0]))),
        "EPS risk": int(round(100 * float(sc["dR"]["eps"][0]))),
        "Hyponatremia risk": int(round(100 * float(sc["dR"]["hyponatremia"][0]))),
        "Occupancy": theta, "Receptor Contrib": {r: 100.0 * float(sc["contrib"][r][0]) / max(1e-9, float(sum(sc["w"][q][0] for q in THERAPEUTIC_RECEPTORS)))
                             for r in THERAPEUTIC_RECEPTORS},
        "Projected dQTc (ms)": round(float(sc["dqt"][0]), 1),
        "Hard Locked": hard_locked, "Lock Reason": "; ".join(locks),
        "Clinical Correlation Note": clinical_notes(drug, dd, ctx, prior_history, theta["M1"]),
        "Dosage": dd["dosage"], "Warnings": dd["warnings"], "_pharm": ph,
    }
    return row


def run_monte_carlo(ctx, v, prior_history, n, seed=2026):
    """Propagate input uncertainty to M_j; returns per-drug rank-first probability and intervals."""
    S = Sampler(n, seed)
    names = list(DRUG_DATABASE.keys())
    Ms = np.full((len(names), n), -np.inf)
    for i, nm in enumerate(names):
        Ms[i] = evaluate_single(nm, DRUG_DATABASE[nm], ctx, S, v, prior_history)["M"]
    feas = np.isfinite(Ms)
    any_feas = feas.any(axis=0)
    best = np.where(any_feas, np.argmax(Ms, axis=0), -1)
    out = {}
    for i, nm in enumerate(names):
        vals = Ms[i][feas[i]]
        out[nm] = {
            "P_first": float(np.mean(best == i)),
            "P_feasible": float(np.mean(feas[i])),
            "p05": float(np.percentile(vals, 5)) if len(vals) >= 20 else np.nan,
            "p50": float(np.percentile(vals, 50)) if len(vals) >= 20 else np.nan,
            "p95": float(np.percentile(vals, 95)) if len(vals) >= 20 else np.nan,
        }
    return out


def variance_shares(drug, ctx, v, prior_history, n, seed=2027):
    """One-at-a-time variance contribution of each uncertainty group to M for one drug."""
    var = {}
    for g in Sampler.GROUPS:
        S = Sampler(n, seed, vary={g})
        m = evaluate_single(drug, DRUG_DATABASE[drug], ctx, S, v, prior_history)["M"]
        m = m[np.isfinite(m)]
        var[g] = float(np.var(m)) if len(m) > 2 else 0.0
    tot = sum(var.values())
    return {g: (val / tot if tot > 0 else 0.0) for g, val in var.items()}


# -----------------------------------------------------------------------------
# 4.7 COMBINATION REGIMENS & CROSS-TITRATION
# -----------------------------------------------------------------------------
# Multi-agent combination regimen (competitive-binding occupancy, same scoring rules)
def evaluate_combination_regimens(point_rows, ctx, S0, v, prior_history):
    valid = [r for r in point_rows if not r["Hard Locked"]]
    if not valid:
        return {"mode": "NONE", "is_viable": False}
    best_mono = valid[0]
    combos = []
    for r1, r2 in itertools.combinations(valid, 2):
        sc = score_regimen([r1["_pharm"], r2["_pharm"]], ctx, S0, v, prior_history)
        if not bool(sc["qtc_ok"][0]) or not bool(sc["indicated"][0]):
            continue
        M = float(sc["M"][0])
        combos.append({"regimen": [r1["Drug"], r2["Drug"]], "M": M, "U": float(sc["U"][0]),
                       "margin": M - best_mono["Raw_Mj"]})
    combos.sort(key=lambda c: c["M"], reverse=True)
    best = combos[0] if combos else None
    best_u = best_mono["Therapeutic Gain"] / 100.0
    if best and best["margin"] >= COMBO_SUPERIORITY_DELTA and (
            best_u < COMBO_U_TARGET or best["margin"] >= COMBO_STRONG_MARGIN or ctx.get("polypharmacy_branch", False)):
        return {"mode": "COMBINATION", "is_viable": True, "regimen": best["regimen"],
                "score": round(100 * best["M"], 1), "margin": round(100 * best["margin"], 1)}
    return {"mode": "MONOTHERAPY", "is_viable": False}


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
# 5. VERIFICATION: SELF-CHECKS
# =============================================================================
def _test_ctx(**kw):
    ctx = {"age": 80.0, "frailty_score": 0.3, "seizure_freq_year": 0.0, "active_aeds": [], "hba1c": 6.0, "bmi": 25.0,
           "sbp_drop": 12.0, "has_thyroid_dysfunction": False, "has_maoi": False, "has_achei": False, "has_ssri": False,
           "polypharmacy_branch": False, "mmse": 14.0, "qtc_ms": 430.0, "egfr": 60.0, "lft_impairment": 0.0, "morse": 40.0,
           "sas": 4.0, "dementia_subtype": "Alzheimer's Disease (AD)", "pref_sedation": 1.0, "pref_falls": 1.0,
           "dose_policy": "standard", "background_mode": "addon", "active_db_drugs": [], "active_nondb": [],
           "sodium": 140.0, "thiazide": False}
    ctx.update(kw)
    return ctx


def _test_v(**kw):
    v0 = {k: 0.0 for k in ["delusions", "hallucinations", "agitation", "depression", "anxiety", "euphoria", "apathy",
                           "disinhibition", "irritability", "motor", "sleep", "appetite"]}
    v0.update(kw)
    return v0


def _rows(ctx, v, S=None):
    S = S or Sampler(1)
    return {nm: result_row(evaluate_single(nm, dd, ctx, S, v, {}), ctx, {}) for nm, dd in DRUG_DATABASE.items()}


def run_self_checks():
    """Verification + face-validity checks. Returns a list of dicts (Check, Result, Detail)."""
    out = []

    def add(name, ok, detail=""):
        out.append({"Check": name, "Result": "PASS" if ok else "FAIL", "Detail": detail})

    base_v = _test_v(delusions=0.5, agitation=0.7, depression=0.4, anxiety=0.4, sleep=0.3, apathy=0.3, irritability=0.3)
    base_ctx = _test_ctx()

    # V1: vulnerability endpoints
    ok = True
    for name, (x0, k, lo, hi) in VULN_PARAMS.items():
        a = rescaled_logistic(np.array([lo]), x0, k, lo, hi)[0]
        b = rescaled_logistic(np.array([hi]), x0, k, lo, hi)[0]
        ok &= abs(a) < 1e-12 and abs(b - 1.0) < 1e-12
    add("Vulnerability curves reach exactly 0 and 1 at their range ends", ok, f"{len(VULN_PARAMS)} curves")

    # V1: occupancy / hazards nondecreasing in dose
    S = Sampler(1)
    bad = []
    for nm, dd in DRUG_DATABASE.items():
        prev = None
        for delta in (0.25, 0.5, 1.0, 2.0):
            ph = drug_pharmacology(nm, dd, base_ctx, S, delta=delta)
            h, _, th, _ = compute_hazards([ph], S)
            cur = np.array([float(th[r][0]) for r in RECEPTORS_ALL] + [float(h[d][0]) for d in DOMAINS])
            if prev is not None and np.any(cur < prev - 1e-12):
                bad.append(nm)
            prev = cur
    add("Occupancy and all hazards are nondecreasing with dose", not bad, "violations: " + (", ".join(sorted(set(bad))) or "none"))

    # V1: competitive binding
    p1 = drug_pharmacology("Risperidone", DRUG_DATABASE["Risperidone"], base_ctx, S)
    p2 = drug_pharmacology("Olanzapine", DRUG_DATABASE["Olanzapine"], base_ctx, S)
    _, _, tt, thi = compute_hazards([p1, p2], S)
    ok = all(abs(thi(p1, r)[0] + thi(p2, r)[0] - tt[r][0]) < 1e-12 and tt[r][0] < 1.0 for r in RECEPTORS_ALL)
    add("Competitive binding: per-drug occupancies sum to total (<1)", bool(ok), f"D2 total = {tt['D2'][0]:.3f}")

    # V3/V4: anchors reproduced
    c90 = _test_ctx(egfr=90.0)
    bad = []
    for nm, anc in REFERENCE_ANCHORS.items():
        if anc[0] == "occ":
            ph = drug_pharmacology(nm, DRUG_DATABASE[nm], c90, S)
            x = ph["x"][anc[1]][0]
            if abs(x / (1.0 + x) - anc[2]) > 1e-6:
                bad.append(nm)
    add("Occupancy anchors are reproduced at reference exposure", not bad, "mismatch: " + (", ".join(bad) or "none"))

    # V2: hard constraints
    rows = _rows(_test_ctx(dementia_subtype=LEWY_SUBTYPES[0]), base_v)
    ok = all((rows[n]["Hard Locked"] and "Full D2" in rows[n]["Lock Reason"]) == bool(DRUG_DATABASE[n].get("d2_full_antagonist")) for n in rows)
    add("DLB/PDD: exactly the full D2 antagonists are locked by the D2 rule", ok,
        "locked: " + ", ".join(n for n in rows if "Full D2" in rows[n]["Lock Reason"]))
    rows = _rows(_test_ctx(qtc_ms=495.0), base_v)
    hi_qt = [n for n, d in DRUG_DATABASE.items() if d["Risk_QTc"] >= 0.4]
    zero_qt = [n for n, d in DRUG_DATABASE.items() if d["Risk_QTc"] <= 0.10]
    ok = all("QTc" in rows[n]["Lock Reason"] for n in hi_qt) and not any("QTc" in rows[n]["Lock Reason"] for n in zero_qt)
    add("QTc 495 ms: high-QT-risk agents locked, low-QT-risk (≤0.10) agents not", ok, f"{len(hi_qt)} high-risk / {len(zero_qt)} zero-risk checked")
    rows = _rows(_test_ctx(seizure_freq_year=2.0), base_v)
    add("Seizure history: bupropion is locked", "prohibited" in rows["Bupropion"]["Lock Reason"])
    rows = _rows(_test_ctx(has_maoi=True), base_v)
    add("Active MAOI: all serotonergic agents are locked", all("MAOI" in rows[n]["Lock Reason"] for n in SEROTONERGIC_DRUGS))

    # Hyponatremia
    rows_low, rows_ok = _rows(_test_ctx(sodium=128.0), base_v), _rows(_test_ctx(sodium=140.0), base_v)
    gated = [n for n, d in DRUG_DATABASE.items() if d["pKi"].get("SERT", 0) >= SERT_ACTIVE_PKI or d.get("hypoNa_index", 0) >= 0.30]
    ok = all("serum Na" in rows_low[n]["Lock Reason"] for n in gated) and not any("serum Na" in rows_ok[n]["Lock Reason"] for n in gated)
    add("Na 128 locks SERT-active/SIADH agents; Na 140 does not", ok, ", ".join(gated))
    hv = _test_v(depression=0.7)
    r_hi = _rows(_test_ctx(sodium=132.0, thiazide=True, age=85.0), hv)["Sertraline"]["Hyponatremia risk"]
    r_lo = _rows(_test_ctx(sodium=142.0, thiazide=False, age=70.0), hv)["Sertraline"]["Hyponatremia risk"]
    add("Hyponatremia index is higher for low Na + thiazide + older age", r_hi > r_lo, f"{r_hi} vs {r_lo}")

    # Exposure
    rows = _rows(_test_ctx(egfr=15.0), base_v)
    rows_auto = _rows(_test_ctx(egfr=15.0, dose_policy="auto"), base_v)
    g = rows["Gabapentin"]
    ok = g["Exposure Factor (E)"] > E_MAX_STANDARD and "Standard dose infeasible" in g["Lock Reason"] \
        and "Standard dose infeasible" not in rows_auto["Gabapentin"]["Lock Reason"]
    add("eGFR 15: gabapentin accumulates (>3×) and is locked at standard dose, unlocked by auto-reduce", ok, f"E = {g['Exposure Factor (E)']}")
    rows = _rows(_test_ctx(active_aeds=["carbamazepine"]), base_v)
    add("Carbamazepine induction lowers quetiapine exposure (E < 0.7)", rows["Quetiapine"]["Exposure Factor (E)"] < 0.7,
        f"E = {rows['Quetiapine']['Exposure Factor (E)']}")

    # Face validity of mechanism
    def top_of(v, **kw):
        rws = _rows(_test_ctx(**kw), v)
        ok_rows = [r for r in rws.values() if not r["Hard Locked"]]
        return max(ok_rows, key=lambda r: r["Raw_Mj"]) if ok_rows else None
    t = top_of(_test_v(depression=0.8))
    add("Depression-only profile: top agent engages SERT or NET (θ ≥ 0.30)", bool(t) and max(t["Occupancy"]["SERT"], t["Occupancy"]["NET"]) >= 0.30,
        t["Drug"] if t else "none")
    t = top_of(_test_v(delusions=0.8, hallucinations=0.8))
    add("Psychosis-only profile: top agent is an antipsychotic", bool(t) and t["Category"] in ANTIPSYCHOTIC_CATEGORIES, t["Drug"] if t else "none")
    t = top_of(_test_v(anxiety=0.8))
    add("Anxiety-only profile: top agent engages SERT or GABA-A (θ ≥ 0.30)", bool(t) and max(t["Occupancy"]["SERT"], t["Occupancy"]["GABA-A"]) >= 0.30,
        t["Drug"] if t else "none")

    # Preferences
    ms = [_rows(_test_ctx(pref_sedation=p), base_v)["Mirtazapine"]["Raw_Mj"] for p in (0.5, 1.0, 1.5)]
    add("Higher caregiver sedation priority never raises mirtazapine's score", ms[0] >= ms[1] >= ms[2], f"{ms[0]:.3f} ≥ {ms[1]:.3f} ≥ {ms[2]:.3f}")

    # Monte Carlo probabilities
    mc_t = run_monte_carlo(base_ctx, base_v, {}, 300, seed=11)
    tot = sum(m["P_first"] for m in mc_t.values())
    add("Monte Carlo rank-first probabilities sum to 1", abs(tot - 1.0) < 1e-9, f"sum = {tot:.3f}")
    # V1/V3: database integrity and evidence consistency
    bad = [n for n, d in DRUG_DATABASE.items()
           if set(d["pKi"]) != set(RECEPTORS_ALL) or abs(d["Fr_renal"] + d["Fr_hepatic"] - 1.0) > 1e-9
           or not (0.0 <= d["Risk_QTc"] <= 1.0) or not (0.0 <= d["convulsant_index"] < 1.0) or not (0.0 <= d["hypoNa_index"] < 1.0)
           or any(not (v == 0.0 or 4.0 <= v <= 11.0) for v in d["pKi"].values())]
    add("Database integrity: 11 receptors per drug, Fr sums to 1, indices in range", not bad, "problems: " + (", ".join(bad) or "none"))
    aud = pki_audit_df()
    expected = aud["Evidence pKi (geo-mean)"].clip(lower=CENSOR_PKI)   # evidence below the floor is stored as the floor (5.0)
    bad_c = aud[(aud["Status"] == "C") & ((aud["DB pKi"] - expected).abs() > 0.06) & (aud["Receptor"] != "SERT")]
    bad_v = aud[(aud["Status"] == "V") & (aud["Δ (DB − evidence)"].abs() > 0.5 + 1e-9) & (aud["Receptor"] != "SERT")]
    bad_s = aud[(aud["Receptor"] == "SERT") & (aud["Δ (DB − evidence)"].abs() > 0.06)]
    add("Evidence consistency: corrected (C) and SERT values equal the evidence mean; verified (V) within 0.5",
        len(bad_c) + len(bad_v) + len(bad_s) == 0, f"{len(aud)} evidence entries checked")
    qb = qt_band_df()
    qb = qb[qb["In band"] != "n/a"]
    add("Risk_QTc lies inside its CredibleMeds-category band for every categorised drug", bool((qb["In band"] == "yes").all()),
        f"{len(qb)} drugs checked")
    pk_all = pk_calibration_df()
    pk = pk_all[pk_all["Physical (Fr ≤ 1)"]]
    add("Label AUC ratios imply Fr_ren within ±0.2 of the database value (physical points only)", bool((pk["Δ (DB − implied)"].abs() <= 0.2).all()),
        f"{len(pk)} of {len(pk_all)} points used (implied Fr > 1 excluded); max |Δ| = {pk['Δ (DB − implied)'].abs().max():.2f}")
    dlb = [n for n, d in DRUG_DATABASE.items() if d["pKi"].get("SERT", 0) >= SERT_ACTIVE_PKI]
    add("SERT-active set is the expected SSRIs/SNRIs/TCA/vortioxetine", set(dlb) == set(SEROTONERGIC_DRUGS), ", ".join(sorted(dlb)))
    return out

# =============================================================================
# 6. USER INTERFACE
# =============================================================================
# -----------------------------------------------------------------------------
# 6.1 SETTINGS & WARNINGS
# -----------------------------------------------------------------------------
st.warning(
    "⚠️ **Framework v3 (occupancy-based, audited database).** Reference anchors, severity weights, risk liabilities and vulnerability curves are "
    "partly calibrated: see the Calibration, Audit & Improvement Center. This prototype is not validated for clinical use."
)
with st.expander("⚙️ Model & uncertainty settings (framework v3)", expanded=False):
    col_set1, col_set2, col_set3 = st.columns(3)
    with col_set1:
        dose_policy_label = st.radio(
            "Dose policy for organ function / DDI",
            ["Standard geriatric dose (δ = 1)", "Auto-reduce dose to clearance (δ = min(1, φ))"],
            index=0
        )
    with col_set2:
        background_label = st.radio(
            "Candidate role",
            ["Add-on to current regimen (conservative)", "Replacement of current psychotropics"],
            index=0
        )
    with col_set3:
        mc_draws = st.select_slider("Monte Carlo draws (0 = off)", options=[0, 200, 500, 1000, 2000], value=500)
        show_anchors = st.checkbox("Show reference anchors", value=False)
dose_policy = "auto" if dose_policy_label.startswith("Auto") else "standard"
background_mode = "addon" if background_label.startswith("Add-on") else "replace"

# -----------------------------------------------------------------------------
# 6.2-6.4 PATIENT PARAMETERS, MEDICATION AUDIT, SYMPTOMS & PREFERENCES
# -----------------------------------------------------------------------------
st.subheader("📋 Patient Clinical Parameters")

c_etiology, c_bio1, c_bio2 = st.columns(3)

with c_etiology:
    dementia_subtype = st.selectbox(
        "Dementia Subtype / Etiology",
        ["Alzheimer's Disease (AD)", "Dementia with Lewy Bodies (DLB)", "Parkinson's Disease Dementia (PDD)", "Vascular Dementia (VaD)", "Frontotemporal Dementia (FTD)"]
    )
    mmse_score = st.number_input("MMSE / MoCA Score (0-30)", 0, 30, 14)
    age_val = st.number_input("Patient Age (years)", 18, 110, 78)

with c_bio1:
    morse_score = st.number_input("Morse Fall Scale (0-125)", 0, 125, 40)
    sbp_drop = st.number_input("Standing SBP Drop (mmHg)", 0, 60, 12)
    sas_score = st.number_input("SAS Motor / EPS Score (0-40)", 0, 40, 4)
    
    frailty_str = st.selectbox("Frailty Index Tier", list(FRAILTY_MAPPING.keys()), index=1)
    frailty_val = FRAILTY_MAPPING[frailty_str]

with c_bio2:
    qtc_ms = st.number_input("Baseline QTc Interval (ms)", 300, 600, 430)
    egfr_val = st.number_input("eGFR (ml/min)", 5, 120, 55)
    hepatic_status = st.selectbox("Hepatic Function Status", list(HEPATIC_MAPPING.keys()), index=0)
    lft_val = HEPATIC_MAPPING[hepatic_status]

# Special Conditions Input Sub-Section (Integer Seizure Frequency Rounding)
with st.expander("🩺 Special Conditions Risk Inputs (Epilepsy, NCDs, Comorbidities)", expanded=False):
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        seizure_freq_input = st.number_input("Seizure Frequency (events/year)", 0.0, 50.0, 0.0, step=0.5)
        seizure_freq_val = int(np.round(seizure_freq_input))  # Rounded to integer
        active_aeds_val = st.multiselect(
            "Active Anti-Epileptic Drugs (AEDs)", 
            ["carbamazepine", "valproate", "lamotrigine", "gabapentin", "phenytoin", "levetiracetam"]
        )
    with col_sc2:
        hba1c_val = st.number_input("HbA1c (%)", 4.0, 15.0, 6.5, step=0.1)
        bmi_val = st.number_input("BMI (kg/m²)", 12.0, 50.0, 25.0, step=0.5)
        has_thyroid_val = st.checkbox("Active Thyroid Dysfunction", value=False)
        sodium_val = st.number_input("Baseline serum sodium (mmol/L)", 110.0, 155.0, 140.0, step=1.0)
        thiazide_val = st.checkbox("Thiazide-type diuretic use", value=False)

st.markdown("---")
st.subheader("💊 Step 1: Expanded Prior Medication Audit & Regimen Risk Stratification")

col_prior1, col_prior2 = st.columns(2)

ALL_AUDIT_OPTIONS = list(DRUG_DATABASE.keys()) + MAOI_AGENTS + ANTICHOLINERGIC_AGENTS + ACHEI_AGENTS + ["Phenytoin (Dilantin)"]

with col_prior1:
    prior_drugs = st.multiselect(
        "Select Active / Prior Psychotropic & Neurological Medications",
        options=ALL_AUDIT_OPTIONS
    )

# AED SYNCHRONIZATION ---
combined_prior_drugs = list(set(prior_drugs + [AED_TO_DRUG_NAME[aed] for aed in active_aeds_val if aed in AED_TO_DRUG_NAME]))

prior_history = {}
active_psychotropic_count = 0
has_active_maoi = False
has_active_achei = False
has_active_ssri = False

if combined_prior_drugs:
    with col_prior2:
        for drug in combined_prior_drugs:
            if drug in prior_drugs:
                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    dosage = st.text_input(f"Dose for {drug}", value=" mg/day", key=f"dose_{drug}")
                with c_d2:
                    outcome = st.selectbox(
                        f"Outcome for {drug}",
                        ["Active Regimen - Ongoing", "Partial Response / Tolerated", "Treatment Failure / Ineffective", "Severe Adverse Effects / Intolerant"],
                        key=f"outcome_{drug}"
                    )
            else:
                dosage = DRUG_DATABASE.get(drug, {}).get("dosage", "Standard dose")
                outcome = "Active Regimen - Ongoing"
            
            prior_history[drug] = {"dosage": dosage, "outcome": outcome}

            # Count Active Psychotropic Load & Flags
            if outcome == "Active Regimen - Ongoing":
                if drug in DRUG_DATABASE or drug in MAOI_AGENTS or drug in ANTICHOLINERGIC_AGENTS:
                    active_psychotropic_count += 1
                if drug in MAOI_AGENTS:
                    has_active_maoi = True
                if drug in ACHEI_AGENTS:
                    has_active_achei = True
                if drug in ["Escitalopram", "Sertraline"]:
                    has_active_ssri = True

# Build active_aeds_final for DDI
combined_aeds_set = set([aed.lower() for aed in active_aeds_val])
for pdrug in combined_prior_drugs:
    if pdrug in AED_NAME_MAP:
        combined_aeds_set.add(AED_NAME_MAP[pdrug])
active_aeds_final = list(combined_aeds_set)

# Mandatory Epilepsy Protocol Banner
if seizure_freq_val > 0 or len(active_aeds_final) > 0:
    st.warning(
        f"⚠️ **EPILEPSY PROTOCOL ACTIVE**: Baseline AED dosing is **strictly mandatory** (Current integer seizure frequency: **{seizure_freq_val} events/year**). "
        f"Do NOT reduce or discontinue AED dosages during psychotropic cross-titration or regimen optimization."
    )

# Stratify Branch A vs Branch B based on active drug burden
polypharmacy_branch = active_psychotropic_count >= 2

if polypharmacy_branch:
    st.warning(
        f"⚡ **Branch B Stratification Triggered (Polypharmacy Burden = {active_psychotropic_count} Active Agents):** "
        f"Patient is taking $\\ge 2$ active psychotropics. Escalating directly to Rational Combination Pharmacotherapy & Regimen Optimization."
    )
else:
    st.info(
        f"ℹ️ **Branch A Stratification (Low Drug Burden = {active_psychotropic_count} Active Agents):** "
        f"Standard Sequential Monotherapy Pathway active."
    )

# Assemble Patient Profile Dictionary
patient_profile = {
    "age": float(age_val),
    "frailty_score": frailty_val,
    "seizure_freq_year": float(seizure_freq_val),
    "active_aeds": active_aeds_final,
    "hba1c": hba1c_val,
    "bmi": bmi_val,
    "sbp_drop": float(sbp_drop),
    "has_thyroid_dysfunction": has_thyroid_val,
    "has_maoi": has_active_maoi,
    "has_achei": has_active_achei,
    "has_ssri": has_active_ssri,
    "polypharmacy_branch": polypharmacy_branch
}

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

# Receptor Weight Mapping
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


# Symptom severity vector (inputs to the noisy-OR need weights)
v = {
    "delusions": v_delusions, "hallucinations": v_hallucinations, "agitation": v_agitation,
    "depression": v_depression, "anxiety": v_anxiety, "euphoria": v_euphoria, "apathy": v_apathy,
    "disinhibition": v_disinhibition, "irritability": v_irritability, "motor": v_motor,
    "sleep": v_sleep, "appetite": v_appetite,
}

# Extend the patient profile with everything the v2 engine needs
patient_profile.update({
    "mmse": float(mmse_score),
    "qtc_ms": float(qtc_ms),
    "egfr": float(egfr_val),
    "lft_impairment": float(lft_val),
    "morse": float(morse_score),
    "sas": float(sas_score),
    "sodium": float(sodium_val),
    "thiazide": bool(thiazide_val),
    "dementia_subtype": dementia_subtype,
    "pref_sedation": pref_sedation,
    "pref_falls": pref_falls,
    "dose_policy": dose_policy,
    "background_mode": background_mode,
    "active_db_drugs": [d for d, h in prior_history.items() if h["outcome"] == "Active Regimen - Ongoing" and d in DRUG_DATABASE],
    "active_nondb": [d for d, h in prior_history.items() if h["outcome"] == "Active Regimen - Ongoing" and d in NONDB_BACKGROUND_HAZARD],
})

# -----------------------------------------------------------------------------
# 6.5 DECISION ENGINE & RECOMMENDATION
# -----------------------------------------------------------------------------
symptoms_present = any(val > 0.0 for val in v.values())

S0 = Sampler(1)   # nominal (median) inputs
results = [
    result_row(evaluate_single(drug, drug_data, patient_profile, S0, v, prior_history), patient_profile, prior_history)
    for drug, drug_data in DRUG_DATABASE.items()
]
results = sorted(results, key=lambda x: x["Raw_Mj"], reverse=True)
top_drug = results[0]

# Uncertainty propagation (Monte Carlo over Ki, anchors, kappa, beta, zeta, vulnerability curves)
mc = None
if mc_draws > 0 and symptoms_present:
    mc = run_monte_carlo(patient_profile, v, prior_history, int(mc_draws))
for r in results:
    if mc is not None and not r["Hard Locked"]:
        m = mc[r["Drug"]]
        r["P(rank 1)"] = f"{100 * m['P_first']:.0f}%"
        r["Mj 5–95%"] = "—" if np.isnan(m["p05"]) else f"{100 * m['p05']:.0f} … {100 * m['p95']:.0f}"
    else:
        r["P(rank 1)"], r["Mj 5–95%"] = "—", "—"

# Multi-Agent Combination Regimen Engine
comb_analysis = evaluate_combination_regimens(results, patient_profile, S0, v, prior_history)

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
    if not symptoms_present:
        st.info("ℹ️ No target symptoms are rated (all NPI items = 0), so no therapeutic need can be matched. Rate at least one symptom.")
    else:
        st.error("🚨 No suitable candidate found. All agents were excluded by Stage A hard constraints "
                 "(contraindications, QTc, exposure limits, no modelled therapeutic engagement) or manual rule-outs.")
        if any(r["Lock Reason"].startswith("Standard dose infeasible") for r in results):
            st.info("💡 Several agents are excluded only because organ impairment makes the **standard dose** unsafe. "
                    "Switch the dose policy in *Model & uncertainty settings* to **Auto-reduce dose** to rank them at a clearance-adjusted dose.")
else:
    if comb_analysis["is_viable"]:
        st.warning(
            f"⚡ **Multi-Agent Combination Regimen Suggested:** Monotherapy coverage is constrained or Polypharmacy Branch B is active. "
            f"Combining **{' + '.join(comb_analysis['regimen'])}** (competitive-binding occupancy model) scores higher "
            f"(Combined Match Index: **{comb_analysis['score']}**, +{comb_analysis['margin']} index points over the best single agent)."
        )

    mc_line = ""
    if mc is not None:
        mt = mc[top_drug["Drug"]]
        interval = "—" if np.isnan(mt["p05"]) else f"{100 * mt['p05']:.0f} … {100 * mt['p95']:.0f}"
        mc_line = (f"<br/>P(rank 1) = <strong>{100 * mt['P_first']:.0f}%</strong> &nbsp;|&nbsp; "
                   f"Match Index 90% interval: <strong>{interval}</strong> &nbsp;|&nbsp; Monte Carlo draws: {int(mc_draws)}")

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
                Match Index (100·Mj): <strong>{top_drug['Net Score (Mj)']}</strong> &nbsp;|&nbsp;
                Therapeutic utility: <strong>+{top_drug['Therapeutic Gain']}</strong> &nbsp;|&nbsp;
                Safety deductions: <strong>-{top_drug['Risk Deductions']}</strong> &nbsp;|&nbsp;
                History/interaction burden: <strong>-{top_drug['History Burden']}</strong> &nbsp;|&nbsp;
                Exposure at standard dose: <strong>×{top_drug['Exposure Factor (E)']}</strong>
                {mc_line}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Statistical-indistinguishability guard
    if mc is not None:
        ranked_mc = sorted(((nm, m["P_first"]) for nm, m in mc.items()), key=lambda t: t[1], reverse=True)
        if len(ranked_mc) > 1 and ranked_mc[0][1] - ranked_mc[1][1] < 0.10:
            st.warning(
                f"⚖️ **Ranking indeterminate:** the top candidates are statistically indistinguishable under input uncertainty "
                f"({ranked_mc[0][0]}: {100 * ranked_mc[0][1]:.0f}% vs {ranked_mc[1][0]}: {100 * ranked_mc[1][1]:.0f}% probability of ranking first). "
                f"Treat the leading options as a short-list, not a single answer."
            )
        elif ranked_mc[0][0] != top_drug["Drug"]:
            st.info(f"ℹ️ The nominal top candidate differs from the most frequent Monte Carlo winner ({ranked_mc[0][0]}, "
                    f"{100 * ranked_mc[0][1]:.0f}%). Inspect the uncertainty panel below.")

    dose_scaling_msg = ""
    if top_drug["Suggested Dose Scaling"] < 0.90:
        dose_scaling_msg = (f"<br/><strong>Organ/DDI dose scaling:</strong> clearance is reduced (φ = {top_drug['Suggested Dose Scaling']}); "
                            f"consider ≈ ×{top_drug['Suggested Dose Scaling']} of the standard dose to hold exposure at the reference level.")
    st.markdown(
        f"""
        <div style="background-color: #e2f0d9; border-left: 6px solid #385723; color: #274411; padding: 12px 18px; border-radius: 6px; margin-bottom: 15px; font-size: 15px;">
            <strong>💊 Recommended Dosage Spectrum:</strong> {top_drug['Dosage']}{dose_scaling_msg}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Organ Monitoring Callout Alerts
    if top_drug['Drug'] == "Gabapentin":
        st.info(f"🩺 **Organ Monitoring Alert (Renal)**: 100% Renal Elimination ($Fr_\\text{{renal}} = 1.00$). Current eGFR = {egfr_val} mL/min; exposure factor E = {top_drug['Exposure Factor (E)']}. Strict renal dose titration required.")
    elif top_drug['Drug'] == "Valproic Acid":
        st.info(f"🩺 **Organ Monitoring Alert (Hepatic)**: High Hepatic Metabolism ($Fr_\\text{{hepatic}} = 0.95$); exposure factor E = {top_drug['Exposure Factor (E)']}. Baseline LFTs, CBC, and plasma level monitoring (target 50–80 µg/mL) required.")
    elif top_drug['Drug'] == "Memantine" and egfr_val < 30:
        st.info(f"🩺 **Organ Monitoring Alert (Renal)**: High Renal Elimination ($Fr_\\text{{renal}} = 0.80$). Current eGFR = {egfr_val} mL/min (< 30 mL/min). Maximum dose cap: 10 mg daily.")

    if (not top_drug["Hard Locked"]) and top_drug["Occupancy"].get("SERT", 0.0) >= 0.30:
        st.info(f"🧂 **Hyponatremia Monitoring Alert**: SERT occupancy ≈ {100 * top_drug['Occupancy']['SERT']:.0f}%. "
                f"Baseline Na = {sodium_val:.0f} mmol/L{' with thiazide use' if thiazide_val else ''}; hyponatremia index = {top_drug['Hyponatremia risk']}. "
                f"Recheck sodium at ~2 and ~4 weeks (and with any new confusion, falls or nausea).")

    if top_drug['Clinical Correlation Note']:
        st.info(f"💡 **Clinical Correlation Note:** {top_drug['Clinical Correlation Note']}")

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
        f"🚫 Rule Out **{top_drug['Drug']}** (Clinical contraindication / Tailored preference)",
        key=f"chk_ruleout_{top_drug['Drug']}"
    )
    if rule_out_flag:
        st.session_state.ruled_out.add(top_drug["Drug"])
        st.rerun()

# -----------------------------------------------------------------------------
# 6.6-6.7 DASHBOARDS & PHENOTYPE CLUSTER
# -----------------------------------------------------------------------------
with st.expander("🔄 Sequential Cross-Titration & Dose Optimization Protocol", expanded=True):
    if combined_prior_drugs:
        distinct_priors = [d for d in combined_prior_drugs if d != top_drug["Drug"]]

        if top_drug["Drug"] in combined_prior_drugs:
            st.info(
                f"💡 **Active Regimen Optimization Protocol:** **{top_drug['Drug']}** is already part of the patient's active regimen. "
                f"Rather than cross-tapering, evaluate optimizing current dosage toward targeted spectrum (**{top_drug['Dosage']}**)."
            )

        if distinct_priors:
            prior_selected = st.selectbox(
                "Select distinct active/prior agent to cross-taper from:",
                options=distinct_priors,
                key="cross_taper_selector"
            )
            # Epilepsy Safety Lock Check
            if (seizure_freq_val > 0 or len(active_aeds_final) > 0) and prior_selected in AED_NAME_MAP:
                st.error(
                    f"⛔ **CRITICAL EPILEPSY SAFETY LOCK**: **{prior_selected}** is an essential AED in a patient with active seizure history/epilepsy. "
                    f"Baseline AED dosing is mandatory; do NOT taper or discontinue baseline AEDs during psychotropic cross-titration."
                )
            else:
                st.markdown(f"### Cross-Titration: Taper **{prior_selected}** $\\rightarrow$ Initiate **{top_drug['Drug']}**")
                tt_df = generate_cross_titration_schedule(prior_selected, top_drug["Drug"])
                if tt_df is not None:
                    st.table(tt_df)
        elif top_drug["Drug"] not in combined_prior_drugs:
            st.info("Treatment Naive: Initiate top candidate at starting dose without cross-tapering.")
    else:
        st.info("Treatment Naive: Initiate top candidate at starting dose without cross-tapering.")

with st.expander("📊 Patient-Weighted Marginal Risk Indices (framework v3)", expanded=True):
    st.markdown("#### Marginal risk ΔR = λ · (1 − h_bg) · h  (0–100, uncalibrated index — not a probability)")
    col_vis1, col_vis2, col_vis3, col_vis4 = st.columns(4)

    with col_vis1:
        st.write("**Sedation / Falls Index:**")
        st.progress(int(min(100, max(0, top_drug["Sedation risk"]))))
        st.caption(f"Index: {top_drug['Sedation risk']}")

    with col_vis2:
        st.write("**Orthostatic Index:**")
        st.progress(int(min(100, max(0, top_drug["Orthostatic risk"]))))
        st.caption(f"Index: {top_drug['Orthostatic risk']}")

    with col_vis3:
        st.write("**EPS (D2 window) Index:**")
        st.progress(int(min(100, max(0, top_drug["EPS risk"]))))
        st.caption(f"Index: {top_drug['EPS risk']}")

    with col_vis4:
        st.write("**Hyponatremia Index:**")
        st.progress(int(min(100, max(0, top_drug["Hyponatremia risk"]))))
        st.caption(f"Index: {top_drug['Hyponatremia risk']}")

with st.expander("🧬 Receptor Occupancy & Domain Risk Breakdown (top-ranked candidate)", expanded=False):
    if top_drug["Hard Locked"]:
        st.info("No eligible top candidate to break down.")
    else:
        occ_rows = [{
            "Receptor": rcp,
            "Occupancy θ": round(top_drug["Occupancy"][rcp], 3),
            "Utility contribution (100·w·s·Δ·θ / Σw)": round(top_drug["Receptor Contrib"][rcp], 1) if rcp in THERAPEUTIC_RECEPTORS else None,
        } for rcp in RECEPTORS_ALL]
        st.dataframe(pd.DataFrame(occ_rows), use_container_width=True)
        st.markdown("**Weighted safety deductions by domain (100·ζ·ΔR):**")
        st.dataframe(pd.DataFrame([top_drug["Domain Breakdown"]]), use_container_width=True)
        st.caption(f"Projected ΔQTc ≈ {top_drug['Projected dQTc (ms)']} ms; exposure factor E = {top_drug['Exposure Factor (E)']}.")

if mc is not None:
    with st.expander("🎲 Uncertainty & Sensitivity (Monte Carlo)", expanded=False):
        mc_rows = sorted(
            [{"Drug": nm, "P(rank 1)": round(m["P_first"], 3), "P(feasible)": round(m["P_feasible"], 3),
              "Mj p05": None if np.isnan(m["p05"]) else round(100 * m["p05"], 1),
              "Mj median": None if np.isnan(m["p50"]) else round(100 * m["p50"], 1),
              "Mj p95": None if np.isnan(m["p95"]) else round(100 * m["p95"], 1)} for nm, m in mc.items()],
            key=lambda r: r["P(rank 1)"], reverse=True
        )
        st.dataframe(pd.DataFrame(mc_rows).head(10), use_container_width=True)
        if not top_drug["Hard Locked"]:
            shares = variance_shares(top_drug["Drug"], patient_profile, v, prior_history, int(mc_draws))
            st.markdown(f"**Which uncertainties drive the score of {top_drug['Drug']}?** (one-at-a-time variance share)")
            st.dataframe(pd.DataFrame(
                [{"Uncertainty group": Sampler.GROUP_LABELS[g], "Variance share": f"{100 * val:.0f}%"} for g, val in
                 sorted(shares.items(), key=lambda kv: kv[1], reverse=True)]
            ), use_container_width=True)
            st.caption("Large shares identify the inputs that most need better evidence (calibration targets).")

if show_anchors:
    with st.expander("⚓ Reference Anchors (uncalibrated placeholders)", expanded=True):
        anchor_rows = [
            {"Drug": nm,
             "Anchor type": "Occupancy" if a[0] == "occ" else "Unbound conc. (nM)",
             "Receptor": a[1] if a[0] == "occ" else "—",
             "Value": a[2] if a[0] == "occ" else a[1]}
            for nm, a in REFERENCE_ANCHORS.items()
        ]
        st.dataframe(pd.DataFrame(anchor_rows), use_container_width=True)
        st.caption("Replace with PET occupancy or measured unbound concentrations at the standard geriatric dose.")

with st.expander("🚦 Dashboard & Ranking", expanded=False):
    df_results = pd.DataFrame(results)

    df_results_display = df_results[[
        "Drug", "Category", "Net Score (Mj)", "P(rank 1)", "Mj 5–95%", "Therapeutic Gain",
        "Risk Deductions", "History Burden", "Exposure Factor (E)", "Suggested Dose Scaling",
        "Dosage", "Lock Reason"
    ]].copy()

    def apply_traffic_lights(val):
        try:
            val_float = float(val)
            if val_float > 10.0:
                return 'background-color: #d4edda; color: #155724; font-weight: bold;'
            elif val_float >= -10.0:
                return 'background-color: #fff3cd; color: #856404;'
            else:
                return 'background-color: #f8d7da; color: #721c24;'
        except (ValueError, TypeError):
            return 'background-color: #f8d7da; color: #721c24; font-weight: bold;'

    st.dataframe(
        df_results_display.style.map(apply_traffic_lights, subset=['Net Score (Mj)']),
        use_container_width=True
    )

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

# -----------------------------------------------------------------------------
# 6.8 CALIBRATION, AUDIT & IMPROVEMENT CENTER
# -----------------------------------------------------------------------------
with st.expander("🧪 Calibration, Audit & Improvement Center", expanded=False):
    reg_df = pd.DataFrame(PLACEHOLDER_REGISTER, columns=REGISTER_COLUMNS)
    st.markdown("### 1. Calibration status at a glance")
    st.dataframe(reg_df["Status"].value_counts().rename_axis("Status").reset_index(name="Parameter groups"), use_container_width=True)
    st.markdown("### 2. Parameter register: status, what was applied, and how to calibrate / validate")
    st.dataframe(reg_df, use_container_width=True)
    st.download_button("⬇️ Download register (CSV)", reg_df.to_csv(index=False).encode("utf-8"),
                       file_name="parameter_register.csv", mime="text/csv")
    st.markdown("### 3. Improvement roadmap")
    st.dataframe(pd.DataFrame(IMPROVEMENT_PLAN, columns=IMPROVEMENT_COLUMNS), use_container_width=True)
    st.markdown("### 4. Validation roadmap")
    st.dataframe(pd.DataFrame(VALIDATION_PLAN, columns=VALIDATION_COLUMNS), use_container_width=True)
    st.markdown("### 5. Database audit status")
    st.caption("  ·  ".join(f"**{k}** = {v}" for k, v in STATUS_LEGEND.items()))
    st.dataframe(db_status_df(), use_container_width=True)
    st.markdown("### 6. Affinity evidence audit (database vs literature)")
    st.dataframe(pki_audit_df(), use_container_width=True)
    st.markdown("### 7. Renal PK calibration against label AUC ratios")
    st.dataframe(pk_calibration_df(), use_container_width=True)
    st.markdown("### 8. QTc categories and seizure indices")
    st.dataframe(qt_band_df(), use_container_width=True)
    st.dataframe(seizure_df(), use_container_width=True)
    st.markdown("### 9. Expert-elicitation templates (Delphi)")
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.download_button("⬇️ κ matrix", delphi_template_df("kappa").to_csv(index=False).encode("utf-8"), file_name="delphi_kappa.csv", mime="text/csv")
    with col_t2:
        st.download_button("⬇️ ζ / β weights", delphi_template_df("zeta_beta").to_csv(index=False).encode("utf-8"), file_name="delphi_zeta_beta.csv", mime="text/csv")
    with col_t3:
        st.download_button("⬇️ Vulnerability curves", delphi_template_df("vuln").to_csv(index=False).encode("utf-8"), file_name="delphi_vulnerability.csv", mime="text/csv")
    st.markdown("### 10. Model self-checks (verification, face validity, database integrity)")
    if st.button("Run self-checks", key="btn_selfchecks"):
        chk = pd.DataFrame(run_self_checks())
        st.dataframe(chk, use_container_width=True)
        n_fail = int((chk["Result"] == "FAIL").sum())
        (st.success if n_fail == 0 else st.error)(f"{len(chk) - n_fail}/{len(chk)} checks passed.")

# -----------------------------------------------------------------------------
# 6.9 RATIONALE (FRAMEWORK V3)
# -----------------------------------------------------------------------------
with st.expander("🧮 Algorithmic Architecture & Clinical Rationale", expanded=False):
    st.markdown(r"""
    This tool uses an **occupancy-based, two-stage multi-criteria model**: hard constraints first (Stage A), then a bounded utility–risk score (Stage B). Every term is dimensionless and bounded.

    ---

    **1. Patient-adjusted exposure**

    $$\varphi_j = 1 - Fr_{\text{ren}}(1-g_{\text{ren}}) - Fr_{\text{hep}}(1-g_{\text{hep}} f_{\text{ind}}),\qquad E_j = \frac{1}{\varphi_j},\qquad C_{rj} = \delta_j E_j C^{\text{ref}}_{rj}$$

    $g_{\text{ren}} = \mathrm{eGFR}/90$, $g_{\text{hep}} = 1-$ hepatic impairment, $f_{\text{ind}}$ = CYP-induction factor (carbamazepine/phenytoin).

    **2. Receptor occupancy (law of mass action)**

    $$\theta_{rj} = \frac{C_{rj}}{C_{rj}+K_{i,rj}},\qquad \mathrm{p}K_i = 9 - \log_{10}K_i[\text{nM}]$$

    Reference concentrations are fixed by anchors (PET-style occupancy or unbound concentration). Two ligands at one receptor bind competitively: $\theta_{ir} = x_{ir}/(1+\sum_k x_{kr})$ with $x = C/K_i$. Floor values $\mathrm{p}K_i\le 5$ are censored (no occupancy).

    **3. Neurological-treatment coupling (noisy-OR)**

    $$w_r = 1-\prod_{s}\left(1 - v_s\,|\kappa_{sr}|\right),\qquad s_r = \operatorname{sign}\Big(\sum_s v_s\,\kappa_{sr}\Big)$$

    **4. Therapeutic utility**

    $$U_j = \frac{\sum_r w_r\, s_r\, \Delta_{rj}\, \theta_{rj}}{\sum_r w_r}\in[-1,1],\qquad \Delta_{rj} = \alpha_{rj}-t_r$$

    ($\alpha$ = intrinsic activity, $t_r$ = endogenous tone; partial agonists fall between antagonists and full agonists.)

    **5. Hazards, vulnerability and marginal risk**

    $$h_{dj} = 1-\prod_{r\in R_d}(1-\beta_{rd}\theta_{rj}),\qquad h_{\text{EPS}} = \sigma\!\left(\frac{\theta_{D_2}^{\text{net}}-0.80}{0.05}\right),\qquad h_{\text{QTc}} = 1-(1-R_{\text{QTc}})^{E_j}$$

    $$\lambda_d(x)=\frac{\sigma(k(x-x_0))-\sigma(k(x_{\min}-x_0))}{\sigma(k(x_{\max}-x_0))-\sigma(k(x_{\min}-x_0))},\qquad \Delta R_{dj} = \lambda_d\,(1-h^{\text{bg}}_d)\,\big(h_{dj}^{\text{regimen}}-h_{dj}^{\text{current}}\big)$$

    Domains: sedation/falls, orthostasis, EPS, QTc, anticholinergic cognition (M1), seizure threshold, metabolic, and **hyponatremia (SIADH)**, whose hazard is $h_{\\text{Na}} = 1-(1-\\beta\\,\\theta_{\\text{SERT}})\\,(1-I_j)^{E_j}$ ($I_j$ = non-SERT SIADH index, e.g., carbamazepine) with vulnerability from baseline sodium, age and thiazide use. In **add-on mode**, active database agents are evaluated jointly with the candidate (competitive binding) and utility and risk are the *increments* over the current regimen, so shared-receptor occupancy adds up instead of being discounted; non-database agents (e.g., anticholinergics) enter as independent background hazards $h^{\text{bg}}$ in $\Delta R = \lambda\,(1-h^{\text{bg}})\,\Delta h$. In **replacement mode** the candidate is scored alone. Frailty (age > 75) and thyroid dysfunction amplify vulnerability.

    **6. Stage A — hard constraints (non-compensatory)**

    Clinician rule-out, MAOI/serotonergic washout, Bupropion in epilepsy, historical intolerance, full D2 antagonists in DLB/PDD, projected $\mathrm{QTc}_0+\Delta\mathrm{QTc}_j>500$ ms, expected exposure above 3× reference at the standard dose (dose reduction required), baseline Na below 130 mmol/L with a SERT-active/SIADH agent, and no modelled therapeutic engagement ($U_j\le 0.02$).

    **7. Stage B — net match score**

    $$M_j = U_j - \sum_d \zeta_d\,\Delta R_{dj} - \zeta_h\,\rho_j,\qquad -1-Z \le M_j \le 1$$

    $\zeta_d$ are severity weights (caregiver sedation/fall priorities scale $\zeta$); $\rho_j$ is the soft history/interaction burden (prior failure, AChEI opposition, duplicate AED, polypharmacy). The displayed Match Index is $100\,M_j$.

    **8. Uncertainty**

    Monte Carlo propagates uncertainty in $K_i$ ($\sigma = 0.5$ log units), anchors, $\kappa$, $\beta$, $\zeta$, and vulnerability curves, reporting $P(\text{rank }1)$ and 90% intervals. If the top two candidates are within 10 percentage points, the ranking is flagged as indeterminate.

    ---

    **9. Stratified Pathway & Multi-Agent Combination Engine**

    * **Branch A (Low Burden, < 2 Active Psychotropics):** Standard monotherapy trial hierarchy.
    * **Branch B (Polypharmacy, $\ge 2$ Active Psychotropics):** Direct escalation to combination optimization, scored with the same occupancy/hazard rules (competitive binding, additive ΔQTc, noisy-OR hazards).

    **Known limitations:** reference anchors, $\zeta$, $\beta$, and vulnerability curves are uncalibrated placeholders; SERT is modelled (literature Ki; PET occupancy anchors); the database was audited against the literature (see the Calibration Center), but other targets (e.g., sodium channels for lamotrigine/carbamazepine, 5-HT1A) are not, so those agents may show no therapeutic engagement; hERG data are not in the database, so QTc uses the per-drug risk index scaled by exposure.
    """)

# -----------------------------------------------------------------------------
# 6.10 REFERENCES
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
        7. **Cheng, Y., & Prusoff, W. H. (1973).** *Relationship between the inhibition constant (Ki) and the concentration of inhibitor which causes 50 per cent inhibition (I50) of an enzymatic reaction*. Biochem Pharmacol 22:3099–3108.
        8. **Kapur, S., et al. (2000).** *Relationship between dopamine D2 occupancy, clinical response, and side effects: a double-blind PET study of first-episode schizophrenia*. Am J Psychiatry 157:514–520. (Basis of the 65–80% D2 occupancy window; anchors in this tool still require verification.)
        9. **Tatsumi, M., et al. (1997).** *Pharmacological profile of antidepressants and related compounds at human monoamine transporters*. Eur J Pharmacol 340:249–258.
        10. **Owens, M. J., et al. (1997).** *Neurotransmitter receptor and transporter binding profile of antidepressants and their metabolites*. J Pharmacol Exp Ther 283:1305–1322.
        11. **Bang-Andersen, B., et al. (2011).** *Discovery of vortioxetine (Lu AA21004)*. J Med Chem 54:3206–3221.
        12. **Meyer, J. H., et al. (2004).** *Serotonin transporter occupancy of five SSRIs at therapeutic doses*. Am J Psychiatry 161:826–835; and *The relationship between dose and serotonin transporter occupancy of antidepressants: a systematic review*. Mol Psychiatry (2021), doi:10.1038/s41380-021-01285-w.
        13. **Maeda, K., et al. (2014).** *Brexpiprazole I: in vitro and in vivo characterization*. J Pharmacol Exp Ther 350:589–604. Antipsychotic Ki cross-check: Neuropsychopharmacol Rep 2021;41:134–143.
        15. **Ki sources used in the database audit:** Neuropsychopharmacol Rep 2021;41:134–143 (Table 1); Psychopharmacology review of antipsychotic receptor profiles (PMC5112764); IUPHAR/Guide to Pharmacology and vendor-compiled Ki (amitriptyline, mirtazapine); US prescribing information (trazodone, pimavanserin); Song X, et al. Nature 2018;556:515 (memantine/MK-801).
        16. **CredibleMeds QTdrugs lists** (Known / Conditional / Possible risk of TdP), crediblemeds.org; seizure-risk data from US label incidence tables and pharmacoepidemiology (bupropion SIR 4.08; tricyclics SIR 1.58).
        14. **PDSP Ki Database** (NIMH Psychoactive Drug Screening Program), pdsp.unc.edu — to be used for primary verification of every affinity input.
        """
    )

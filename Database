"""
database.py: knowledge base of the BPSD Compass prototype (framework v3.1).

Contents
  1. Mappings (NPI severity options, AED names, medication-audit agent lists)
  2. Knowledge base: audited drug database, model constants, receptor mechanisms, anchors, safety domains
  3. Evidence and calibration registry: Ki evidence, PET/PK/QT/seizure tables, parameter register,
     validation and improvement roadmaps, expert-elicitation (Delphi) tools
No Streamlit code lives here, so the data can be imported and tested on its own.
NOT validated for clinical use.
"""
import copy

import numpy as np
import pandas as pd

# =============================================================================
# 1. MAPPINGS (UI option lists, AED names, agent lists)
# =============================================================================

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
# Risk_QTc (+ CredibleMeds category), convulsant_index, hypoNa_index (non-SERT SIADH hazard), d2_full_antagonist, dosage, warnings.
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
        "hypoNa_index": 0.04,
        "dosage": 'Generally avoid in dementia (5-10 mg PO hs strictly if indicated).',
        "warnings": 'Severe anticholinergic toxicity, fall risk, cognitive decline, and cardiotoxicity.'
    },
    'Venlafaxine': {
        "category": 'Antidepressant (SNRI)',
        "pKi": {'5HT2A': 5.0, 'D2': 5.0, 'NET': 5.79, 'SERT': 7.57, 'α2A': 5.0, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 5.0, 'α1': 5.0, 'M1': 5.0, '5HT2C': 5.0},
        "pKi_status": {'5HT2A': 'F', 'D2': 'F', 'NET': 'C', 'SERT': 'V', 'α2A': 'F', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'F', 'α1': 'F', 'M1': 'F', '5HT2C': 'F'},
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
        "Fr_renal": 0.7, "Fr_hepatic": 0.3, "pk_status": 'U', "pk_note": 'carried over (not re-checked)',
        "Risk_QTc": 0.45, "qt_category": 'CR', "qt_status": 'V',
        "convulsant_index": 0.1, "seizure_status": 'U', "seizure_note": 'carried over (not re-checked)',
        "hypoNa_index": 0.0,
        "dosage": 'Start 12.5 mg - 25 mg hs / PRN; titrate up to 25 mg - 100 mg/day for nighttime agitation.',
        "warnings": 'Orthostatic hypotension, priapism, daytime sedation.'
    },
    'Mirtazapine': {
        "category": 'Pyridine Tetracyclic Antidepressant (NaSSA)',
        "pKi": {'5HT2A': 8.1, 'D2': 5.0, 'NET': 5.0, 'SERT': 5.0, 'α2A': 7.3, 'NMDA': 0.0, 'GABA-A': 0.0, 'H1': 9.3, 'α1': 6.0, 'M1': 5.0, '5HT2C': 7.6},
        "pKi_status": {'5HT2A': 'V', 'D2': 'F', 'NET': 'F', 'SERT': 'F', 'α2A': 'V', 'NMDA': 'X', 'GABA-A': 'X', 'H1': 'V', 'α1': 'U', 'M1': 'F', '5HT2C': 'V'},
        "Fr_renal": 0.5, "Fr_hepatic": 0.5, "pk_status": 'C', "pk_note": 'Label clearance falls ~30-50% in moderate-severe impairment -> implied Fr_ren 0.4-0.55 (urinary recovery 75% is mostly metabolites)',
        "Risk_QTc": 0.3, "qt_category": 'PR', "qt_status": 'V',
        "convulsant_index": 0.05, "seizure_status": 'C', "seizure_note": 'Label-type incidence 0.04%',
        "hypoNa_index": 0.14,
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
PHI_MAX = 8.0           # cap on induction-driven clearance gain (quetiapine clearance x7.5 with carbamazepine) [Ref 21]
E_MAX_STANDARD = 2.0     # CALIBRATED (provisional): memantine AUC x1.97 in moderate impairment is label-accepted without dose change, labelled reductions start at x2.3-2.5 [Ref 34]
# Drug-specific clearance ratio under carbamazepine/phenytoin induction (total-clearance ratio R; hepatic multiplier derived from Fr_hepatic)
INDUCTION_CL_RATIO = {
    "Quetiapine": (7.5, "V", "Carbamazepine 200 mg tid: AUC -87%, Cmax -80%, oral clearance x7.5 [Ref 21]"),
    "Risperidone": (2.5, "V", "Active-moiety concentration -50 to -70% -> clearance x2-3.3 (midpoint 2.5) [Ref 21]"),
    "Olanzapine": (1.46, "V", "Olanzapine clearance +46% (AUC -34%) in healthy volunteers [Ref 21]"),
    "Trazodone": (4.2, "V", "Carbamazepine 400 mg: trazodone concentration -76% -> clearance x4.2 [Ref 22]"),
    "Mirtazapine": (2.0, "V", "Carbamazepine 400 mg bid: clearance ~2-fold, concentration -60% (US label) [Ref 22]"),
}
INDUCTION_CL_DEFAULT = (2.0, "U", "PLACEHOLDER ratio for hepatically cleared drugs without data in this audit")
INDUCTION_MIN_FR_HEP = 0.25   # drugs with a smaller hepatic fraction are assumed unaffected by induction
INDUCER_AEDS = {"carbamazepine", "phenytoin"}

CENSOR_PKI = 5.0        # pKi <= 5.0 in the database = floor/"no meaningful binding" -> occupancy 0
SIGMA_PKI = 0.40        # inter-assay noise (log10 units); provisional value, re-estimated from all multi-source evidence in section 3 [Ref 14, 15]

THETA_STAR = 0.78       # D2 net-antagonism threshold for EPS: response >65%, prolactin >72%, EPS/akathisia >78% occupancy [Ref 8]
TAU_EPS = 0.03          # CALIBRATED (provisional): no EPS below 78% and 4 of 5 patients above it in Kapur 2000 -> steep window [Ref 8]
QTC_MAX = 500.0         # hard QTc limit (ms)
QT_EFFECT_MS = 20.0     # CALIBRATED (provisional): dQTc (ms) at Risk_QTc = 1 so that Known-risk agents (~0.8) reach ~16 ms and unlisted agents stay <5 ms, around the ICH E14 5-10 ms concern thresholds [Ref 30]
QT_MIN_MEANINGFUL_MS = 5.0    # ICH E14 mean-effect threshold of regulatory concern; smaller rises do not trigger the QTc hard constraint [Ref 30]
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
    "5HT2A": {"delusions": -0.7, "hallucinations": -0.8, "agitation": -0.15, "disinhibition": -0.5,
              "sleep": -0.6, "motor": -0.5, "appetite": -0.4},
    "D2":    {"agitation": -0.58, "delusions": -0.34, "hallucinations": -0.34, "apathy": +0.3, "motor": -0.5},   # agitation/psychosis: SMD/0.5 [Ref 31-33]
    "NET":   {"apathy": +0.8, "depression": +0.7},
    "SERT":  {"depression": +0.7, "anxiety": +0.6, "irritability": +0.4, "agitation": +0.52},   # agitation: SSRI SMD -0.26/0.5 [Ref 31]; others PLACEHOLDER
    "α2A":   {"agitation": -0.5, "irritability": -0.4},
    "NMDA":  {"apathy": -0.4, "delusions": -0.40, "hallucinations": -0.40},   # psychosis: memantine SMD -0.20/0.5 [Ref 32]; apathy PLACEHOLDER
    "GABA-A": {"anxiety": +0.7, "irritability": +0.30, "euphoria": +0.5, "disinhibition": +0.5},   # irritability halved: valproate RCTs negative for agitation/psychosis [Ref 31]
}

# Reference anchors: fix the unbound target-site concentration at the standard geriatric dose in
# normal organ function. ("occ", receptor, theta) back-calculates C_ref = Ki * theta / (1 - theta);
# ("conc", nM) gives the unbound concentration directly.
# PLACEHOLDER / ORDER-OF-MAGNITUDE ESTIMATES: replace with PET occupancy or measured unbound
# concentrations. These anchors are the single most influential uncalibrated inputs of the model.
REFERENCE_ANCHORS = {
    "Brexpiprazole": ("dose", "D2", 1.0, 0.873, 0.364),   # PET 64% at 1 mg, 80% at 4 mg [Ref 19]
    "Pimavanserin":  ("occ", "5HT2A", 0.80),
    "Risperidone":   ("dose", "D2", 1.0, 1.0, 1.08),      # PET: 2 mg barely reaches 65% [Ref 17]
    "Quetiapine":    ("occ", "D2", 0.30),      # PLACEHOLDER (time-average): PET peak 58-64% at 2-3 h, minimal by 12 h [Ref 21]
    "Olanzapine":    ("dose", "D2", 5.0, 1.0, 3.9),       # PET 55% at 5 mg, 73% at 10 mg [Ref 18]
    "Haloperidol":   ("dose", "D2", 1.0, 1.0, 0.97),      # PET: 65% at 1.5-2.1 mg [Ref 17]
    "Valproic Acid": ("occ", "GABA-A", 0.80),
    "Gabapentin":    ("occ", "GABA-A", 0.70),
    "Memantine":     ("occ", "NMDA", 0.50),   # PLACEHOLDER: NMDA Ki is in the µM range (~0.85 µM)
    "Escitalopram":  ("occ", "SERT", 0.80),   # PET: ~80% SERT occupancy at usual minimum dose (10 mg)
    "Sertraline":    ("occ", "SERT", 0.80),   # PET: ~80% at ~50 mg
    "Amitriptyline": ("conc", 1.0),
    "Venlafaxine":   ("occ", "SERT", 0.80),   # PET: ~80% at 75 mg
    "Vortioxetine":  ("dose", "SERT", 5.0, 1.0, 5.0),     # PET ~50% at 5 mg, 65% at 10 mg, 80% at 20 mg [Ref 20]
    "Bupropion":     ("conc", 80.0),
    "Trazodone":     ("conc", 50.0),
    "Mirtazapine":   ("conc", 17.0),
    "Mianserin":     ("conc", 5.5),
    "Lamotrigine":   ("conc", 1000.0),
    "Carbamazepine": ("conc", 1000.0),
}
DEFAULT_ANCHOR = ("conc", 10.0)
ANCHOR_LOGIT_SD = 0.35   # logit-SD of occupancy anchors: brexpiprazole PET 64 +/- 8% at 1 mg (logit SD ~0.35) [Ref 19]

# Risk domains: severity weight zeta, receptor liabilities beta (probability that full occupancy
# produces the adverse effect), and which patient-vulnerability function applies.
DOMAINS = {
    "sedation":    {"label": "Sedation / falls",                  "zeta": 0.20, "beta": {"H1": 0.40}},   # beta: mirtazapine somnolence 54% vs 18% placebo (excess 0.36 at θ_H1≈0.97), amitriptyline 60% [Ref 22]
    "orthostasis": {"label": "Orthostatic hypotension",           "zeta": 0.15, "beta": {"α1": 0.9}},
    "eps":         {"label": "Extrapyramidal (D2 window)",        "zeta": 0.25, "beta": {}},
    "qtc":         {"label": "QTc prolongation",                  "zeta": 0.25, "beta": {}},
    "cognition":   {"label": "Anticholinergic cognitive burden",  "zeta": 0.20, "beta": {"M1": 0.8}},
    "seizure":     {"label": "Seizure threshold",                 "zeta": 0.30, "beta": {}},
    "metabolic":   {"label": "Metabolic",                         "zeta": 0.10, "beta": {"H1": 0.5, "5HT2C": 0.7}},
    "hyponatremia": {"label": "Hyponatremia (SIADH)",             "zeta": 0.20, "beta": {"SERT": 0.55}},   # beta: SSRI hyponatremia 9.8% in older adults at θ_SERT 0.8 (reference patient age 75, Na 138) [Ref 24]
}
ZETA_HISTORY = 0.20     # severity weight of the soft history / interaction burden

# Rescaled-logistic vulnerability parameters: (x0, k, x_min, x_max). x0 and k carried over from the
# previous version (uncalibrated); the MMSE and seizure ranges are illustrative.
VULN_PARAMS = {
    "sedation_falls": (45.0, 0.08, 0.0, 125.0),   # Morse Fall Scale; x0 = high-risk cut-off ≥45 [Ref 27]
    "orthostasis":    (20.0, 0.25, 0.0, 60.0),    # standing SBP drop (mmHg); x0 = consensus definition ≥20 [Ref 28]
    "eps":            (8.0, 0.30, 0.0, 40.0),     # SAS score
    "qtc":            (470.0, 0.05, 400.0, 550.0),# baseline QTc (ms); x0 = prolonged threshold (women ≥470), >500 high risk [Ref 29]
    "cognition":      (20.0, 0.25, 0.0, 30.0),    # MMSE (inverted: low = vulnerable)
    "seizure":        (1.0, 0.8, 0.0, 10.0),      # seizures / year
    "hba1c":          (8.0, 1.0, 4.0, 15.0),
    "bmi":            (30.0, 0.15, 12.0, 50.0),
    "sodium":         (135.0, 0.5, 120.0, 145.0), # baseline serum Na (mmol/L), inverted; x0 = hyponatremia definition <135 [Ref 26]
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
THIAZIDE_RR = float(np.sqrt(1.24 * 11.2))      # geometric mean of adjusted OR 1.24 (large 2025 cohort) and earlier ORs 11.2-13.5 [Ref 25]
AGE_RR = float(np.sqrt(1.0 * 6.3))             # geometric mean of "no age difference" (2025 meta-analysis) and OR 6.3 (older review) [Ref 23, 24]


def _lam_na_reference():
    """Baseline-sodium vulnerability of the reference patient (Na 140 mmol/L), same rescaled-logistic form as section 4."""
    x0, k, lo, hi = VULN_PARAMS["sodium"]
    sg = lambda z: 1.0 / (1.0 + np.exp(-z))
    lam = (sg(k * (140.0 - x0)) - sg(k * (lo - x0))) / (sg(k * (hi - x0)) - sg(k * (lo - x0)))
    return 1.0 - float(lam)


def amplifier_from_rr(rr, lam0):
    """Amplifier a in lambda' = 1-(1-lambda)(1-a) that multiplies the risk of a patient with vulnerability lam0 by rr."""
    return (rr - 1.0) * lam0 / (1.0 - lam0)


THIAZIDE_AMPLIFIER = round(amplifier_from_rr(THIAZIDE_RR, _lam_na_reference()), 2)   # extra hyponatremia vulnerability with thiazide use
AGE_NA_AMPLIFIER_MAX = round(amplifier_from_rr(AGE_RR, _lam_na_reference()), 2)   # extra vulnerability reached at age >= 85 (ramp from 65)
SERT_ACTIVE_PKI = 7.0                          # pKi(SERT) at/above which an agent counts as SERT-active

SEROTONERGIC_DRUGS = ["Escitalopram", "Sertraline", "Venlafaxine", "Vortioxetine", "Amitriptyline"]
LEWY_SUBTYPES = ["Dementia with Lewy Bodies (DLB)", "Parkinson's Disease Dementia (PDD)"]
ANTIPSYCHOTIC_CATEGORIES = ["Atypical Antipsychotic", "Typical Antipsychotic"]



# Per-entry affinity uncertainty from the database (wider where sources disagree)
PKI_SIGMA = {}   # per-entry affinity uncertainty; filled in section 3 from the multi-source evidence


_DEFAULT_PARAMS = {"KAPPA": copy.deepcopy(KAPPA), "DOMAINS": copy.deepcopy(DOMAINS),
                   "VULN_PARAMS": dict(VULN_PARAMS), "ZETA_HISTORY": ZETA_HISTORY}


def apply_elicited_overrides(elicited=None):
    """Restore the calibrated defaults in place, then apply expert-panel medians (κ, ζ, β, vulnerability x0/k) if given.
    app.py calls this at the top of every Streamlit run: imported modules persist between reruns, so state is reset explicitly."""
    global ZETA_HISTORY
    KAPPA.clear()
    KAPPA.update(copy.deepcopy(_DEFAULT_PARAMS["KAPPA"]))
    DOMAINS.clear()
    DOMAINS.update(copy.deepcopy(_DEFAULT_PARAMS["DOMAINS"]))
    VULN_PARAMS.clear()
    VULN_PARAMS.update(_DEFAULT_PARAMS["VULN_PARAMS"])
    ZETA_HISTORY = _DEFAULT_PARAMS["ZETA_HISTORY"]
    el = elicited
    if not el:
        return
    for (rec, sym), val in el.get("kappa", {}).items():
        KAPPA.setdefault(rec, {})[sym] = float(np.clip(val, -1.0, 1.0))
    for key, val in el.get("zeta", {}).items():
        if key == "history":
            ZETA_HISTORY = float(max(val, 0.0))
        else:
            DOMAINS[key]["zeta"] = float(max(val, 0.0))
    for (key, rec), val in el.get("beta", {}).items():
        DOMAINS[key]["beta"][rec] = float(np.clip(val, 0.01, 0.99))
    for name, (x0, k) in el.get("vuln", {}).items():
        VULN_PARAMS[name] = (float(x0), float(k), VULN_PARAMS[name][2], VULN_PARAMS[name][3])

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
# 3.1a-2 Data-driven affinity uncertainty (calibration of SIGMA_PKI)
# -----------------------------------------------------------------------------
_NON_REPLICATE = {("Trazodone", "α1")}   # second value is the prior database entry, not an independent replicate


def estimate_sigma_pki():
    """Pooled SD of log10 Ki across independent sources (entries with >= 2 sources)."""
    ss, dfree, n_entries = 0.0, 0, 0
    for key, (kis, _) in PKI_EVIDENCE.items():
        if len(kis) > 1 and key not in _NON_REPLICATE:
            logs = np.log10(kis)
            ss += float(np.var(logs, ddof=1)) * (len(logs) - 1)
            dfree += len(logs) - 1
            n_entries += 1
    return (float(np.sqrt(ss / dfree)) if dfree else SIGMA_PKI), n_entries


SIGMA_PKI_ESTIMATE, SIGMA_PKI_N = estimate_sigma_pki()
SIGMA_PKI = round(max(SIGMA_PKI_ESTIMATE, 0.25), 2)
for (_d, _r), (_kis, _src) in PKI_EVIDENCE.items():
    if len(_kis) > 1:
        _half = (max(np.log10(_kis)) - min(np.log10(_kis))) / 2.0
        if _half > SIGMA_PKI:
            PKI_SIGMA[(_d, _r)] = round(float(_half), 2)

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


DOSE_OCCUPANCY_POINTS = [   # (drug, receptor, dose mg, published occupancy, source)
    ("Olanzapine", "D2", 5.0, 0.55, "PET, Kapur 1998 [Ref 18]"), ("Olanzapine", "D2", 10.0, 0.73, "PET, Kapur 1998 [Ref 18]"),
    ("Risperidone", "D2", 2.0, 0.65, "PET: 2 mg barely reaches 65% (Kapur & Seeman 2001) [Ref 17]"),
    ("Haloperidol", "D2", 1.8, 0.65, "PET: 65% at 1.5-2.1 mg (Kapur & Seeman 2001) [Ref 17]"),
    ("Brexpiprazole", "D2", 1.0, 0.64, "PET, Girgis 2020 (64 ± 8%) [Ref 19]"), ("Brexpiprazole", "D2", 4.0, 0.80, "PET, Girgis 2020 (80 ± 12%) [Ref 19]"),
    ("Vortioxetine", "SERT", 5.0, 0.50, "PET summary in US label / Stenkrona 2013 [Ref 20]"),
    ("Vortioxetine", "SERT", 10.0, 0.65, "PET summary in US label / Stenkrona 2013 [Ref 20]"),
    ("Vortioxetine", "SERT", 20.0, 0.80, "PET summary in US label / Stenkrona 2013 [Ref 20]"),
]


def dose_occupancy_df():
    rows = []
    for drug, rec, dose, obs, src in DOSE_OCCUPANCY_POINTS:
        a = REFERENCE_ANCHORS[drug]
        pred = a[3] * dose / (a[4] + dose)
        rows.append({"Drug": drug, "Receptor": rec, "Dose (mg)": dose, "Published occupancy": obs, "Model occupancy": round(pred, 3),
                     "Δ (pp)": round(100 * (pred - obs), 1), "Within ±5 pp": bool(abs(pred - obs) <= 0.05 + 1e-9), "Source": src})
    return pd.DataFrame(rows)


EFFECT_SIZE_OBS = {   # standardized mean difference vs placebo (negative = benefit)
    "agitation": {"Risperidone": (-0.26, "CMAI (Yunusa NMA, via Psychiatric Times review)"), "Brexpiprazole": (-0.30, "NPI-NH/CGI-S Cohen's d, 0.5-2 mg"),
                  "Sertraline": (-0.08, "Frontiers Aging Neurosci NMA 2023"), "Mirtazapine": (-0.04, "Frontiers Aging Neurosci NMA 2023"),
                  "Trazodone": (0.03, "Frontiers Aging Neurosci NMA 2023"), "Valproic Acid": (0.0, "negative large RCTs (no benefit vs placebo)")},
    "psychosis": {"Memantine": (-0.20, "Ageing Res Rev NMA 2022"), "Olanzapine": (-0.17, "Yunusa NMA (non-significant)"),
                  "Quetiapine": (0.04, "Yunusa NMA")},
}

KAPPA_EVIDENCE = [   # (receptor, symptom, kappa, status, basis)
    ("D2", "agitation", -0.58, "E", "SMD −0.26 (risperidone), −0.30 (aripiprazole, brexpiprazole): mean −0.29 → κ = 0.29/0.5 [Ref 31, 33]"),
    ("D2", "delusions", -0.34, "E", "Psychosis SMD −0.17 (aripiprazole, olanzapine) → κ = 0.17/0.5 [Ref 32, 33]"),
    ("D2", "hallucinations", -0.34, "E", "Psychosis SMD −0.17 (aripiprazole, olanzapine) → κ = 0.17/0.5 [Ref 32, 33]"),
    ("SERT", "agitation", 0.52, "E", "Citalopram −0.44, sertraline −0.08: mean −0.26 → κ = 0.26/0.5 [Ref 31]"),
    ("NMDA", "delusions", -0.40, "E", "Memantine psychosis SMD −0.20 → κ = 0.20/0.5 [Ref 32]"),
    ("NMDA", "hallucinations", -0.40, "E", "Memantine psychosis SMD −0.20 → κ = 0.20/0.5 [Ref 32]"),
    ("5HT2A", "agitation", -0.15, "E", "Trazodone (+0.03) and mirtazapine (−0.04) have ≈0 effect despite 5-HT2A antagonism [Ref 31]"),
    ("GABA-A", "irritability", 0.30, "E", "Large valproate RCTs negative for agitation/psychosis; magnitude halved [Ref 31]"),
]


def kappa_evidence_df():
    return pd.DataFrame([{"Receptor": r, "Symptom": s_, "κ in model": KAPPA[r][s_], "κ from evidence": k, "Status": st_, "Basis": b}
                         for r, s_, k, st_, b in KAPPA_EVIDENCE])


VULN_BASIS = {   # curve: (status, basis for x0)
    "sedation_falls": ("Criterion-based x0", "Morse Fall Scale ≥45 = high risk [Ref 27]"),
    "orthostasis": ("Criterion-based x0", "Orthostatic hypotension = SBP fall ≥20 mmHg [Ref 28]"),
    "qtc": ("Criterion-based x0", "Prolonged QTc ≥470 ms (women), >500 ms high risk [Ref 29]"),
    "sodium": ("Criterion-based x0", "Hyponatremia defined as Na <135 mmol/L [Ref 26]"),
    "hba1c": ("Criterion-based x0", "8% = less-stringent target in older adults with complex health (ADA)"),
    "bmi": ("Criterion-based x0", "BMI 30 = obesity (WHO)"),
    "eps": ("Placeholder", "SAS cut-off not verified"), "cognition": ("Placeholder", "MMSE 20 = moderate-impairment boundary; no outcome-linked cut-off"),
    "seizure": ("Placeholder", "No outcome-linked cut-off"),
}


def vuln_basis_df():
    return pd.DataFrame([{"Curve": n, "x0": VULN_PARAMS[n][0], "k (placeholder)": VULN_PARAMS[n][1], "Range": f"{VULN_PARAMS[n][2]}–{VULN_PARAMS[n][3]}",
                          "Status": st_, "Basis": b} for n, (st_, b) in VULN_BASIS.items()])


def induction_df():
    rows = [{"Drug": d, "Clearance ratio R": r, "Status": st_, "Basis": b} for d, (r, st_, b) in INDUCTION_CL_RATIO.items()]
    rows.append({"Drug": "all other drugs with Fr_hepatic ≥ 0.25", "Clearance ratio R": INDUCTION_CL_DEFAULT[0], "Status": INDUCTION_CL_DEFAULT[1], "Basis": INDUCTION_CL_DEFAULT[2]})
    return pd.DataFrame(rows)


HYPONATREMIA_OBS = {   # older-adult event rates of any hyponatremia (geriatric meta-analysis) [Ref 24]
    "Sertraline": (0.0981, "SSRIs"), "Escitalopram": (0.0981, "SSRIs"), "Venlafaxine": (0.1040, "SNRIs"),
    "Amitriptyline": (0.0350, "TCAs"), "Mirtazapine": (0.0309, "mirtazapine"),
}
SEDATION_OBS = {   # excess somnolence vs placebo (US label) [Ref 22]
    "Mirtazapine": (0.36, "54% vs 18%"), "Amitriptyline": (0.42, "60% vs 18% (as reported in the mirtazapine label)"),
}


def _expert_cols(df):
    return [c for c in df.columns if str(c).startswith("Expert")]


def parse_elicitation(kind, df, max_iqr=0.2, min_n=3):
    """Medians of completed Delphi sheets -> (overrides, summary table). Rows with <min_n ratings or IQR > max_iqr are not applied."""
    summ, out = [], {}
    if kind == "vuln":
        res = {}
        for _, r in df.iterrows():
            x0 = pd.to_numeric(r.get("Expert x0 (median of panel)"), errors="coerce")
            k = pd.to_numeric(r.get("Expert k (median of panel)"), errors="coerce")
            ok = pd.notna(x0) and pd.notna(k) and r.get("Curve") in VULN_PARAMS and k > 0
            summ.append({"Item": r.get("Curve"), "x0": None if pd.isna(x0) else float(x0), "k": None if pd.isna(k) else float(k), "Applied": bool(ok)})
            if ok:
                res[r["Curve"]] = (float(x0), float(k))
        return {"vuln": res}, pd.DataFrame(summ)
    cols = _expert_cols(df)
    for _, r in df.iterrows():
        vals = pd.to_numeric(pd.Series([r[c] for c in cols]), errors="coerce").dropna()
        n = len(vals)
        med = float(vals.median()) if n else None
        iqr = float(vals.quantile(0.75) - vals.quantile(0.25)) if n else None
        applied = bool(n >= min_n and iqr is not None and iqr <= max_iqr + 1e-12)
        item = f'{r["Receptor"]} / {r["Symptom"]}' if kind == "kappa" else str(r["Item"])
        summ.append({"Item": item, "n": n, "Median": med, "IQR": iqr, "Consensus (n≥3, IQR≤0.2)": applied})
        if not applied:
            continue
        if kind == "kappa":
            out.setdefault("kappa", {})[(r["Receptor"], r["Symptom"])] = med
        else:
            if item.startswith("ζ severity weight: "):
                label = item[len("ζ severity weight: "):]
                key = next((k for k, c in DOMAINS.items() if c["label"] == label), None)
                if key:
                    out.setdefault("zeta", {})[key] = med
            elif item.startswith("ζ history"):
                out.setdefault("zeta", {})["history"] = med
            elif item.startswith("β liability: "):
                label, rec = item[len("β liability: "):].rsplit(" / ", 1)
                key = next((k for k, c in DOMAINS.items() if c["label"] == label), None)
                if key and rec in DOMAINS[key]["beta"]:
                    out.setdefault("beta", {})[(key, rec)] = med
    return out, pd.DataFrame(summ)


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
                          "Expert x0 (median of panel)": "", "Expert k (median of panel)": "", "Cohort evidence / OR": ""} for n, p in VULN_PARAMS.items()])


# -----------------------------------------------------------------------------
# 3.2 Parameter register: every placeholder, its calibration status in this version, and next steps
# -----------------------------------------------------------------------------
REGISTER_COLUMNS = ["ID", "Group", "Parameter (current value)", "Status", "Applied in this version", "Why uncertain",
                    "Suggested source / value", "How to calibrate", "How to validate", "Priority"]
def build_placeholder_register():
    """Parameter register with the current (possibly elicited) values filled in."""
    _STATUS_COUNTS = {k: sum(1 for dd in DRUG_DATABASE.values() for v in dd["pKi_status"].values() if v == k) for k in "VCUFXP"}
    _N_DOSE_ANCHORS = sum(1 for a in REFERENCE_ANCHORS.values() if a[0] == "dose")
    _N_CONC_ANCHORS = sum(1 for a in REFERENCE_ANCHORS.values() if a[0] == "conc")

    PLACEHOLDER_REGISTER = [
        ("P01", "Exposure", f"Fr_renal / Fr_hepatic per drug; PHI_MIN = {PHI_MIN}; PHI_MAX = {PHI_MAX}; E_MAX_STANDARD = {E_MAX_STANDARD}",
         "Partially calibrated",
         "E_MAX_STANDARD 3.0 → 2.0 from label AUC data (memantine ×1.97 accepted without dose change; labelled reductions start at ×2.3–2.5) [Ref 34]; PHI_MAX 3 → 8 to allow quetiapine induction (×7.5) [Ref 21]; Fr re-checked for 13 drugs (5 corrected); 5 label AUC points tested.",
         "Global limits; drug-specific therapeutic index differs; 7 drugs' Fr still unverified.",
         "Label PK in renal/hepatic impairment (AUC ratio vs normal) and labelled dose adjustments.",
         "Fr = (1 − 1/AUC)/(1 − g); per-drug E_MAX = largest label-accepted AUC ratio.",
         "Implied Fr within ±0.2 of the DB for physical AUC points (self-check); lock decisions match labelled 'avoid' thresholds.", "High"),
        ("P02", "Exposure", f"INDUCTION_CL_RATIO: quetiapine ×7.5, risperidone ×2.5, olanzapine ×1.46, trazodone ×4.2, mirtazapine ×2.0; default ×{INDUCTION_CL_DEFAULT[0]} (others)",
         "Partially calibrated",
         "Generic ×2 factor replaced by drug-specific clearance ratios from carbamazepine interaction studies for 5 drugs [Ref 21, 22]; hepatic multiplier derived from Fr_hepatic; drugs with Fr_hepatic < 0.25 unaffected.",
         "Haloperidol and 9 other hepatically cleared drugs still use the default; phenytoin assumed equal to carbamazepine.",
         "DDI studies/labels for each substrate (haloperidol, sertraline, escitalopram, venlafaxine, vortioxetine, valproate, lamotrigine, amitriptyline, bupropion).",
         "Total-clearance ratio R from AUC ratio (R = 1/AUC ratio); phenytoin-specific values.",
         "Predicted exposure change within ±30% of published AUC ratios (quetiapine check in self-checks).", "High"),
        ("P03", "Affinity", f"SIGMA_PKI = {SIGMA_PKI} (pooled estimate {SIGMA_PKI_ESTIMATE:.2f} from {SIGMA_PKI_N} multi-source entries); CENSOR_PKI = {CENSOR_PKI}",
         "Calibrated (provisional)",
         f"σ re-estimated as the pooled SD of log10 Ki over all {SIGMA_PKI_N} entries with ≥2 independent sources (excluding the conflicting trazodone α1 pair); per-entry σ widened to half the log-spread where sources disagree [Ref 14, 15].",
         "Limited replicates; assay/species effects not separated; '>' values still floored rather than censored in the likelihood.",
         "All replicate Ki entries from PDSP, ChEMBL, IUPHAR per drug-receptor pair.",
         "Hierarchical log-normal model by receptor class; left-censoring for '>' values.",
         "Leave-one-source-out: held-out Ki inside the 90% interval in ≥85% of cases.", "High"),
        ("P04", "Anchors", f"REFERENCE_ANCHORS: {_N_DOSE_ANCHORS} dose–occupancy fits, SERT occupancy anchors, {_N_CONC_ANCHORS} concentration anchors, quetiapine/pimavanserin/gabapentin/valproate/memantine occupancy placeholders; ANCHOR_LOGIT_SD = {ANCHOR_LOGIT_SD}",
         "Partially calibrated",
         "Emax dose–occupancy curves fitted to PET points for risperidone (2 mg ≈65%), olanzapine (5/10 mg = 55/73%), haloperidol (65% at 1.5–2.1 mg), brexpiprazole (64/80% at 1/4 mg) and vortioxetine SERT (50/65/80% at 5/10/20 mg); anchors evaluated at the DB reference dose [Ref 17–20]; SERT ≈80% for escitalopram/sertraline/venlafaxine; anchor SD from brexpiprazole PET (±8 pp).",
         "Quetiapine occupancy is transient (peak 58–64%, minimal at 12 h); pimavanserin, memantine, GABAergic and tricyclic/trazodone/mirtazapine/mianserin/bupropion anchors remain estimates; PET data are from younger adults.",
         "PET occupancy at geriatric doses (D2: Kapur 2000/2001; SERT: Meyer 2004 and the 2021 review) or Css × fu × Kp,uu from label PK.",
         "Extend dose–occupancy fits to remaining drugs; use time-averaged occupancy for quetiapine; propagate PET SD.",
         "Predicted occupancy within ±5 pp of the fitted PET points (self-check); ±10 pp for new PET data.", "Critical"),
        ("P05", "D2 window", f"THETA_STAR = {THETA_STAR}; TAU_EPS = {TAU_EPS}", "Literature-based",
         "θ* 0.80 → 0.78 (EPS/akathisia threshold in Kapur 2000: none below 78%, 4 of 5 above); τ 0.05 → 0.03 to reproduce that step [Ref 8].",
         "Derived in first-episode adults; older and DLB/PDD patients may be harmed at lower occupancy; τ from very few events.",
         "PET–EPS studies in older adults.", "Logistic EPS probability vs D2 occupancy by age stratum.",
         "EPS hazard <5% at 65% and >80% at 85% occupancy (self-check); calibration slope 0.8–1.2 in age strata.", "High"),
        ("P06", "QTc", f"Risk_QTc per drug (CredibleMeds bands); QT_EFFECT_MS = {QT_EFFECT_MS}; QTC_MAX = {QTC_MAX}; QT_MIN_MEANINGFUL_MS = {QT_MIN_MEANINGFUL_MS}",
         "Partially calibrated",
         "Risk_QTc inside category bands (4 corrected; 2 unverified); QT_EFFECT_MS 40 → 20 and meaningful-rise threshold 10 → 5 ms to match the ICH E14 5–10 ms concern thresholds [Ref 30, CredibleMeds].",
         "No per-drug concentration–QTc slopes yet; carbamazepine and mianserin categories unverified.",
         "Thorough-QT results and label ΔQTc per drug; current CredibleMeds lists.",
         "Per-drug linear concentration–QTc slope m_j; map to category bands.",
         "Band check (self-check); predicted ΔQTc within ±5 ms of label values.", "High"),
        ("P07", "Engagement", f"U_MIN = {U_MIN}", "Placeholder", "Not changed.", "Arbitrary cut-off for 'no modelled engagement'.",
         "Utility equal to 10–20% occupancy of the primary target.", "Sensitivity sweep 0.01–0.10; list drugs that flip.",
         "No drug with trial evidence for the rated symptom is excluded across the sweep.", "Medium"),
        ("P08", "Pharmacology", "RECEPTOR_PROFILE (α, tone); ALPHA_OVERRIDES: brexpiprazole D2 α = 0.43", "Placeholder", "Not changed.",
         "Intrinsic activity is assay-specific; tone is not measured per patient.", "Functional assays (cAMP/GTPγS, % Emax), e.g., Maeda 2014.",
         "α = Emax relative to full agonist; propagate tone range 0.5–1.0.", "Top-3 ranking robust (≤1 position change) across tone 0.5–1.0.", "Medium"),
        ("P09", "Need map", "KAPPA (signed): D2 agitation 0.58, D2 psychosis 0.34, SERT agitation 0.52, NMDA psychosis 0.40, 5HT2A agitation 0.15, GABA-A irritability 0.30 set from effect sizes; others judgment-based",
         "Partially calibrated",
         "κ = |SMD|/0.5 for 6 couplings from network meta-analyses (agitation: risperidone −0.26, brexpiprazole −0.30, SSRIs −0.26, sertraline/mirtazapine/trazodone ≈0; psychosis: memantine −0.20, aripiprazole/olanzapine −0.17); valproate negative RCTs halved GABA-A irritability [Ref 31–33]; Delphi import enabled.",
         "Most couplings (≈25) still judgment-based; SMDs are pooled across multi-receptor drugs; 5-HT2A psychosis κ (0.7/0.8) is the main outlier vs olanzapine −0.17 and quetiapine +0.04 and awaits pimavanserin data; depression/anxiety/apathy not integrated; model over-predicts sertraline, trazodone and mirtazapine agitation utility.",
         "Meta-analytic effect sizes by symptom × mechanism; Delphi panel (≥10 clinicians, IQR ≤ 0.2).",
         "Import completed Delphi sheets in the Calibration Center (medians applied when n ≥ 3 and IQR ≤ 0.2); or regress effect size on receptor profile.",
         "Spearman between model utility and published effect sizes (table in Calibration Center) ≥ 0.6.", "Critical"),
        ("P10", "Safety weights", "DOMAINS ζ (0.10–0.30); caregiver scaling of sedation/orthostasis", "Placeholder (elicitation-ready)",
         "No evidence-based adjustment possible: ζ are value judgments. Delphi/swing-weighting import implemented (median applied when n ≥ 3 and IQR ≤ 0.2); ζ already feed Monte Carlo.",
         "Value judgments, not elicited.", "Swing-weighting or discrete-choice experiment with clinicians and caregivers.",
         "Import via Calibration Center; Dirichlet prior for Monte Carlo.", "Top-3 set unchanged for ±30% change of any single ζ in ≥80% of vignettes.", "Critical"),
        ("P11", "Liabilities", f"DOMAINS β: H1 sedation {DOMAINS['sedation']['beta']['H1']}, SERT hyponatremia {DOMAINS['hyponatremia']['beta']['SERT']}; α1 0.9, M1 0.8, metabolic 0.5/0.7 unchanged",
         "Partially calibrated",
         "H1 sedation β 0.9 → 0.40 from mirtazapine somnolence excess 36% (54% vs 18%; θ_H1 ≈ 0.97) and amitriptyline 42% (60% vs 18%) [Ref 22]; SERT hyponatremia β 0.50 → 0.55 so that SSRIs give ≈9.8% in older adults [Ref 24]; both checked in the Calibration Center.",
         "α1, M1 and metabolic β lack incidence-by-occupancy data in this audit; amitriptyline anchor under-predicts sedation.",
         "Incidence by occupancy/class from trials and cohorts.", "Fit β so that λ·h equals observed incidence.",
         "Predicted vs observed incidence within a factor of 2 (hyponatremia and sedation tables).", "High"),
        ("P12", "Vulnerability", "VULN_PARAMS: x0 from clinical cut-offs for Morse (45), orthostasis (20 mmHg), QTc (470 ms), sodium (135 mmol/L); k, SAS, MMSE, seizure, HbA1c, BMI unchanged",
         "Partially calibrated",
         "Centres moved to published thresholds: Morse ≥45 high risk [Ref 27]; orthostatic hypotension ≥20 mmHg [Ref 28]; prolonged QTc ≥470 ms [Ref 29]; hyponatremia <135 mmol/L [Ref 26]; HbA1c 8% (older-adult target) and BMI 30 kept as criterion-based.",
         "Slopes k are still judgment-based; SAS, MMSE and seizure curves not criterion-verified.",
         "Risk-factor odds ratios and external cohorts.", "x0 where risk doubles vs reference; k from OR slope; alternative link functions; Delphi import for k.",
         "Discrimination AUC ≥ 0.65 per curve in an external cohort.", "High"),
        ("P13", "Hyponatremia", f"hypoNa_index (mirtazapine 0.14, amitriptyline 0.04, carbamazepine 0.40); NA_HARD_LIMIT = {NA_HARD_LIMIT}; THIAZIDE_AMPLIFIER = {THIAZIDE_AMPLIFIER}; AGE_NA_AMPLIFIER_MAX = {AGE_NA_AMPLIFIER_MAX}; SERT_ACTIVE_PKI = {SERT_ACTIVE_PKI}",
         "Partially calibrated",
         "Indices set so older-adult event rates are reproduced (SSRI 9.8%, SNRI 10.4%, TCA 3.5%, mirtazapine 3.1%) [Ref 23, 24]; amplifiers derived from relative risks: thiazide RR 3.7 (geometric mean of 1.24 and 11.2), age RR 2.5 (1.0–6.3) [Ref 23–25]; Na limit 130 = guideline mild/moderate boundary [Ref 26].",
         "Thiazide and age effects conflict between studies; carbamazepine and antipsychotic indices lack data.",
         "Pharmacovigilance/cohort data for antipsychotics, trazodone, valproate; AGS Beers use-with-caution list.",
         "Replace geometric means by study-specific adjusted ORs once an effect-modifier model is fitted.", "Observed vs predicted event rate within a factor of 2 for 5 drugs (table + self-check).", "High"),
        ("P14", "Amplifiers", f"THYROID_QTC_AMPLIFIER = {THYROID_QTC_AMPLIFIER}; EPS_COMBO_AMPLIFIER = {EPS_COMBO_AMPLIFIER}; frailty × age ramp (75→90)", "Placeholder", "Not changed.",
         "Interaction strengths assumed.", "Cohort ORs (thyroid & QT; AChEI + SSRI + antipsychotic & EPS; frailty & falls).", "Convert ORs to amplifiers with amplifier_from_rr().", "As P12.", "Medium"),
        ("P15", "Soft burden", f"RHO_FAILURE = {RHO_FAILURE}; RHO_ACHEI_MAX = {RHO_ACHEI_MAX}; RHO_AED_DUPLICATE = {RHO_AED_DUPLICATE}; RHO_POLYPHARMACY = {RHO_POLYPHARMACY}; ZETA_HISTORY = {ZETA_HISTORY}",
         "Placeholder", "Not changed.", "Judgment-based.", "Re-challenge outcome data; anticholinergic + AChEI pharmacoepidemiology; polypharmacy event rates.",
         "Elicit as in P10 or estimate from claims data.", "±50% on each ρ leaves the top-3 set unchanged in ≥80% of vignettes.", "Medium"),
        ("P16", "Combinations", f"COMBO_SUPERIORITY_DELTA = {COMBO_SUPERIORITY_DELTA}; COMBO_STRONG_MARGIN = {COMBO_STRONG_MARGIN}; COMBO_U_TARGET = {COMBO_U_TARGET}",
         "Placeholder", "Not changed.", "Heuristic thresholds.", "Minimal clinically important index difference from expert-rated vignette pairs.",
         "δ = smallest difference experts distinguish with ≥75% agreement.", "Combination suggestions concordant with experts in ≥80% of paired vignettes.", "Medium"),
        ("P17", "Background", f"NONDB_BACKGROUND_HAZARD = {NONDB_BACKGROUND_HAZARD}", "Placeholder", "Not changed.", "Fixed anticholinergic hazard.",
         "ACB score → cognitive-decline/delirium risk from cohorts.", "Fit h = 1 − exp(−γ·ACB).", "Predicted vs observed risk by ACB stratum within a factor of 1.5.", "Medium"),
        ("P18", "Database", f"pKi entries (220): V {_STATUS_COUNTS['V']}, C {_STATUS_COUNTS['C']}, U {_STATUS_COUNTS['U']}, F {_STATUS_COUNTS['F']}, X {_STATUS_COUNTS['X']}, P {_STATUS_COUNTS['P']}; Fr, Risk_QTc, convulsant_index, hypoNa_index, induction ratios",
         "Partially calibrated",
         "60 affinity entries checked against multi-source evidence (18 corrected by |Δ|>0.5, trazodone α1 averaged, SERT added for all drugs); brexpiprazole SERT floor confirmed by PET (<5% occupancy at 4 mg) [Ref 19]; Fr, QT bands, seizure indices, induction ratios and hyponatremia indices re-based; dosage text not re-checked.",
         "20 unverified (U) and 3 proxy (P) affinity entries; 7 unverified Fr; valproate/gabapentin proxies.",
         "PDSP, ChEMBL, IUPHAR, labels.", "Scripted extraction with assay metadata; source columns in the DB.",
         "100% of safety-critical inputs cited; |ΔpKi| ≤ 0.5 vs median of independent sources (evidence table).", "High"),
        ("P19", "Seizure", "convulsant_index per drug", "Partially calibrated",
         "Bupropion (SIR 4.08 → 0.76), amitriptyline (SIR 1.58 → 0.37), olanzapine/quetiapine/risperidone (label incidence 0.9/0.8/0.3% on one scale), mirtazapine re-based.",
         "Index mixes incidence-based and SIR-based sources.", "Label seizure incidences and SIRs from pharmacoepidemiology.",
         "Express every index as 1 − 1/RR against untreated baseline.", "Ordering agrees with published relative risks (Spearman ≥ 0.7).", "Medium"),
        ("P20", "Uncertainty", f"Sampler widths: anchor logit-SD {ANCHOR_LOGIT_SD}, concentration anchors 0.5 log-SD, ζ log-SD 0.3, κ/β concentration ν = 20, vulnerability jitter", "Partially calibrated",
         "Anchor SD set from PET variability; affinity σ from evidence (P03).", "Other widths chosen for plausibility.", "Posterior distributions from the calibration steps above.",
         "Replace fixed widths by calibrated posteriors.", "90% intervals cover held-out expert-score variation or bootstrap replicates in ≥85% of cases.", "Medium"),
    ]
    return PLACEHOLDER_REGISTER


PLACEHOLDER_REGISTER = build_placeholder_register()

# -----------------------------------------------------------------------------
# 3.3 VALIDATION & IMPROVEMENT ROADMAPS
# -----------------------------------------------------------------------------
VALIDATION_COLUMNS = ["Tier", "What is tested", "Acceptance criterion (suggested)", "Status in this app"]
VALIDATION_PLAN = [
    ("V1 Software verification", "Bounds, monotonicity, competitive binding, exact vulnerability endpoints, database integrity", "All checks pass on every code change", "Implemented (self-checks)"),
    ("V2 Face validity", "Hard constraints and mechanism-appropriate rankings in canonical vignettes", "≥90% agreement with panel on hard constraints; ≥80% top-3 concordance on ≥30 vignettes", "Partly implemented; vignettes to be expanded"),
    ("V3 Input verification", "pKi, Fr, QT and seizure inputs against independent sources", "|ΔpKi| ≤ 0.5 vs median of sources; every input cited", "Partly implemented (evidence, band, AUC, dose-occupancy, hyponatremia and sedation checks)"),
    ("V4 Anchor verification", "Predicted occupancy at standard dose vs PET", "Within ±10 percentage points", "Partly implemented (PET dose-occupancy fits for 5 drugs + SERT anchors)"),
    ("V5 Hazard calibration", "Predicted vs observed adverse-event incidence", "Calibration slope 0.8–1.2; within factor 1.5–2", "Partly implemented (hyponatremia, sedation)"),
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

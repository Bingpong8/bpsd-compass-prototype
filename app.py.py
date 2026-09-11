import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="BPSD Compass Prototype (P5)", layout="wide")
st.title("BPSD Compass Prototype (P5)")
st.caption("Parameter-driven neurotransmitter affinity decision-support tool with dynamic regimen tracking")

ascii_header = r""" THE DEATH OF PEACE OF MIND

---(-.
{        )   )
,   ) -~~- ( ,-' )
(   -,\_..., )-- ',)
( \_)  (  -\~( -\_,  }
(-  _  _-~~~~,  ,' )  -^(    _;-,((()))
~~~~ { -_(())
`\  }
{ }
Dolor et Astra, Nihil est Veritas
"""
st.code(ascii_header, language=None)

### -----------------------------------------------------------------------------
### 1. FULL PHARMACODYNAMIC DATABASE (RESTORED P4 & DRAFT SPECTRUM)
### -----------------------------------------------------------------------------
# pKi values (-log10(Ki)) & Intrinsic Activity (Ar: +1 = agonist/target, -1 = inverse/antagonist toxicity, 0 = neutral)
DRUG_DATABASE = {
    "Brexpiprazole": {
        "pKi": {"5HT2A": 9.33, "D2": 9.52, "NET": 5.00, "a2a": 8.00, "NMDA": 5.00, "GABA_A": 5.00, "H1": 7.72, "a1": 8.42, "M1": 5.50},
        "Ar":  {"5HT2A": 1.0,  "D2": 1.0,  "NET": 0.0,  "a2a": 1.0,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "Atypical Antipsychotic", "start_dose": "0.5 mg/day", "max_dose": "2.0 mg/day"
    },
    "Aripiprazole": {
        "pKi": {"5HT2A": 8.40, "D2": 8.80, "NET": 5.00, "a2a": 6.80, "NMDA": 5.00, "GABA_A": 5.00, "H1": 7.50, "a1": 7.30, "M1": 5.00},
        "Ar":  {"5HT2A": 1.0,  "D2": 1.0,  "NET": 0.0,  "a2a": 0.5,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "Atypical Antipsychotic", "start_dose": "2.0 mg/day", "max_dose": "15.0 mg/day"
    },
    "Pimavanserin": {
        "pKi": {"5HT2A": 9.05, "D2": 5.00, "NET": 5.00, "a2a": 5.00, "NMDA": 5.00, "GABA_A": 5.00, "H1": 5.00, "a1": 5.00, "M1": 5.00},
        "Ar":  {"5HT2A": 1.0,  "D2": 0.0,  "NET": 0.0,  "a2a": 0.0,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "Selective 5-HT2A Inverse Agonist", "start_dose": "34 mg/day", "max_dose": "34 mg/day"
    },
    "Risperidone": {
        "pKi": {"5HT2A": 9.30, "D2": 8.43, "NET": 5.00, "a2a": 7.20, "NMDA": 5.00, "GABA_A": 5.00, "H1": 7.70, "a1": 8.15, "M1": 5.00},
        "Ar":  {"5HT2A": 1.0,  "D2": 1.0,  "NET": 0.0,  "a2a": 0.5,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "Atypical Antipsychotic", "start_dose": "0.25 mg/day", "max_dose": "2.0 mg/day"
    },
    "Quetiapine": {
        "pKi": {"5HT2A": 6.70, "D2": 5.80, "NET": 6.60, "a2a": 6.10, "NMDA": 5.00, "GABA_A": 5.00, "H1": 8.00, "a1": 7.00, "M1": 6.00},
        "Ar":  {"5HT2A": 1.0,  "D2": 1.0,  "NET": 1.0,  "a2a": 0.5,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "Atypical Antipsychotic", "start_dose": "12.5 mg/day", "max_dose": "150.0 mg/day"
    },
    "Olanzapine": {
        "pKi": {"5HT2A": 8.40, "D2": 7.96, "NET": 5.00, "a2a": 6.20, "NMDA": 5.00, "GABA_A": 5.00, "H1": 8.15, "a1": 7.72, "M1": 7.59},
        "Ar":  {"5HT2A": 1.0,  "D2": 1.0,  "NET": 0.0,  "a2a": 0.5,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "Atypical Antipsychotic", "start_dose": "2.5 mg/day", "max_dose": "10.0 mg/day"
    },
    "Haloperidol": {
        "pKi": {"5HT2A": 7.10, "D2": 8.70, "NET": 5.00, "a2a": 5.00, "NMDA": 5.00, "GABA_A": 5.00, "H1": 6.00, "a1": 7.20, "M1": 5.00},
        "Ar":  {"5HT2A": 1.0,  "D2": -1.0, "NET": 0.0,  "a2a": 0.0,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "Typical Antipsychotic", "start_dose": "0.5 mg/day", "max_dose": "2.0 mg/day"
    },
    "Mirtazapine": {
        "pKi": {"5HT2A": 8.10, "D2": 5.00, "NET": 5.00, "a2a": 7.80, "NMDA": 5.00, "GABA_A": 5.00, "H1": 9.00, "a1": 7.20, "M1": 5.00},
        "Ar":  {"5HT2A": 1.0,  "D2": 0.0,  "NET": 0.0,  "a2a": 1.0,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "NaSSA Antidepressant", "start_dose": "7.5 mg/day", "max_dose": "30.0 mg/day"
    },
    "Memantine": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "NET": 5.00, "a2a": 5.00, "NMDA": 6.30, "GABA_A": 5.00, "H1": 5.00, "a1": 5.00, "M1": 5.00},
        "Ar":  {"5HT2A": 0.0,  "D2": 0.0,  "NET": 0.0,  "a2a": 0.0,  "NMDA": 1.0,  "GABA_A": 0.0},
        "class": "NMDA Antagonist", "start_dose": "5.0 mg/day", "max_dose": "20.0 mg/day"
    },
    "Clonidine": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "NET": 5.00, "a2a": 7.50, "NMDA": 5.00, "GABA_A": 5.00, "H1": 5.00, "a1": 6.20, "M1": 5.00},
        "Ar":  {"5HT2A": 0.0,  "D2": 0.0,  "NET": 0.0,  "a2a": 1.0,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "Alpha-2 Agonist", "start_dose": "0.05 mg/day", "max_dose": "0.3 mg/day"
    },
    "Lorazepam": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "NET": 5.00, "a2a": 5.00, "NMDA": 5.00, "GABA_A": 7.80, "H1": 5.00, "a1": 5.00, "M1": 5.00},
        "Ar":  {"5HT2A": 0.0,  "D2": 0.0,  "NET": 0.0,  "a2a": 0.0,  "NMDA": 0.0,  "GABA_A": 1.0},
        "class": "Benzodiazepine", "start_dose": "0.25 mg/day", "max_dose": "2.0 mg/day"
    },
    "Escitalopram": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "NET": 5.00, "a2a": 5.00, "NMDA": 5.00, "GABA_A": 5.00, "H1": 5.00, "a1": 5.00, "M1": 5.00},
        "Ar":  {"5HT2A": 0.0,  "D2": 0.0,  "NET": 0.0,  "a2a": 0.0,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "SSRI Antidepressant", "start_dose": "5.0 mg/day", "max_dose": "10.0 mg/day"
    },
    "Sertraline": {
        "pKi": {"5HT2A": 5.00, "D2": 6.60, "NET": 5.00, "a2a": 5.00, "NMDA": 5.00, "GABA_A": 5.00, "H1": 5.00, "a1": 5.00, "M1": 5.00},
        "Ar":  {"5HT2A": 0.0,  "D2": 0.5,  "NET": 0.0,  "a2a": 0.0,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "SSRI Antidepressant", "start_dose": "25.0 mg/day", "max_dose": "100.0 mg/day"
    },
    "Fluoxetine": {
        "pKi": {"5HT2A": 6.70, "D2": 5.00, "NET": 5.00, "a2a": 5.00, "NMDA": 5.00, "GABA_A": 5.00, "H1": 5.00, "a1": 5.00, "M1": 5.00},
        "Ar":  {"5HT2A": 0.5,  "D2": 0.0,  "NET": 0.0,  "a2a": 0.0,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "SSRI Antidepressant", "start_dose": "10.0 mg/day", "max_dose": "20.0 mg/day"
    },
    "Nortriptyline": {
        "pKi": {"5HT2A": 7.50, "D2": 5.00, "NET": 7.90, "a2a": 6.30, "NMDA": 5.00, "GABA_A": 5.00, "H1": 8.00, "a1": 7.20, "M1": 6.70},
        "Ar":  {"5HT2A": 1.0,  "D2": 0.0,  "NET": 1.0,  "a2a": 0.0,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "TCA Antidepressant", "start_dose": "10.0 mg/day", "max_dose": "50.0 mg/day"
    },
    "Amitriptyline": {
        "pKi": {"5HT2A": 7.80, "D2": 5.00, "NET": 7.40, "a2a": 6.10, "NMDA": 5.00, "GABA_A": 5.00, "H1": 8.90, "a1": 7.60, "M1": 8.00},
        "Ar":  {"5HT2A": 1.0,  "D2": 0.0,  "NET": 1.0,  "a2a": 0.0,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "TCA Antidepressant", "start_dose": "10.0 mg/day", "max_dose": "50.0 mg/day"
    },
    "Venlafaxine": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "NET": 5.60, "a2a": 5.00, "NMDA": 5.00, "GABA_A": 5.00, "H1": 5.00, "a1": 5.00, "M1": 5.00},
        "Ar":  {"5HT2A": 0.0,  "D2": 0.0,  "NET": 1.0,  "a2a": 0.0,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "SNRI Antidepressant", "start_dose": "37.5 mg/day", "max_dose": "150.0 mg/day"
    },
    "Duloxetine": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "NET": 7.70, "a2a": 5.00, "NMDA": 5.00, "GABA_A": 5.00, "H1": 5.00, "a1": 5.00, "M1": 5.00},
        "Ar":  {"5HT2A": 0.0,  "D2": 0.0,  "NET": 1.0,  "a2a": 0.0,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "SNRI Antidepressant", "start_dose": "20.0 mg/day", "max_dose": "60.0 mg/day"
    },
    "Donepezil": {
        "pKi": {"5HT2A": 5.00, "D2": 5.00, "NET": 5.00, "a2a": 5.00, "NMDA": 5.00, "GABA_A": 5.00, "H1": 5.00, "a1": 5.00, "M1": 5.00},
        "Ar":  {"5HT2A": 0.0,  "D2": 0.0,  "NET": 0.0,  "a2a": 0.0,  "NMDA": 0.0,  "GABA_A": 0.0},
        "class": "AChE Inhibitor", "start_dose": "5.0 mg/day", "max_dose": "10.0 mg/day"
    }
}

### -----------------------------------------------------------------------------
### 2. RISK ENGINE & REGIMEN EVALUATION FUNCTIONS
### -----------------------------------------------------------------------------
def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))

def calculate_risks(fall_risk, orthostasis, eps, qtc):
    lambda_H1 = sigmoid(fall_risk, 45, 0.1) 
    lambda_a1 = sigmoid(orthostasis, 20, 0.2)
    lambda_D2 = sigmoid(eps, 3, 1.0)
    qtc_lock = True if qtc > 500 else False
    return lambda_H1, lambda_a1, lambda_D2, qtc_lock

def evaluate_regimen_adjustment(drug, history_data):
    duration = history_data["duration"]
    outcome = history_data["outcome"]
    
    note = ""
    history_penalty = 0.0
    hard_lock = False
    is_failed = False

    if outcome == "Severe Adverse Effects":
        hard_lock = True
        is_failed = True
        note = f"Severe toxicity/intolerance at {history_data['dosage']}. Discontinue immediately and initiate a 10-day washout/taper period."
        return history_penalty, note, hard_lock, is_failed

    if outcome in ["Minimal Improvement (CGI-I 3) / NPI Unresolved", "No Improvement / Worsening"]:
        if drug in ["Risperidone", "Quetiapine"]:
            if duration <= 21:
                note = f"Trial underway ({duration} days). Extend trial up to 21-28 days before final decision."
            elif duration >= 42:
                history_penalty += 20.0
                is_failed = True
                note = f"Trial failed at day {duration} (≥42 days extension limit reached). Initiate cross-titration to alternative target."
            else:
                note = f"In extension window (day {duration}). Re-evaluate response closely before day 42 limit."
        else:
            if duration >= 21:
                history_penalty += 20.0
                is_failed = True
                note = f"Trial failed at day {duration} (≥21 days limit). Initiate cross-titration switch."
            else:
                note = f"Early trial phase ({duration} days). Optimize dose toward target range; re-evaluate at day 21."
    elif outcome == "Good Response / Stabilized":
        note = f"Maintenance Phase: Patient stabilized on {history_data['dosage']}. Re-evaluate every 4-6 weeks; consider tapering after 3-4 months of stability."
        history_penalty -= 5.0 # Suited/Bonus score

    return history_penalty, note, hard_lock, is_failed

### -----------------------------------------------------------------------------
### 3. USER INTERFACE & BEDSIDE INPUTS
### -----------------------------------------------------------------------------
st.sidebar.header("1. Symptom Targets (NPI Equivalent)")
npi_agitation = st.sidebar.slider("Agitation/Psychosis (w_5HT2A, w_D2)", 0.0, 1.0, 0.8)
npi_apathy = st.sidebar.slider("Apathy/Executive (w_NET, w_NMDA)", 0.0, 1.0, 0.2)
npi_affective = st.sidebar.slider("Affective/Anxiety/Sleep (w_GABA, w_a2a)", 0.0, 1.0, 0.4)

st.sidebar.header("2. Risk Vulnerabilities")
fall_risk = st.sidebar.number_input("Morse Fall Scale Score", 0, 100, 50)
orthostasis = st.sidebar.number_input("Orthostatic SBP Drop (mmHg)", 0, 50, 15)
eps = st.sidebar.number_input("Simpson-Angus Scale (SAS)", 0, 10, 1)
qtc = st.sidebar.number_input("Baseline QTc (ms)", 350, 600, 440)

st.sidebar.header("3. Caregiver/Clinician Directives")
cg_preference = st.sidebar.selectbox("Treatment Priority", ["Balanced Profile", "Minimize Sedation", "Fastest Onset"])

# Main UI Area: Medical Regimen Tracking
st.subheader("💊 Active & Prior Medical Profile / Regimen Trial Status")
st.caption("Document active or previous trials to compute current status, re-evaluate skipped drugs, and schedule cross-titration.")

prior_drugs = st.multiselect(
    "Select Current or Prior Psychotropic Regimens",
    options=list(DRUG_DATABASE.keys()),
    default=[]
)

patient_history = {}
if prior_drugs:
    for drug in prior_drugs:
        c_d1, c_d2, c_d3 = st.columns([1, 1, 2])
        with c_d1:
            dosage = st.text_input(f"Dose ({drug})", value=DRUG_DATABASE[drug]['start_dose'], key=f"dose_{drug}")
        with c_d2:
            duration = st.number_input(f"Days on {drug}", min_value=1, max_value=365, value=21, key=f"dur_{drug}")
        with c_d3:
            outcome = st.selectbox(
                f"Response to {drug}",
                ["Minimal Improvement (CGI-I 3) / NPI Unresolved", "No Improvement / Worsening", "Good Response / Stabilized", "Severe Adverse Effects"],
                key=f"out_{drug}"
            )
        patient_history[drug] = {"dosage": dosage, "duration": duration, "outcome": outcome}

st.markdown("---")

### -----------------------------------------------------------------------------
### 4. CALCULATION ENGINE
### -----------------------------------------------------------------------------
lambda_H1, lambda_a1, lambda_D2, qtc_lock = calculate_risks(fall_risk, orthostasis, eps, qtc)

results = []
failed_or_locked_drugs = []
clinical_notes = {}

for drug, data in DRUG_DATABASE.items():
    pKi = data["pKi"]
    Ar = data["Ar"]
    
    # Therapeutic Match Score
    thera_score = (npi_agitation * pKi["5HT2A"] * Ar.get("5HT2A", 0.0)) + \
                  (npi_agitation * pKi["D2"] * Ar.get("D2", 0.0)) + \
                  (npi_apathy * pKi["NET"] * Ar.get("NET", 0.0)) + \
                  (npi_apathy * pKi["NMDA"] * Ar.get("NMDA", 0.0)) + \
                  (npi_affective * pKi["GABA_A"] * Ar.get("GABA_A", 0.0)) + \
                  (npi_affective * pKi["a2a"] * Ar.get("a2a", 0.0))
                  
    # Off-Target Risk Penalty
    risk_score = (lambda_H1 * pKi["H1"]) + \
                 (lambda_a1 * pKi["a1"]) + \
                 (lambda_D2 * pKi["D2"])
                 
    # Anticholinergic Cognitive Burden (ACB) Penalty based on M1 affinity
    P_ACB = 3.0 if pKi["M1"] >= 7.0 else (1.5 if pKi["M1"] >= 6.0 else 0.0)
    
    # Net Match Score Calculation
    net_score = thera_score - risk_score - P_ACB
    
    # Evaluate Prior History Adjustments
    hard_locked = False
    if drug in patient_history:
        history_penalty, note, h_lock, is_failed = evaluate_regimen_adjustment(drug, patient_history[drug])
        net_score -= history_penalty
        clinical_notes[drug] = note
        hard_locked = h_lock
        if is_failed or hard_locked:
            failed_or_locked_drugs.append(drug)
            
    # QTc Hard Lock Constraint
    if qtc_lock and drug in ["Haloperidol", "Quetiapine", "Amitriptyline", "Nortriptyline"]:
        net_score = -999.0
        clinical_notes[drug] = "CONTRAINDICATED: Baseline QTc > 500 ms."
        hard_locked = True
        
    if hard_locked:
        net_score = -999.0

    results.append({
        "Medication": drug,
        "Class": data["class"],
        "Net Match Score": round(net_score, 2),
        "Therapeutic Score": round(thera_score, 2),
        "Risk Penalty": round(risk_score, 2),
        "ACB Penalty": P_ACB,
        "Start Dose": data["start_dose"],
        "Max Dose": data["max_dose"],
        "Status": "Locked/Failed" if net_score <= -500.0 else "Viable"
    })

df_results = pd.DataFrame(results).sort_values(by="Net Match Score", ascending=False)

### -----------------------------------------------------------------------------
### 5. RECOMMENDATIONS & CLINICAL DECISION OUTPUTS
### -----------------------------------------------------------------------------
top_drug = df_results.iloc[0]

st.header(f"🏆 Primary Recommendation: {top_drug['Medication']}")
st.write(f"**Drug Class:** {top_drug['Class']} | **Net Match Score:** {top_drug['Net Match Score']} | **Recommended Starting Dose:** {top_drug['Start Dose']} (Max: {top_drug['Max Dose']})")

# Skipped First-Line & Regimen Action Alerts
if failed_or_locked_drugs or prior_drugs:
    st.markdown("### ⚠️ Active Regimen Action & Skipped First-Line Alerts")
    
    # Current active drug status notes
    for p_drug in prior_drugs:
        note_text = clinical_notes.get(p_drug, "Continue monitoring as indicated.")
        if p_drug in failed_or_locked_drugs:
            st.error(f"**{p_drug} Action Notice:** {note_text}")
        else:
            st.info(f"**{p_drug} Status Notice:** {note_text}")
            
    # Reconsideration of Skipped Medications Logic
    skipped_candidates = df_results[(df_results["Net Match Score"] > 0) & (~df_results["Medication"].isin(prior_drugs))]
    if not skipped_candidates.empty and failed_or_locked_drugs:
        st.warning("💡 **Reconsider Skipped Medications:** The following non-trialed options demonstrate superior net receptor suitability. Consider cross-titrating to one of these primary targets:")
        for idx, row in skipped_candidates.head(3).iterrows():
            st.write(f"- **{row['Medication']}** ({row['Class']}) — Net Score: **{row['Net Match Score']}** | Start Dose: {row['Start Dose']}")

st.markdown("### 📊 Comprehensive Receptor Suitability Matrix")
st.dataframe(
    df_results.style.map(lambda x: 'background-color: #ffcccc' if x == 'Locked/Failed' else 'background-color: #ccffcc', subset=['Status']),
    use_container_width=True
)

### -----------------------------------------------------------------------------
### 6. EXPANDERS (RESTORED CLINICAL GUIDELINES & MATHEMATICAL ARCHITECTURE)
### -----------------------------------------------------------------------------
with st.expander("📐 Mathematical Engine & Algorithmic Equations"):
    st.markdown("""
    #### 1. Net Match Score Equation ($M_j$)
    The Net Suitability Match Score ($M_j$) for a candidate psychotropic drug $j$ is calculated as:
    $$M_j = T_j - R_j - P_{\text{ACB}} - P_{\text{History}}$$
    
    Where:
    *   **Therapeutic Target Score ($T_j$):**
        $$T_j = \sum_{r} w_r \cdot pK_{i,j,r} \cdot A_{r,j}$$
        Calculated over target receptors $r \in \{5\text{-HT}_{2\text{A}}, D_2, \text{NET}, \text{NMDA}, \text{GABA}_{\text{A}}, \alpha_{2\text{a}}\}$ using clinical symptom weights $w_r$ derived from NPI sub-scores[span_8](start_span)[span_8](end_span)[span_9](start_span)[span_9](end_span).
    *   **Risk Vulnerability Penalty ($R_j$):**
        $$R_j = \lambda_{H1} \cdot pK_{i,j,H1} + \lambda_{\alpha1} \cdot pK_{i,j,\alpha1} + \lambda_{D2} \cdot pK_{i,j,D2}$$
        Where risk weights ($\lambda$) are non-linearly scaled via Sigmoid curves from physiological bedside metrics (Morse Fall Scale, Orthostatic Blood Pressure drop, SAS score)[span_10](start_span)[span_10](end_span).
    *   **Anticholinergic Cognitive Burden ($P_{\text{ACB}}$):**
        Dose-independent penalty based on $M_1$ muscarinic receptor affinity ($pK_{i,M1} \ge 7.0 \Rightarrow 3.0$; $pK_{i,M1} \ge 6.0 \Rightarrow 1.5$)[span_11](start_span)[span_11](end_span).
    *   **Prior History Penalty ($P_{\text{History}}$):**
        Penalty applied to previously failed regimens (15.0 to 20.0 pts) or hard-lock score ($-999.0$) upon severe adverse events[span_12](start_span)[span_12](end_span).
    """)

with st.expander("🔄 Cross-Titration & Washout Protocol Timelines"):
    st.markdown("""
    #### Standardized Discontinuation & Switching Matrix
    When switching from an ineffective or poorly tolerated regimen to a newly recommended primary agent:
    1. **10-Day Tapering Schedule:** Reduce active drug dose by 50% on Day 1, and further to 25% on Day 5 prior to complete discontinuation on Day 10[span_13](start_span)[span_13](end_span).
    2. **Cross-Titration Overlap:** Introduce the new agent at its minimum starting dose (e.g., Brexpiprazole 0.5 mg/day or Risperidone 0.25 mg/day) when the primary drug has been reduced to 50%[span_14](start_span)[span_14](end_span).
    3. **Monitoring Window:** Perform symptom re-evaluations using NPI or CGI-I every 1–2 weeks during titration, and maintain post-withdrawal evaluation for at least 4 months to monitor for recurrence[span_15](start_span)[span_15](end_span).
    """)

with st.expander("📚 Pharmacological & Guideline Citations"):
    st.markdown("""
    1. **Dynamic Target Algorithms:** Maps BPSD clinical phenotypes directly to monoaminergic, cholinergic, and GABAergic receptor profiles using binding potencies ($pK_i$)[span_16](start_span)[span_16](end_span)[span_17](start_span)[span_17](end_span).
    2. **Treatment Duration Rules:** Standard antipsychotic trials require 21 days for efficacy decision-making. Risperidone and quetiapine protocol allows extension up to 42 days before deeming a trial completely failed[span_18](start_span)[span_18](end_span)[span_19](start_span)[span_19](end_span).
    3. **Guideline Consensus:** Incorporates principles from APA (2015), NICE (2018), and CANADA guidelines (2018) regarding regular reassessment, tapering considerations after 3–4 months of stability, and avoidance of unnecessary long-term antipsychotics[span_20](start_span)[span_20](end_span).
    """)

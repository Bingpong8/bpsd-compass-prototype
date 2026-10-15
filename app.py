# ==============================================================================
# BPSD COMPASS PROTOTYPE — FRAMEWORK V3.1 (PROD INFRASTRUCTURE)
# ARCHITECTURAL PROFILE: Swiss Industrial Brutalist / Neue Haas Grotesk Core
# ZERO-SLOP COMPLIANT: No rounded radii, no gradients, strict monochrome, 
# instant linear transitions, dense asymmetric data layout.
# ==============================================================================

import numpy as np
import pandas as pd
import streamlit as st
from database import *   # noqa: F401,F403
from engine import *     # noqa: F401,F403

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION & ARCHITECTURAL CSS INJECTION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="BPSD Compass Interface (P5 - Framework v3.1)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <style>
    /* TYPOGRAPHY & MOTIF: Neue Haas Grotesk Core & Neo-Grotesk Fallback */
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: "Neue Haas Grotesk Display Pro", "Neue Haas Grotesk Text Pro", "Helvetica Neue", Helvetica, Arial, sans-serif !important;
        color: #0B0B0B;
        background-color: #FFFFFF;
        -webkit-font-smoothing: antialiased;
    }

    /* STRICT MONOCHROMATIC PALETTE & ZERO-RADIUS (ANTI-SLOP LAW) */
    .stButton>button, input, select, textarea, .stSelectbox div[data-baseweb="select"], .stMultiSelect div[data-baseweb="select"] {
        border-radius: 0px !important;
        border: 1px solid #222222 !important;
        background-color: #FFFFFF !important;
        color: #0B0B0B !important;
        transition: background-color 0.1s linear, color 0.1s linear, border-color 0.1s linear !important;
    }

    /* INTERACTION: EMIL KOWALSKI MOTION PRINCIPLES (<150ms linear, no bounce) */
    .stButton>button:hover {
        background-color: #0B0B0B !important;
        color: #FFFFFF !important;
        border-color: #0B0B0B !important;
    }

    /* FOCUS STATES */
    button:focus-visible, input:focus-visible, select:focus-visible, div[data-baseweb="select"] > div:focus-visible {
        outline: 2px solid #0B0B0B !important;
        outline-offset: 0px !important;
    }

    /* REMOVE ALL AI SLOP (Gradients, Glassmorphism, Shadows, Blurs) */
    * {
        box-shadow: none !important;
        backdrop-filter: none !important;
        background-image: none !important;
    }

    /* TYPOGRAPHIC HIERARCHY */
    h1, h2, h3, h4 {
        letter-spacing: -0.03em !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        color: #0B0B0B !important;
    }

    /* DATA DENSITY & TABLE BRUTALISM */
    .dataframe {
        border: 1px solid #222222 !important;
        border-radius: 0px !important;
    }

    /* ALERT / BANNER OVERRIDES FOR MONOCHROME BRUTALISM */
    div.stAlert {
        border-radius: 0px !important;
        border: 1px solid #222222 !important;
        background-color: #F5F5F5 !important;
        color: #0B0B0B !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------------------------------------
# HEADER & METADATA STRUCTURE
# ------------------------------------------------------------------------------
st.markdown("<span style='font-size: 11px; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;'>SYSTEM ARCHITECTURE // BPSD COMPASS V3.1</span>", unsafe_allow_html=True)
st.title("BPSD COMPASS INTERFACE (P5)")
st.markdown("<p style='font-size: 14px; font-weight: 400; line-height: 1.5; margin-top: -10px;'>Parameter-driven decision support engine — occupancy-based framework v3.1 (Occupancy & Hazard Matrix)</p>", unsafe_allow_html=True)

ascii_header = r"""
								THE DEATH OF PEACE OF MIND
													
								      _---~~(~~-_.
								    _{        )   )
								  ,   ) -~~- ( ,-' )_
								 (   `-,\_..`., )-- '_,)
								( `)_  (  -\~( -\_`,  }
								(_-  _  ~_-~~~~`,  ,' )
								  `~ -^(    __;-,((()))
								        ~~~~ {_ -_(())
								               `\  }
								                 { }
																	 
							     Dolor et Astra, Nihil est Veritas
"""
st.code(ascii_header, language=None)

if "ruled_out" not in st.session_state:
    st.session_state.ruled_out = set()
apply_elicited_overrides(st.session_state.get("elicited_params"))

st.warning(
    "**FRAMEWORK V3.1 (Occupancy-based, audited and partly calibrated database).** Reference anchors, severity weights, risk liabilities and vulnerability curves are "
    "partly calibrated: see the Calibration, Audit & Improvement Center. This prototype is not validated for clinical use."
)

with st.expander("MODEL & UNCERTAINTY SETTINGS (FRAMEWORK V3.1)", expanded=False):
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

st.markdown("---")
st.markdown("### 01 // PATIENT CLINICAL PARAMETERS")
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

with st.expander("SPECIAL CONDITIONS RISK INPUTS (EPILEPSY, NCDS, COMORBIDITIES)", expanded=False):
    col_sc1, col_sc2 = st.columns(2)
    with col_sc1:
        seizure_freq_input = st.number_input("Seizure Frequency (events/year)", 0.0, 50.0, 0.0, step=0.5)
        seizure_freq_val = int(np.round(seizure_freq_input))
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
st.markdown("### 02 // EXPANDED PRIOR MEDICATION AUDIT & REGIMEN RISK STRATIFICATION")

col_prior1, col_prior2 = st.columns(2)
ALL_AUDIT_OPTIONS = list(DRUG_DATABASE.keys()) + MAOI_AGENTS + ANTICHOLINERGIC_AGENTS + ACHEI_AGENTS + ["Phenytoin (Dilantin)"]

with col_prior1:
    prior_drugs = st.multiselect(
        "Select Active / Prior Psychotropic & Neurological Medications",
        options=ALL_AUDIT_OPTIONS
    )

combined_prior_drugs = list(set(prior_drugs + [AED_TO_DRUG_NAME[aed] for aed in active_aeds_val if aed in AED_TO_DRUG_NAME]))
OUTCOMES = ["Active Regimen - Ongoing", "Partial Response / Tolerated", "Treatment Failure / Ineffective", "Severe Adverse Effects / Intolerant"]
ACTIVE = OUTCOMES[0]
prior_history = {}

if combined_prior_drugs:
    with col_prior2:
        for drug in combined_prior_drugs:
            if drug in prior_drugs:
                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    dosage = st.text_input(f"Dose for {drug}", value=" mg/day", key=f"dose_{drug}")
                with c_d2:
                    outcome = st.selectbox(f"Outcome for {drug}", OUTCOMES, key=f"outcome_{drug}")
            else:
                dosage = DRUG_DATABASE.get(drug, {}).get("dosage", "Standard dose")
                outcome = ACTIVE
            prior_history[drug] = {"dosage": dosage, "outcome": outcome}

active_drugs = [d for d, h in prior_history.items() if h["outcome"] == ACTIVE]
active_psychotropic_count = sum(d in DRUG_DATABASE or d in MAOI_AGENTS or d in ANTICHOLINERGIC_AGENTS for d in active_drugs)
has_active_maoi = any(d in MAOI_AGENTS for d in active_drugs)
has_active_achei = any(d in ACHEI_AGENTS for d in active_drugs)
has_active_ssri = any(d in ("Escitalopram", "Sertraline") for d in active_drugs)

active_aeds_final = list({aed.lower() for aed in active_aeds_val} | {AED_NAME_MAP[d] for d in combined_prior_drugs if d in AED_NAME_MAP})
epilepsy_active = seizure_freq_val > 0 or bool(active_aeds_final)

if epilepsy_active:
    st.warning(
        f"**EPILEPSY PROTOCOL ACTIVE**: Baseline AED dosing is **strictly mandatory** (Current integer seizure frequency: **{seizure_freq_val} events/year**). "
        "Do NOT reduce or discontinue AED dosages during psychotropic cross-titration or regimen optimization."
    )

polypharmacy_branch = active_psychotropic_count >= 2
if polypharmacy_branch:
    st.warning(
        f"**Branch B Stratification Triggered (Polypharmacy Burden = {active_psychotropic_count} Active Agents):** "
        "Patient is taking $\\ge 2$ active psychotropics. Escalating directly to Rational Combination Pharmacotherapy & Regimen Optimization."
    )
else:
    st.info(
        f"**Branch A Stratification (Low Drug Burden = {active_psychotropic_count} Active Agents):** "
        "Standard Sequential Monotherapy Pathway active."
    )

st.markdown("---")
st.markdown("### 03 // TARGET SYMPTOM SEVERITY (12 NPI SUBSCALES)")

NPI_ITEMS = (  
    ("delusions", "Delusions (5-HT2A / D2 Target)", 1), ("hallucinations", "Hallucinations (5-HT2A Target)", 1),
    ("agitation", "Agitation / Aggression (D2 / α2A Target)", 2), ("depression", "Depression / Dysphoria (NET / 5-HT Target)", 1),
    ("anxiety", "Anxiety (GABA-A / 5-HT Target)", 1), ("euphoria", "Euphoria / Elation (GABA-A Target)", 0),
    ("apathy", "Apathy / Indifference (NET / NMDA Target)", 1), ("disinhibition", "Disinhibition (GABA-A / 5-HT Target)", 0),
    ("irritability", "Irritability / Lability (GABA-A / α2A Target)", 1), ("motor", "Aberrant Motor Behavior (D2 / 5-HT2A Target)", 0),
    ("sleep", "Sleep / Night-time Disturbances (5-HT2A Target)", 1), ("appetite", "Appetite / Eating Changes (5-HT2A Target)", 0),
)
npi_options = list(NPI_MAPPING.keys())
v = {}
for col, items in zip(st.columns(3), (NPI_ITEMS[:4], NPI_ITEMS[4:8], NPI_ITEMS[8:])):
    with col:
        for key, label, index in items:
            v[key] = NPI_MAPPING[st.selectbox(label, npi_options, index=index)]

st.markdown("---")
st.markdown("### 04 // CAREGIVER PRIORITIES & SHARED DECISION (PETRUSHKA MODEL)")
caregiver_options = list(CAREGIVER_CONCERN_MAPPING.keys())
col_pref1, col_pref2 = st.columns(2)

with col_pref1:
    pref_sedation = CAREGIVER_CONCERN_MAPPING[st.selectbox("Caregiver Avoid-Sedation Weight", caregiver_options, index=1)]
with col_pref2:
    pref_falls = CAREGIVER_CONCERN_MAPPING[st.selectbox("Caregiver Avoid-Fall Concern Weight", caregiver_options, index=1)]

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
    "polypharmacy_branch": polypharmacy_branch,
    "mmse": float(mmse_score),
    "qtc_ms": float(qtc_ms),
    "egfr": float(egfr_val),
    "lft_impairment": float(lft_val),
    "morse": float(morse_score),
    "sas": float(sas_score),
    "sodium": float(sodium_val),
    "thiazide": thiazide_val,
    "dementia_subtype": dementia_subtype,
    "pref_sedation": pref_sedation,
    "pref_falls": pref_falls,
    "ruled_out": st.session_state.ruled_out,
    "dose_policy": dose_policy,
    "background_mode": background_mode,
    "active_db_drugs": [d for d in active_drugs if d in DRUG_DATABASE],
    "active_nondb": [d for d in active_drugs if d in NONDB_BACKGROUND_HAZARD],
}

# Decision engine execution
symptoms_present = any(val > 0.0 for val in v.values())
S0 = Sampler(1)  
results = sorted(
    (result_row(evaluate_single(drug, drug_data, patient_profile, S0, v, prior_history), patient_profile, prior_history)
     for drug, drug_data in DRUG_DATABASE.items()),
    key=lambda x: x["Raw_Mj"], reverse=True)
top_drug = results[0]

def mj_interval(m):
    return "—" if np.isnan(m["p05"]) else f"{100 * m['p05']:.0f} … {100 * m['p95']:.0f}"
	
mc = None
if mc_draws > 0 and symptoms_present:
    mc = run_monte_carlo(patient_profile, v, prior_history, mc_draws)
for r in results:
    if mc is not None and not r["Hard Locked"]:
        m = mc[r["Drug"]]
        r["P(rank 1)"] = f"{100 * m['P_first']:.0f}%"
        r["Mj 5–95%"] = mj_interval(m)
    else:
        r["P(rank 1)"], r["Mj 5–95%"] = "—", "—"

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
        st.info("No target symptoms are rated (all NPI items = 0), so no therapeutic need can be matched. Rate at least one symptom.")
    else:
        st.error("No suitable candidate found. All agents were excluded by Stage A hard constraints "
                 "(contraindications, QTc, exposure limits, no modelled therapeutic engagement) or manual rule-outs.")
        if any(r["Lock Reason"].startswith("Standard dose infeasible") for r in results):
            st.info("Several agents are excluded only because organ impairment makes the **standard dose** unsafe. "
                    "Switch the dose policy in *Model & uncertainty settings* to **Auto-reduce dose** to rank them at a clearance-adjusted dose.")
else:
    if comb_analysis["is_viable"]:
        st.warning(
            "**Multi-Agent Combination Regimen Suggested:** Monotherapy coverage is constrained or Polypharmacy Branch B is active. "
            f"Combining **{' + '.join(comb_analysis['regimen'])}** (competitive-binding occupancy model) scores higher "
            f"(Combined Match Index: **{comb_analysis['score']}**, +{comb_analysis['margin']} index points over the best single agent)."
        )

    mc_line = ""
    if mc is not None:
        mt = mc[top_drug["Drug"]]
        mc_line = (f"<br/>P(rank 1) = <strong>{100 * mt['P_first']:.0f}%</strong> &nbsp;|&nbsp; "
                   f"Match Index 90% interval: <strong>{mj_interval(mt)}</strong> &nbsp;|&nbsp; Monte Carlo draws: {mc_draws}")

    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 1px solid #0B0B0B; border-left: 6px solid #0B0B0B; padding: 22px; margin-bottom: 15px;">
            <span style="font-size: 11px; color: #0B0B0B; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;">
                RECOMMENDED CANDIDATE OPTION
            </span>
            <h2 style="color: #0B0B0B; margin: 6px 0 0 0; font-size: 28px; font-weight: 700;">
                {top_drug['Drug']} <span style="font-size: 16px; font-weight: 400;">({top_drug['Category']})</span>
            </h2>
            <p style="color: #0B0B0B; font-size: 14px; margin: 10px 0 0 0; line-height: 1.5;">
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

    if mc is not None:
        ranked_mc = sorted(((nm, m["P_first"]) for nm, m in mc.items()), key=lambda t: t[1], reverse=True)
        if ranked_mc[0][1] - ranked_mc[1][1] < 0.10:
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
        <div style="background-color: #F5F5F5; border: 1px solid #222222; color: #0B0B0B; padding: 12px 18px; margin-bottom: 15px; font-size: 14px;">
            <strong>Recommended Dosage Spectrum:</strong> {top_drug['Dosage']}{dose_scaling_msg}
        </div>
        """,
        unsafe_allow_html=True
    )

    if top_drug['Drug'] == "Gabapentin":
        st.info(f"**Organ Monitoring Alert (Renal)**: 100% Renal Elimination ($Fr_\\text{{renal}} = 1.00$). Current eGFR = {egfr_val} mL/min; exposure factor E = {top_drug['Exposure Factor (E)']}. Strict renal dose titration required.")
    elif top_drug['Drug'] == "Valproic Acid":
        st.info(f"**Organ Monitoring Alert (Hepatic)**: High Hepatic Metabolism ($Fr_\\text{{hepatic}} = 0.95$); exposure factor E = {top_drug['Exposure Factor (E)']}. Baseline LFTs, CBC, and plasma level monitoring (target 50–80 µg/mL) required.")
    elif top_drug['Drug'] == "Memantine" and egfr_val < 30:
        st.info(f"**Organ Monitoring Alert (Renal)**: High Renal Elimination ($Fr_\\text{{renal}} = 0.80$). Current eGFR = {egfr_val} mL/min (< 30 mL/min). Maximum dose cap: 10 mg daily.")

    if top_drug["Occupancy"]["SERT"] >= 0.30:
        st.info(f"**Hyponatremia Monitoring Alert**: SERT occupancy ≈ {100 * top_drug['Occupancy']['SERT']:.0f}%. "
                f"Baseline Na = {sodium_val:.0f} mmol/L{' with thiazide use' if thiazide_val else ''}; hyponatremia index = {top_drug['Hyponatremia risk']}. "
                f"Recheck sodium at ~2 and ~4 weeks (and with any new confusion, falls or nausea).")

    if top_drug['Clinical Correlation Note']:
        st.info(f"**Clinical Correlation Note:** {top_drug['Clinical Correlation Note']}")

    st.markdown(
        f"""
        <div style="background-color: #FFFFFF; border: 1px solid #0B0B0B; border-left: 6px solid #0B0B0B; color: #0B0B0B; padding: 18px; margin-bottom: 15px;">
            <h4 style="margin: 0 0 6px 0; color: #0B0B0B; font-weight: 700; font-size: 15px;">
            	 CLINICAL WARNINGS FOR {top_drug['Drug'].upper()}
            </h4>
            <p style="margin: 0; font-size: 14px; line-height: 1.5; color: #0B0B0B;">
                {top_drug['Warnings']}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if st.checkbox(
        f"🚫 Rule Out **{top_drug['Drug']}** (Clinical contraindication / Tailored preference)",
        key=f"chk_ruleout_{top_drug['Drug']}"
    ):
        st.session_state.ruled_out.add(top_drug["Drug"])
        st.rerun()

# ------------------------------------------------------------------------------
# DASHBO

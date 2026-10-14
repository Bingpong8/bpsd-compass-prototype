"""BPSD Compass prototype (framework v3.1): Streamlit UI. Run with `streamlit run app.py`. Not validated for clinical use."""
import numpy as np
import pandas as pd
import streamlit as st
from database import *   # noqa: F401,F403
from engine import *     # noqa: F401,F403

st.set_page_config(page_title="BPSD Compass Prototype (P5 - Framework v3.1)", layout="wide")
st.title("BPSD Compass Prototype (P5)")
st.caption("Parameter-driven decision support tools — occupancy-based framework v3.1")

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

if "ruled_out" not in st.session_state:
    st.session_state.ruled_out = set()
apply_elicited_overrides(st.session_state.get("elicited_params"))

st.warning(
    "**Framework v3.1 (occupancy-based, audited and partly calibrated database).** Reference anchors, severity weights, risk liabilities and vulnerability curves are "
    "partly calibrated: see the Calibration, Audit & Improvement Center. This prototype is not validated for clinical use."
)
with st.expander("Model & uncertainty settings (framework v3.1)", expanded=False):
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

with st.expander("Special Conditions Risk Inputs (Epilepsy, NCDs, Comorbidities)", expanded=False):
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
st.subheader("Step 1: Expanded Prior Medication Audit & Regimen Risk Stratification")

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
st.subheader("Target Symptom Severity (12 NPI Subscales)")

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
st.subheader("Caregiver Priorities & Shared Decision (PETRUSHKA Model)")
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

# Decision engine
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
        <div style="background-color: #e2f0d9; border-left: 6px solid #385723; color: #274411; padding: 12px 18px; border-radius: 6px; margin-bottom: 15px; font-size: 15px;">
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
        <div style="background-color: #fff3cd; border: 1px solid #ffebaa; border-left: 8px solid #ffc107; color: #856404; padding: 18px; border-radius: 6px; margin-bottom: 15px;">
            <h4 style="margin: 0 0 6px 0; color: #856404; font-weight: 800; font-size: 17px;">
            	 Clinical Warnings for {top_drug['Drug']}
            </h4>
            <p style="margin: 0; font-size: 15px; line-height: 1.5; color: #533f03;">
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

# --- Dashboards ---
with st.expander("Sequential Cross-Titration & Dose Optimization Protocol", expanded=True):
    distinct_priors = [d for d in combined_prior_drugs if d != top_drug["Drug"]]

    if top_drug["Drug"] in combined_prior_drugs:
        st.info(
            f"**Active Regimen Optimization Protocol:** **{top_drug['Drug']}** is already part of the patient's active regimen. "
            f"Rather than cross-tapering, evaluate optimizing current dosage toward targeted spectrum (**{top_drug['Dosage']}**)."
        )

    if distinct_priors:
        prior_selected = st.selectbox(
            "Select distinct active/prior agent to cross-taper from:",
            options=distinct_priors,
            key="cross_taper_selector"
        )
        if epilepsy_active and prior_selected in AED_NAME_MAP:
            st.error(
                f"**CRITICAL EPILEPSY SAFETY LOCK**: **{prior_selected}** is an essential AED in a patient with active seizure history/epilepsy. "
                f"Baseline AED dosing is mandatory; do NOT taper or discontinue baseline AEDs during psychotropic cross-titration."
            )
        else:
            st.markdown(f"### Cross-Titration: Taper **{prior_selected}** $\\rightarrow$ Initiate **{top_drug['Drug']}**")
            st.table(generate_cross_titration_schedule(prior_selected, top_drug["Drug"]))
    elif top_drug["Drug"] not in combined_prior_drugs:
        st.info("Treatment Naive: Initiate top candidate at starting dose without cross-tapering.")

with st.expander("📊 Patient-Weighted Marginal Risk Indices (framework v3.1)", expanded=True):
    st.markdown("#### Marginal risk ΔR = λ · (1 − h_bg) · h  (0–100, uncalibrated index — not a probability)")
    risk_bars = (("Sedation / Falls", "Sedation risk"), ("Orthostatic", "Orthostatic risk"),
                 ("EPS (D2 window)", "EPS risk"), ("Hyponatremia", "Hyponatremia risk"))
    for col, (title, key) in zip(st.columns(4), risk_bars):
        with col:
            st.write(f"**{title} Index:**")
            st.progress(top_drug[key])
            st.caption(f"Index: {top_drug[key]}")

with st.expander("Receptor Occupancy & Domain Risk Breakdown (top-ranked candidate)", expanded=False):
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
    with st.expander("Uncertainty & Sensitivity (Monte Carlo)", expanded=False):
        def index_or_none(x):
            return None if np.isnan(x) else round(100 * x, 1)

        mc_rows = sorted(
            [{"Drug": nm, "P(rank 1)": round(m["P_first"], 3), "P(feasible)": round(m["P_feasible"], 3),
              "Mj p05": index_or_none(m["p05"]), "Mj median": index_or_none(m["p50"]), "Mj p95": index_or_none(m["p95"])}
             for nm, m in mc.items()],
            key=lambda r: r["P(rank 1)"], reverse=True
        )
        st.dataframe(pd.DataFrame(mc_rows).head(10), use_container_width=True)
        if not top_drug["Hard Locked"]:
            shares = variance_shares(top_drug["Drug"], patient_profile, v, prior_history, mc_draws)
            st.markdown(f"**Which uncertainties drive the score of {top_drug['Drug']}?** (one-at-a-time variance share)")
            st.dataframe(pd.DataFrame(
                [{"Uncertainty group": Sampler.GROUP_LABELS[g], "Variance share": f"{100 * val:.0f}%"} for g, val in
                 sorted(shares.items(), key=lambda kv: kv[1], reverse=True)]
            ), use_container_width=True)
            st.caption("Large shares identify the inputs that most need better evidence (calibration targets).")

if show_anchors:
    with st.expander("Reference Anchors (PET-fitted where marked; others placeholders)", expanded=True):
        anchor_labels = {"occ": "Occupancy (placeholder)", "dose": "Dose–occupancy (PET fit)", "conc": "Unbound conc. (nM)"}
        anchor_rows = []
        for nm, a in REFERENCE_ANCHORS.items():
            receptor_occ = anchor_receptor_occupancy(a)   # None for concentration anchors
            anchor_rows.append({"Drug": nm, "Anchor type": anchor_labels[a[0]],
                                "Receptor": receptor_occ[0] if receptor_occ else "—",
                                "Value": round(receptor_occ[1], 3) if receptor_occ else a[1]})
        st.dataframe(pd.DataFrame(anchor_rows), use_container_width=True)
        st.caption("Replace with PET occupancy or measured unbound concentrations at the standard geriatric dose.")

with st.expander("🚦 Dashboard & Ranking", expanded=False):
    df_results = pd.DataFrame(results)

    df_results_display = df_results[[
        "Drug", "Category", "Net Score (Mj)", "P(rank 1)", "Mj 5–95%", "Therapeutic Gain",
        "Risk Deductions", "History Burden", "Exposure Factor (E)", "Suggested Dose Scaling",
        "Dosage", "Lock Reason"
    ]]

    def apply_traffic_lights(val):
        if val > 10.0:
            return 'background-color: #d4edda; color: #155724; font-weight: bold;'
        if val >= -10.0:
            return 'background-color: #fff3cd; color: #856404;'
        return 'background-color: #f8d7da; color: #721c24;'

    st.dataframe(
        df_results_display.style.map(apply_traffic_lights, subset=['Net Score (Mj)']),
        use_container_width=True
    )

st.markdown("---")
st.subheader("Phenotype Cluster Analysis (CCSMH Model)")
p_agitation_psychosis = (v["agitation"] + v["delusions"] + v["hallucinations"]) / 3.0
p_affective = (v["depression"] + v["anxiety"] + v["irritability"]) / 3.0
p_apathy_executive = (v["apathy"] + v["disinhibition"]) / 2.0

col_ph1, col_ph2, col_ph3 = st.columns(3)
col_ph1.metric("Agitation-Psychosis Cluster", f"{p_agitation_psychosis*100:.0f}%")
col_ph2.metric("Affective-Lability Cluster", f"{p_affective*100:.0f}%")
col_ph3.metric("Apathy-Executive Cluster", f"{p_apathy_executive*100:.0f}%")

# --- Calibration, audit & improvement center ---
with st.expander("Calibration, Audit & Improvement Center", expanded=False):
    reg_df = pd.DataFrame(build_placeholder_register(), columns=REGISTER_COLUMNS)
    st.markdown("### 1. Calibration status at a glance")
    st.dataframe(reg_df["Status"].value_counts().rename_axis("Status").reset_index(name="Parameter groups"), use_container_width=True)
    pr_df = reg_df[reg_df["Priority"].isin(["Critical", "High"])][["ID", "Group", "Priority", "Status"]]
    st.markdown("**Critical and High priority items**")
    st.dataframe(pr_df, use_container_width=True)
    st.markdown("### 2. Parameter register: status, what was applied, and how to calibrate / validate")
    st.dataframe(reg_df, use_container_width=True)
    st.download_button("⬇️ Download register (CSV)", reg_df.to_csv(index=False).encode("utf-8"),
                       file_name="parameter_register.csv", mime="text/csv")
    st.markdown("### 3. Improvement roadmap")
    st.dataframe(pd.DataFrame(IMPROVEMENT_PLAN, columns=IMPROVEMENT_COLUMNS), use_container_width=True)
    st.markdown("### 4. Validation roadmap")
    st.dataframe(pd.DataFrame(VALIDATION_PLAN, columns=VALIDATION_COLUMNS), use_container_width=True)
    st.markdown("### 5. Database audit status")
    st.caption("  ·  ".join(f"**{code}** = {meaning}" for code, meaning in STATUS_LEGEND.items()))
    st.dataframe(db_status_df(), use_container_width=True)
    st.markdown("### 6. Affinity evidence audit (database vs literature)")
    st.caption(f"Pooled affinity noise σ = {SIGMA_PKI} log units (estimate {SIGMA_PKI_ESTIMATE:.2f} from {SIGMA_PKI_N} multi-source entries).")
    st.dataframe(pki_audit_df(), use_container_width=True)
    st.markdown("### 7. Exposure: renal AUC calibration, drug-specific induction ratios")
    st.dataframe(pk_calibration_df(), use_container_width=True)
    st.dataframe(induction_df(), use_container_width=True)
    st.markdown("### 8. Anchors: PET dose–occupancy fits")
    st.dataframe(dose_occupancy_df(), use_container_width=True)
    st.markdown("### 9. Need map (κ) from effect sizes, and concordance with published trials")
    st.dataframe(kappa_evidence_df(), use_container_width=True)
    conc_df, conc_rho = effect_size_concordance_df()
    st.dataframe(conc_df, use_container_width=True)
    st.caption("Spearman ρ (model utility vs observed benefit): " + "; ".join(f"{domain}: ρ = {rho:.2f} (n = {n})" for domain, (rho, n) in conc_rho.items())
               + ". Small n: interpret as a direction check, not as validation.")
    st.markdown("### 10. Hazard calibration against incidence")
    st.dataframe(hyponatremia_calibration_df(), use_container_width=True)
    st.dataframe(sedation_calibration_df(), use_container_width=True)
    st.markdown("### 11. QTc categories, seizure indices and vulnerability-curve bases")
    st.dataframe(qt_band_df(), use_container_width=True)
    st.dataframe(seizure_df(), use_container_width=True)
    st.dataframe(vuln_basis_df(), use_container_width=True)
    st.markdown("### 12. Expert elicitation (Delphi): export templates, import completed sheets")
    templates = (("⬇️ κ matrix", "kappa", "delphi_kappa.csv"), ("⬇️ ζ / β weights", "zeta_beta", "delphi_zeta_beta.csv"),
                 ("⬇️ Vulnerability curves", "vuln", "delphi_vulnerability.csv"))
    for col, (label, kind, file_name) in zip(st.columns(3), templates):
        with col:
            st.download_button(label, delphi_template_df(kind).to_csv(index=False).encode("utf-8"), file_name=file_name, mime="text/csv")
    st.caption("Fill the Expert 1…10 columns (vulnerability sheet: panel medians), then upload. A row is applied only if n ≥ 3 ratings and IQR ≤ 0.2.")
    elicited = dict(st.session_state.get("elicited_params") or {})
    for kind, label, key in (("kappa", "κ matrix", "up_kappa"), ("zeta_beta", "ζ / β weights", "up_zb"), ("vuln", "Vulnerability curves", "up_vuln")):
        up = st.file_uploader(f"Upload completed {label} (CSV)", type="csv", key=key)
        if up is not None:
            try:
                res, summ = parse_elicitation(kind, pd.read_csv(up))
                st.dataframe(summ, use_container_width=True)
                if st.button(f"Apply {label}", key=f"apply_{key}"):
                    for section, values in res.items():
                        elicited.setdefault(section, {}).update(values)
                    st.session_state["elicited_params"] = elicited
                    st.rerun()
            except Exception as exc:   # malformed file
                st.error(f"Could not read the sheet: {exc}")
    if elicited:
        st.info(f"Elicited parameters are active ({', '.join(elicited)}). All results use them.")
        if st.button("Reset to calibrated defaults", key="btn_reset_elicited"):
            st.session_state["elicited_params"] = {}
            st.rerun()
    st.markdown("### 13. Model self-checks (verification, face validity, database integrity, calibration)")
    if st.button("Run self-checks", key="btn_selfchecks"):
        chk = pd.DataFrame(run_self_checks())
        st.dataframe(chk, use_container_width=True)
        n_fail = int((chk["Result"] == "FAIL").sum())
        (st.success if n_fail == 0 else st.error)(f"{len(chk) - n_fail}/{len(chk)} checks passed.")

# --- Rationale ---
with st.expander("Algorithmic Architecture & Clinical Rationale", expanded=False):
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

    $$h_{dj} = 1-\prod_{r\in R_d}(1-\beta_{rd}\theta_{rj}),\qquad h_{\text{EPS}} = \sigma\!\left(\frac{\theta_{D_2}^{\text{net}}-0.78}{0.03}\right),\qquad h_{\text{QTc}} = 1-(1-R_{\text{QTc}})^{E_j}$$

    $$\lambda_d(x)=\frac{\sigma(k(x-x_0))-\sigma(k(x_{\min}-x_0))}{\sigma(k(x_{\max}-x_0))-\sigma(k(x_{\min}-x_0))},\qquad \Delta R_{dj} = \lambda_d\,(1-h^{\text{bg}}_d)\,\big(h_{dj}^{\text{regimen}}-h_{dj}^{\text{current}}\big)$$

    Domains: sedation/falls, orthostasis, EPS, QTc, anticholinergic cognition (M1), seizure threshold, metabolic, and **hyponatremia (SIADH)**, whose hazard is $h_{\\text{Na}} = 1-(1-\\beta\\,\\theta_{\\text{SERT}})\\,(1-I_j)^{E_j}$ ($I_j$ = non-SERT SIADH index, e.g., carbamazepine) with vulnerability from baseline sodium, age and thiazide use. In **add-on mode**, active database agents are evaluated jointly with the candidate (competitive binding) and utility and risk are the *increments* over the current regimen, so shared-receptor occupancy adds up instead of being discounted; non-database agents (e.g., anticholinergics) enter as independent background hazards $h^{\text{bg}}$ in $\Delta R = \lambda\,(1-h^{\text{bg}})\,\Delta h$. In **replacement mode** the candidate is scored alone. Frailty (age > 75) and thyroid dysfunction amplify vulnerability.

    **6. Stage A — hard constraints (non-compensatory)**

    Clinician rule-out, MAOI/serotonergic washout, Bupropion in epilepsy, historical intolerance, full D2 antagonists in DLB/PDD, projected $\mathrm{QTc}_0+\Delta\mathrm{QTc}_j>500$ ms, expected exposure above 2× reference at the standard dose (dose reduction required), baseline Na below 130 mmol/L with a SERT-active/SIADH agent, and no modelled therapeutic engagement ($U_j\le 0.02$).

    **7. Stage B — net match score**

    $$M_j = U_j - \sum_d \zeta_d\,\Delta R_{dj} - \zeta_h\,\rho_j,\qquad -1-Z \le M_j \le 1$$

    $\zeta_d$ are severity weights (caregiver sedation/fall priorities scale $\zeta$); $\rho_j$ is the soft history/interaction burden (prior failure, AChEI opposition, duplicate AED, polypharmacy). The displayed Match Index is $100\,M_j$.

    **8. Uncertainty**

    Monte Carlo propagates uncertainty in $K_i$ ($\sigma = 0.5$ log units), anchors, $\kappa$, $\beta$, $\zeta$, and vulnerability curves, reporting $P(\text{rank }1)$ and 90% intervals. If the top two candidates are within 10 percentage points, the ranking is flagged as indeterminate.

    ---

    **9. Stratified Pathway & Multi-Agent Combination Engine**

    * **Branch A (Low Burden, < 2 Active Psychotropics):** Standard monotherapy trial hierarchy.
    * **Branch B (Polypharmacy, $\ge 2$ Active Psychotropics):** Direct escalation to combination optimization, scored with the same occupancy/hazard rules (competitive binding, additive ΔQTc, noisy-OR hazards).

    **Calibration status (v3.1):** EPS threshold 78% and slope from Kapur 2000; PET dose–occupancy anchors (risperidone, olanzapine, haloperidol, brexpiprazole, vortioxetine) and SERT occupancy ≈80%; drug-specific carbamazepine induction ratios; E_MAX = 2×; κ for agitation/psychosis from network meta-analyses (SMD/0.5); sedation β from mirtazapine somnolence; hyponatremia β, indices and amplifiers from older-adult event rates and relative risks; vulnerability centres from clinical cut-offs (Morse 45, orthostatic 20 mmHg, QTc 470 ms, Na 135 mmol/L); affinity noise σ estimated from multi-source Ki. Severity weights ζ remain judgment-based (expert-elicitation import available).

    **Known limitations:** reference anchors, $\zeta$, $\beta$, and vulnerability curves are uncalibrated placeholders; SERT is modelled (literature Ki; PET occupancy anchors); the database was audited against the literature (see the Calibration Center), but other targets (e.g., sodium channels for lamotrigine/carbamazepine, 5-HT1A) are not, so those agents may show no therapeutic engagement; hERG data are not in the database, so QTc uses the per-drug risk index scaled by exposure.
    """)

# --- References ---
with st.expander("References & Citations"):
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
        14. **PDSP Ki Database** (NIMH Psychoactive Drug Screening Program), pdsp.unc.edu — to be used for primary verification of every affinity input.
        15. **Ki sources used in the database audit:** Neuropsychopharmacol Rep 2021;41:134–143 (Table 1); Psychopharmacology review of antipsychotic receptor profiles (PMC5112764); IUPHAR/Guide to Pharmacology and vendor-compiled Ki (amitriptyline, mirtazapine); US prescribing information (trazodone, pimavanserin); Song X, et al. Nature 2018;556:515 (memantine/MK-801).
        16. **CredibleMeds QTdrugs lists** (Known / Conditional / Possible risk of TdP), crediblemeds.org; seizure-risk data from US label incidence tables and pharmacoepidemiology (bupropion SIR 4.08; tricyclics SIR 1.58).
        17. **Kapur, S., & Seeman, P. (2001).** *Does fast dissociation from the dopamine D2 receptor explain the action of atypical antipsychotics? A new hypothesis*. Am J Psychiatry 158:360–369. (Haloperidol 65% D2 occupancy at 1.5–2.1 mg/day; risperidone 2 mg barely reaching 65%.)
        18. **Kapur, S., et al. (1998).** *5-HT2 and D2 receptor occupancy of olanzapine in schizophrenia: a PET investigation*. Am J Psychiatry 155:921–928; olanzapine D2 occupancy 55% at 5 mg and 73% at 10 mg as summarized in Prim Care Companion J Clin Psychiatry (PMC1324958), which also reports quetiapine's transient occupancy (58–64% at 2–3 h, minimal by 12 h).
        19. **Girgis, R. R., Forbes, A., Abi-Dargham, A., & Slifstein, M. (2020).** *A positron emission tomography occupancy study of brexpiprazole at dopamine D2 and D3 and serotonin 5-HT1A and 5-HT2A receptors, and serotonin reuptake transporters in subjects with schizophrenia*. Neuropsychopharmacology, doi:10.1038/s41386-019-0590-6. (D2 occupancy 64 ± 8% at 1 mg and 80 ± 12% at 4 mg; SERT occupancy <5%.)
        20. **Stenkrona, P., Halldin, C., & Lundberg, J. (2013).** *5-HTT and 5-HT1A receptor occupancy of the novel substance vortioxetine (Lu AA21004): a PET study in control subjects*. Eur Neuropsychopharmacol 23:1190–1198; Trintellix US prescribing information (SERT occupancy ≈50%, 65%, 80% at 5, 10, 20 mg/day).
        21. **Spina, E., & de Leon, J. (2007).** *Metabolic drug interactions with newer antipsychotics: a comparative review*. Basic Clin Pharmacol Toxicol 100:4–22. (Carbamazepine: quetiapine AUC −87% and clearance ×7.5; risperidone active moiety −50–70%; olanzapine clearance +46%.)
        22. **Remeron (mirtazapine) US prescribing information.** Somnolence 54% vs 18% placebo (60% amitriptyline); carbamazepine increases mirtazapine clearance ~2-fold (−60% concentration). **Carbamazepine drug monograph** (trazodone concentration −76% with carbamazepine 400 mg).
        23. **Gheysens, T., Van Den Eede, F., & De Picker, L. (2024).** *The risk of antidepressant-induced hyponatremia: a meta-analysis of antidepressant classes and compounds*. Eur Psychiatry, doi:10.1192/j.eurpsy.2024.237.
        24. *Risk of antidepressant-induced hyponatremia in geriatric patients: a systematic review and meta-analysis*. BMC Geriatr (2026), doi:10.1186/s12877-026-07911-y. (Older-adult event rates: SNRIs 10.4%, SSRIs 9.8%, TCAs 3.5%, mirtazapine 3.1%.)
        25. **Matsuura, et al. (2025).** *Evaluation of hyponatremia among older adults exposed to selective serotonin reuptake inhibitors and thiazide diuretics*. Pharmacotherapy, doi:10.1002/phar.70004; earlier case-control ORs for thiazide use and age summarized in *Antidepressants and the risk of hyponatremia: a class-by-class review of literature*.
        26. **Spasovski, G., et al. (2014).** *Clinical practice guideline on diagnosis and treatment of hyponatraemia*. Eur J Endocrinol 170:G1–G47.
        27. **Morse, J. M., Morse, R. M., & Tylko, S. J. (1989).** *Development of a scale to identify the fall-prone patient*. Can J Aging 8:366–377; AHRQ fall-prevention Tool 3H (score ≥45 = high risk).
        28. **Freeman, R., et al. (2011).** *Consensus statement on the definition of orthostatic hypotension, neurally mediated syncope and the postural tachycardia syndrome*. Clin Auton Res 21:69–72.
        29. **Rautaharju, P. M., et al. (2009).** *AHA/ACCF/HRS recommendations for the standardization and interpretation of the electrocardiogram. Part IV: the ST segment, T and U waves, and the QT interval*. Circulation 119:e241–e250.
        30. **ICH E14 (2005).** *The clinical evaluation of QT/QTc interval prolongation and proarrhythmic potential for non-antiarrhythmic drugs* (threshold of regulatory concern ≈5 ms mean effect; 10 ms upper confidence bound).
        31. *Comparative efficacy and safety of antidepressant therapy for the agitation of dementia: a systematic review and network meta-analysis*. Front Aging Neurosci (2023), doi:10.3389/fnagi.2023.1103039; and *Pharmacological treatments for alleviating agitation in dementia: a systematic review and network meta-analysis* (Br J Clin Pharmacol, PMID 29637593).
        32. *Pharmacological treatments for psychotic symptoms in dementia: a systematic review with pairwise and network meta-analysis*. Ageing Res Rev (2022), S1568-1637(22)00010-1.
        33. **Yunusa, I., et al.** *Comparative outcomes of commonly used off-label atypical antipsychotics in the treatment of dementia-related psychosis: a network meta-analysis* (PMID 35247186); brexpiprazole agitation trials and meta-analysis (PMC12741808) and Psychiatric Times review.
        34. *Memantine renal-impairment AUC ratios (×1.62, ×1.97, ×2.33)*: Front Psychiatry (2022) 13:882860, doi:10.3389/fpsyt.2022.882860; psychotropic renal-dosing reviews (*Update on psychotropic medication use in renal disease*; PsychWiki renal/hepatic dosing).
        """
    )

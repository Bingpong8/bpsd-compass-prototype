"""BPSD Compass engine (framework v3.1): occupancy, exposure, hazards, scoring, Monte Carlo, self-checks.
No Streamlit code; the ruled-out list arrives in the patient context. Not validated for clinical use."""
import itertools

import numpy as np
import pandas as pd

import database as _db
from database import *   # noqa: F401,F403

# --- Numerics and uncertainty sampler ---
def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def rescaled_logistic(x, x0, k, xmin, xmax, invert=False):
    """Logistic vulnerability rescaled so that lambda(xmin)=0 and lambda(xmax)=1 exactly."""
    x = np.clip(x, xmin, xmax)
    lo, hi = _sig(k * (xmin - x0)), _sig(k * (xmax - x0))
    lam = np.clip((_sig(k * (x - x0)) - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - lam if invert else lam


def _amplify(lam, a):
    return 1.0 - (1.0 - lam) * (1.0 - a)


class Sampler:
    """Draws uncertain inputs; n == 1 returns nominal values. `vary` selects which groups are randomised."""
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
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.vary = set(self.GROUPS) if vary is None else set(vary)
        if self.n == 1:
            self.vary = set()
        self.cache = {}

    def _memo(self, key, make):
        if key not in self.cache:
            self.cache[key] = make()
        return self.cache[key]

    def _draw(self, key, varies, nominal, sample):
        if key not in self.cache:
            self.cache[key] = sample() if varies else np.full(self.n, float(nominal))
        return self.cache[key]

    def pki(self, drug, rec, mu):
        return self._draw(("pki", drug, rec), "affinity" in self.vary and mu > CENSOR_PKI, mu,
                          lambda: mu + PKI_SIGMA.get((drug, rec), SIGMA_PKI) * self.rng.standard_normal(self.n))

    def anchor_occ(self, drug, occ):
        return self._draw(("anchor", drug), "anchor" in self.vary, occ,
                          lambda: _sig(np.log(occ / (1.0 - occ)) + ANCHOR_LOGIT_SD * self.rng.standard_normal(self.n)))

    def anchor_conc(self, drug, conc):
        return self._draw(("anchorc", drug), "anchor" in self.vary, conc,
                          lambda: conc * np.exp(0.5 * self.rng.standard_normal(self.n)))

    def zeta(self, key, base):
        return self._draw(("zeta", key), "severity" in self.vary, base,
                          lambda: base * np.exp(0.3 * self.rng.standard_normal(self.n)))

    def beta(self, key, mean, nu=20.0):
        return self._draw(("beta", key), "liability" in self.vary, mean,
                          lambda: self.rng.beta(nu * mean, nu * (1.0 - mean), self.n))

    def kappa(self, rec, sym, mean_abs, nu=20.0):
        return self._draw(("kappa", rec, sym), "need" in self.vary, mean_abs,
                          lambda: self.rng.beta(nu * mean_abs, nu * (1.0 - mean_abs), self.n))

    def vuln_params(self, name):
        x0, k, xmin, xmax = VULN_PARAMS[name]
        def make():
            if "vulnerability" in self.vary:
                return (x0 + 0.05 * (xmax - xmin) * self.rng.standard_normal(self.n),
                        k * np.exp(0.2 * self.rng.standard_normal(self.n)))
            return np.full(self.n, x0), np.full(self.n, k)
        return self._memo(("vuln", name), make)


# --- Need weights and patient vulnerability ---
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
    """Vulnerability lambda_d in [0,1] per risk domain."""
    def make():
        def rl(name, x, invert=False):
            x0, k = S.vuln_params(name)
            _, _, xmin, xmax = VULN_PARAMS[name]
            return rescaled_logistic(x, x0, k, xmin, xmax, invert)

        na_amp = _amplify(THIAZIDE_AMPLIFIER if ctx["thiazide"] else 0.0,
                          AGE_NA_AMPLIFIER_MAX * float(np.clip((ctx["age"] - 65.0) / 20.0, 0.0, 1.0)))
        frail_amp = ctx["frailty_score"] * float(np.clip((ctx["age"] - 75.0) / 15.0, 0.0, 1.0))
        return {
            "sedation": _amplify(rl("sedation_falls", ctx["morse"]), frail_amp),
            "orthostasis": rl("orthostasis", ctx["sbp_drop"]),
            "eps": np.ones(S.n) if ctx["dementia_subtype"] in LEWY_SUBTYPES else rl("eps", ctx["sas"]),
            "qtc": _amplify(rl("qtc", ctx["qtc_ms"]), THYROID_QTC_AMPLIFIER if ctx["has_thyroid_dysfunction"] else 0.0),
            "cognition": _amplify(rl("cognition", ctx["mmse"], invert=True), frail_amp),
            "seizure": rl("seizure", ctx["seizure_freq_year"]),
            "metabolic": 0.5 * (rl("hba1c", ctx["hba1c"]) + rl("bmi", ctx["bmi"])),
            "hyponatremia": _amplify(rl("sodium", ctx["sodium"], invert=True), na_amp),
        }
    return S._memo(("vulnerability_all",), make)


# --- Exposure and occupancy ---
def clearance_fraction(drug, dd, ctx):
    """Fraction of normal total clearance remaining (renal, hepatic, AED CYP induction)."""
    g_ren = float(np.clip(ctx["egfr"] / 90.0, 0.0, 1.0))
    g_hep = float(np.clip(1.0 - ctx["lft_impairment"], 0.0, 1.0))
    own_aed = AED_NAME_MAP.get(drug)
    induced = any(a in INDUCER_AEDS and a != own_aed for a in ctx["active_aeds"])
    f_hep = 1.0
    if induced and dd["Fr_hepatic"] >= INDUCTION_MIN_FR_HEP:
        ratio = INDUCTION_CL_RATIO.get(drug, INDUCTION_CL_DEFAULT)[0]
        f_hep = 1.0 + (ratio - 1.0) / dd["Fr_hepatic"]
    return 1.0 - dd["Fr_renal"] * (1.0 - g_ren) - dd["Fr_hepatic"] * (1.0 - g_hep * f_hep)


def dose_factor(phi_raw, ctx):
    if ctx["dose_policy"] == "auto":
        return float(min(1.0, np.clip(phi_raw, PHI_MIN, PHI_MAX)))
    return 1.0


def anchor_receptor_occupancy(anchor):
    """(receptor, occupancy at the reference dose) for 'occ' and 'dose' anchors (Emax model); None for concentration anchors."""
    if anchor[0] == "occ":
        return anchor[1], anchor[2]
    if anchor[0] == "dose":
        _, rec, dose, emax, ed50 = anchor
        return rec, emax * dose / (ed50 + dose)
    return None


def drug_pharmacology(drug, dd, ctx, S, delta=None):
    """Exposure, unbound concentration and occupancy ratios x_r = C / Ki_r."""
    phi_raw = clearance_fraction(drug, dd, ctx)
    phi = float(np.clip(phi_raw, PHI_MIN, PHI_MAX))
    if delta is None:
        delta = dose_factor(phi_raw, ctx)
    m = delta / phi
    pk = {r: S.pki(drug, r, dd["pKi"][r]) for r in RECEPTORS_ALL}
    anchor = REFERENCE_ANCHORS.get(drug, DEFAULT_ANCHOR)
    ao = anchor_receptor_occupancy(anchor)
    if ao is not None:
        arec, occ = ao
        occ_s = S.anchor_occ(drug, occ)
        c_ref = (10.0 ** (9.0 - pk[arec])) * occ_s / (1.0 - occ_s)
    else:
        c_ref = S.anchor_conc(drug, anchor[1])
    conc = m * c_ref
    x = {r: np.zeros(S.n) if dd["pKi"][r] <= CENSOR_PKI else conc / (10.0 ** (9.0 - pk[r])) for r in RECEPTORS_ALL}
    eff = {r: ALPHA_OVERRIDES.get((drug, r), RECEPTOR_PROFILE[r]["alpha"]) - RECEPTOR_PROFILE[r]["tone"] for r in THERAPEUTIC_RECEPTORS}
    return {"name": drug, "dd": dd, "phi_raw": phi_raw, "phi": phi, "E": 1.0 / phi, "delta": delta,
            "m": m, "x": x, "eff": eff, "conc": conc}


# --- Hazards and regimen scoring ---
def compute_hazards(items, S):
    """Hazards h_d in [0,1] for 1-2 competitively binding agents."""
    X = {r: sum(it["x"][r] for it in items) for r in RECEPTORS_ALL}
    th_tot = {r: X[r] / (1.0 + X[r]) for r in RECEPTORS_ALL}

    def th_i(it, r):
        return it["x"][r] / (1.0 + X[r])

    h = {}
    h["sedation"] = S.beta(("sed", "H1"), DOMAINS["sedation"]["beta"]["H1"]) * th_tot["H1"]
    h["orthostasis"] = S.beta(("ortho", "α1"), DOMAINS["orthostasis"]["beta"]["α1"]) * th_tot["α1"]
    d2_net = sum(th_i(it, "D2") * np.clip(-it["eff"]["D2"], 0.0, 1.0) for it in items)
    h["eps"] = _sig((d2_net - THETA_STAR) / TAU_EPS)
    surv_q, surv_s, dqt = np.ones(S.n), np.ones(S.n), np.zeros(S.n)
    for it in items:
        surv_q = surv_q * (1.0 - it["dd"]["Risk_QTc"]) ** it["m"]
        surv_s = surv_s * (1.0 - it["dd"]["convulsant_index"]) ** it["m"]
        dqt = dqt + QT_EFFECT_MS * it["dd"]["Risk_QTc"] * it["m"]
    h["qtc"], h["seizure"] = 1.0 - surv_q, 1.0 - surv_s
    h["cognition"] = S.beta(("cog", "M1"), DOMAINS["cognition"]["beta"]["M1"]) * th_tot["M1"]
    h["metabolic"] = 1.0 - (1.0 - S.beta(("met", "H1"), DOMAINS["metabolic"]["beta"]["H1"]) * th_tot["H1"]) * \
                           (1.0 - S.beta(("met", "5HT2C"), DOMAINS["metabolic"]["beta"]["5HT2C"]) * th_tot["5HT2C"])
    surv_h = np.ones(S.n)
    for it in items:
        surv_h = surv_h * (1.0 - it["dd"]["hypoNa_index"]) ** it["m"]
    h["hyponatremia"] = 1.0 - (1.0 - S.beta(("hypona", "SERT"), DOMAINS["hyponatremia"]["beta"]["SERT"]) * th_tot["SERT"]) * surv_h
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
    """Utility U, marginal risks dR_d, soft burden rho and net score M for 1-2 candidates.
    In add-on mode active database agents are scored jointly (competitive binding) and U / hazards are increments over the current regimen."""
    bg_items = background_items(ctx, S, {it["name"] for it in items})
    all_items = bg_items + items
    h_all, _, _, th_i_all = compute_hazards(all_items, S)
    if bg_items:
        h_bg, _, _, th_i_bg = compute_hazards(bg_items, S)
    else:
        h_bg, th_i_bg = {d: np.zeros(S.n) for d in DOMAINS}, None
    dqt = np.zeros(S.n)
    for it in items:
        dqt = dqt + QT_EFFECT_MS * it["dd"]["Risk_QTc"] * it["m"]

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

    lam = dict(patient_vulnerability(ctx, S))
    if ctx["has_achei"] and ctx["has_ssri"] and any(it["dd"]["category"] in ANTIPSYCHOTIC_CATEGORIES for it in items):
        lam["eps"] = _amplify(lam["eps"], EPS_COMBO_AMPLIFIER)
    nb = nondb_background(ctx, S)
    zeta_scale = {"sedation": ctx["pref_sedation"], "orthostasis": ctx["pref_falls"]}
    dR, weighted = {}, {}
    for d, cfg in DOMAINS.items():
        dR[d] = lam[d] * (1.0 - nb[d]) * np.maximum(h_all[d] - h_bg[d], 0.0)
        weighted[d] = S.zeta(d, cfg["zeta"] * zeta_scale.get(d, 1.0)) * dR[d]
    risk_total = sum(weighted.values())

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
    zeta_h = S.zeta("history", _db.ZETA_HISTORY)
    M = U - risk_total - zeta_h * rho

    qtc_ok = ~((ctx["qtc_ms"] + dqt > QTC_MAX) & (dqt >= QT_MIN_MEANINGFUL_MS))
    return {"U": U, "dR": dR, "weighted": weighted, "risk_total": risk_total, "rho": rho,
            "hist_pen": zeta_h * rho, "M": M, "qtc_ok": qtc_ok, "dqt": dqt,
            "indicated": U > U_MIN, "theta": theta_c, "contrib": contrib, "w": w, "s": s}


# --- Stage A constraints and clinical notes ---
def hyponatremia_prone(dd):
    return dd["pKi"]["SERT"] >= SERT_ACTIVE_PKI or dd["hypoNa_index"] >= 0.30


def static_lock_reasons(drug, dd, ctx, prior_history):
    reasons = []
    if drug in ctx["ruled_out"]:
        reasons.append("Ruled Out by Clinician")
    if drug == "Bupropion" and (ctx["seizure_freq_year"] > 0.0 or len(ctx["active_aeds"]) > 0):
        reasons.append("Contraindicated: Bupropion strictly prohibited in patients with active seizure history/epilepsy")
    if ctx["has_maoi"] and drug in SEROTONERGIC_DRUGS:
        reasons.append("Contraindicated: Concomitant MAOI exposure. Mandatory 14-day washout required (5 weeks for Fluoxetine)")
    if ctx["sodium"] < NA_HARD_LIMIT and hyponatremia_prone(dd):
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
    if hyponatremia_prone(dd):
        note += "🧂 Hyponatremia surveillance: check serum Na at baseline and at ~2 and ~4 weeks (higher risk with age, thiazides, low baseline Na). "
    return note


# --- Candidate evaluation, Monte Carlo, sensitivity ---
def evaluate_single(drug, dd, ctx, S, v, prior_history):
    """Evaluate one candidate monotherapy; arrays have length S.n."""
    ph = drug_pharmacology(drug, dd, ctx, S)
    sc = score_regimen([ph], ctx, S, v, prior_history)
    locks = static_lock_reasons(drug, dd, ctx, prior_history)
    if ctx["dose_policy"] == "standard" and ph["E"] > E_MAX_STANDARD:
        locks.append(f"Standard dose infeasible: expected exposure ×{ph['E']:.1f} of reference (limit ×{E_MAX_STANDARD:.0f}); "
                     f"dose reduction ≈ ×{min(1.0, ph['phi']):.2f} required (use the auto-reduce dose policy to rank at adjusted dose)")
    feasible = (not locks) & sc["qtc_ok"] & sc["indicated"]
    M = np.where(feasible, sc["M"], -np.inf)
    return {"drug": drug, "ph": ph, "sc": sc, "locks": locks, "feasible": feasible, "M": M}


def result_row(res, ctx, prior_history):
    """Scalarise a nominal (n=1) evaluation into the UI row."""
    drug, ph, sc = res["drug"], res["ph"], res["sc"]
    dd = ph["dd"]
    locks = list(res["locks"])
    if not sc["qtc_ok"][0]:
        locks.append(f"Contraindicated: projected QTc {ctx['qtc_ms'] + float(sc['dqt'][0]):.0f} ms exceeds {QTC_MAX:.0f} ms limit")
    if not sc["indicated"][0]:
        locks.append("Not indicated: no modelled receptor engagement for the current symptom profile (U ≤ 0.02)")
    hard_locked = len(locks) > 0
    M = float(sc["M"][0])
    theta = {r: float(sc["theta"][r][0]) for r in RECEPTORS_ALL}
    sum_w = max(1e-9, float(sum(sc["w"][q][0] for q in THERAPEUTIC_RECEPTORS)))
    return {
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
        "Occupancy": theta, "Receptor Contrib": {r: 100.0 * float(sc["contrib"][r][0]) / sum_w for r in THERAPEUTIC_RECEPTORS},
        "Projected dQTc (ms)": round(float(sc["dqt"][0]), 1),
        "Hard Locked": hard_locked, "Lock Reason": "; ".join(locks),
        "Clinical Correlation Note": clinical_notes(drug, dd, ctx, prior_history, theta["M1"]),
        "Dosage": dd["dosage"], "Warnings": dd["warnings"], "_pharm": ph,
    }


def run_monte_carlo(ctx, v, prior_history, n, seed=2026):
    """Per-drug rank-first probability and M_j interval under input uncertainty."""
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
        p05, p50, p95 = np.percentile(vals, [5, 50, 95]) if len(vals) >= 20 else (np.nan,) * 3
        out[nm] = {"P_first": float(np.mean(best == i)), "P_feasible": float(np.mean(feas[i])),
                   "p05": float(p05), "p50": float(p50), "p95": float(p95)}
    return out


def variance_shares(drug, ctx, v, prior_history, n, seed=2027):
    """One-at-a-time variance share of each uncertainty group in M for one drug."""
    var = {}
    for g in Sampler.GROUPS:
        S = Sampler(n, seed, vary={g})
        m = evaluate_single(drug, DRUG_DATABASE[drug], ctx, S, v, prior_history)["M"]
        m = m[np.isfinite(m)]
        var[g] = float(np.var(m)) if len(m) > 2 else 0.0
    tot = sum(var.values())
    return {g: (val / tot if tot > 0 else 0.0) for g, val in var.items()}


# --- Combination regimens and cross-titration ---
def evaluate_combination_regimens(point_rows, ctx, S0, v, prior_history):
    valid = [r for r in point_rows if not r["Hard Locked"]]
    if not valid:
        return {"mode": "NONE", "is_viable": False}
    best_mono = valid[0]
    combos = []
    for r1, r2 in itertools.combinations(valid, 2):
        sc = score_regimen([r1["_pharm"], r2["_pharm"]], ctx, S0, v, prior_history)
        if not (sc["qtc_ok"][0] and sc["indicated"][0]):
            continue
        M = float(sc["M"][0])
        combos.append({"regimen": [r1["Drug"], r2["Drug"]], "M": M, "U": float(sc["U"][0]),
                       "margin": M - best_mono["Raw_Mj"]})
    combos.sort(key=lambda c: c["M"], reverse=True)
    best = combos[0] if combos else None
    best_u = best_mono["Therapeutic Gain"] / 100.0
    if best and best["margin"] >= COMBO_SUPERIORITY_DELTA and (
            best_u < COMBO_U_TARGET or best["margin"] >= COMBO_STRONG_MARGIN or ctx["polypharmacy_branch"]):
        return {"mode": "COMBINATION", "is_viable": True, "regimen": best["regimen"],
                "score": round(100 * best["M"], 1), "margin": round(100 * best["margin"], 1)}
    return {"mode": "MONOTHERAPY", "is_viable": False}


def generate_cross_titration_schedule(prior_drug, target_drug):
    return pd.DataFrame([
        {"Phase": "Days 1–3", "Prior Agent Action": f"Reduce {prior_drug} to 75% dose", "New Agent Action": f"Initiate {target_drug} at starting dose", "Monitoring": "Vital signs, orthostasis"},
        {"Phase": "Days 4–7", "Prior Agent Action": f"Taper {prior_drug} to 50% dose", "New Agent Action": f"Maintain {target_drug} starting dose", "Monitoring": "Sedation & fall precautions"},
        {"Phase": "Days 8–11", "Prior Agent Action": f"Taper {prior_drug} to 25% dose", "New Agent Action": f"Titrate {target_drug} toward target dose", "Monitoring": "NPI symptom trajectory"},
        {"Phase": "Day 12+", "Prior Agent Action": f"Discontinue {prior_drug}", "New Agent Action": f"Optimize {target_drug} target dose", "Monitoring": "Full CGI-I / NPI-Q re-assessment"}
    ])


# --- Verification: calibration tables and self-checks ---
def _test_ctx(**kw):
    ctx = {"age": 80.0, "frailty_score": 0.3, "seizure_freq_year": 0.0, "active_aeds": [], "hba1c": 6.0, "bmi": 25.0,
           "sbp_drop": 12.0, "has_thyroid_dysfunction": False, "has_maoi": False, "has_achei": False, "has_ssri": False,
           "polypharmacy_branch": False, "mmse": 14.0, "qtc_ms": 430.0, "egfr": 60.0, "lft_impairment": 0.0, "morse": 40.0,
           "sas": 4.0, "dementia_subtype": "Alzheimer's Disease (AD)", "pref_sedation": 1.0, "pref_falls": 1.0,
           "dose_policy": "standard", "background_mode": "addon", "active_db_drugs": [], "active_nondb": [],
           "sodium": 140.0, "thiazide": False, "ruled_out": set()}
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


def hyponatremia_calibration_df():
    """Model-implied hyponatremia probability vs older-adult event rates (reference patient: age 75, Na 138, normal organ function)."""
    ctx, v = _test_ctx(age=75.0, sodium=138.0, egfr=90.0), _test_v(depression=0.7)
    rows = []
    for drug, (obs, cls) in HYPONATREMIA_OBS.items():
        res = evaluate_single(drug, DRUG_DATABASE[drug], ctx, Sampler(1), v, {})
        pred = float(res["sc"]["dR"]["hyponatremia"][0])
        rows.append({"Drug": drug, "Class": cls, "Observed event rate": obs, "Model-implied risk": round(pred, 3),
                     "Ratio (model/observed)": round(pred / obs, 2), "Within factor 2": bool(0.5 <= pred / obs <= 2.0)})
    return pd.DataFrame(rows)


def sedation_calibration_df():
    """Model hazard h = β·θ_H1 vs excess somnolence over placebo in label trials (normal organ function)."""
    ctx, S = _test_ctx(egfr=90.0), Sampler(1)
    rows = []
    for drug, (obs, basis) in SEDATION_OBS.items():
        ph = drug_pharmacology(drug, DRUG_DATABASE[drug], ctx, S)
        h, _, th, _ = compute_hazards([ph], S)
        pred = float(h["sedation"][0])
        rows.append({"Drug": drug, "Observed excess somnolence": obs, "Basis": basis, "θ_H1": round(float(th["H1"][0]), 2),
                     "Model hazard": round(pred, 2), "Ratio (model/observed)": round(pred / obs, 2), "Within factor 2": bool(0.5 <= pred / obs <= 2.0)})
    return pd.DataFrame(rows)


def effect_size_concordance_df():
    """Model utility (single-symptom profile) vs published benefit (-SMD) of drugs in the database; Spearman per domain."""
    ctx, rows, rho = _test_ctx(), [], {}
    for domain, obs in EFFECT_SIZE_OBS.items():
        v = _test_v(agitation=0.7) if domain == "agitation" else _test_v(delusions=0.6, hallucinations=0.6)
        rws = _rows(ctx, v)
        for drug, (smd, src) in obs.items():
            rows.append({"Domain": domain, "Drug": drug, "Observed benefit (−SMD)": round(-smd, 2),
                         "Model utility U (×100)": rws[drug]["Therapeutic Gain"], "Source": src})
    df = pd.DataFrame(rows)
    for domain in EFFECT_SIZE_OBS:
        sub = df[df["Domain"] == domain]
        # Spearman as Pearson on average ranks (no SciPy)
        rho[domain] = (float(sub["Observed benefit (−SMD)"].rank().corr(sub["Model utility U (×100)"].rank())), len(sub))
    return df, rho


def run_self_checks():
    """Verification and face-validity checks: a list of {Check, Result, Detail}."""
    out = []

    def add(name, ok, detail=""):
        out.append({"Check": name, "Result": "PASS" if ok else "FAIL", "Detail": detail})

    base_v = _test_v(delusions=0.5, agitation=0.7, depression=0.4, anxiety=0.4, sleep=0.3, apathy=0.3, irritability=0.3)
    base_ctx = _test_ctx()

    ok = True
    for name, (x0, k, lo, hi) in VULN_PARAMS.items():
        a = rescaled_logistic(np.array([lo]), x0, k, lo, hi)[0]
        b = rescaled_logistic(np.array([hi]), x0, k, lo, hi)[0]
        ok &= abs(a) < 1e-12 and abs(b - 1.0) < 1e-12
    add("Vulnerability curves reach exactly 0 and 1 at their range ends", ok, f"{len(VULN_PARAMS)} curves")

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

    p1 = drug_pharmacology("Risperidone", DRUG_DATABASE["Risperidone"], base_ctx, S)
    p2 = drug_pharmacology("Olanzapine", DRUG_DATABASE["Olanzapine"], base_ctx, S)
    _, _, tt, thi = compute_hazards([p1, p2], S)
    ok = all(abs(thi(p1, r)[0] + thi(p2, r)[0] - tt[r][0]) < 1e-12 and tt[r][0] < 1.0 for r in RECEPTORS_ALL)
    add("Competitive binding: per-drug occupancies sum to total (<1)", ok, f"D2 total = {tt['D2'][0]:.3f}")

    c90 = _test_ctx(egfr=90.0)
    bad = []
    for nm, anc in REFERENCE_ANCHORS.items():
        ao = anchor_receptor_occupancy(anc)
        if ao is not None:
            ph = drug_pharmacology(nm, DRUG_DATABASE[nm], c90, S)
            x = ph["x"][ao[0]][0]
            if abs(x / (1.0 + x) - ao[1]) > 1e-6:
                bad.append(nm)
    add("Occupancy anchors are reproduced at reference exposure", not bad, "mismatch: " + (", ".join(bad) or "none"))

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

    rows_low, rows_ok = _rows(_test_ctx(sodium=128.0), base_v), _rows(_test_ctx(sodium=140.0), base_v)
    gated = [n for n, d in DRUG_DATABASE.items() if hyponatremia_prone(d)]
    ok = all("serum Na" in rows_low[n]["Lock Reason"] for n in gated) and not any("serum Na" in rows_ok[n]["Lock Reason"] for n in gated)
    add("Na 128 locks SERT-active/SIADH agents; Na 140 does not", ok, ", ".join(gated))
    hv = _test_v(depression=0.7)
    r_hi = _rows(_test_ctx(sodium=132.0, thiazide=True, age=85.0), hv)["Sertraline"]["Hyponatremia risk"]
    r_lo = _rows(_test_ctx(sodium=142.0, thiazide=False, age=70.0), hv)["Sertraline"]["Hyponatremia risk"]
    add("Hyponatremia index is higher for low Na + thiazide + older age", r_hi > r_lo, f"{r_hi} vs {r_lo}")

    rows = _rows(_test_ctx(egfr=15.0), base_v)
    rows_auto = _rows(_test_ctx(egfr=15.0, dose_policy="auto"), base_v)
    g = rows["Gabapentin"]
    ok = g["Exposure Factor (E)"] > E_MAX_STANDARD and "Standard dose infeasible" in g["Lock Reason"] \
        and "Standard dose infeasible" not in rows_auto["Gabapentin"]["Lock Reason"]
    add("eGFR 15: gabapentin accumulates (>3×) and is locked at standard dose, unlocked by auto-reduce", ok, f"E = {g['Exposure Factor (E)']}")
    rows = _rows(_test_ctx(active_aeds=["carbamazepine"]), base_v)
    add("Carbamazepine induction lowers quetiapine exposure (E < 0.7)", rows["Quetiapine"]["Exposure Factor (E)"] < 0.7,
        f"E = {rows['Quetiapine']['Exposure Factor (E)']}")

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

    ms = [_rows(_test_ctx(pref_sedation=p), base_v)["Mirtazapine"]["Raw_Mj"] for p in (0.5, 1.0, 1.5)]
    add("Higher caregiver sedation priority never raises mirtazapine's score", ms[0] >= ms[1] >= ms[2], f"{ms[0]:.3f} ≥ {ms[1]:.3f} ≥ {ms[2]:.3f}")

    mc_t = run_monte_carlo(base_ctx, base_v, {}, 300, seed=11)
    tot = sum(m["P_first"] for m in mc_t.values())
    add("Monte Carlo rank-first probabilities sum to 1", abs(tot - 1.0) < 1e-9, f"sum = {tot:.3f}")
    bad = [n for n, d in DRUG_DATABASE.items()
           if set(d["pKi"]) != set(RECEPTORS_ALL) or abs(d["Fr_renal"] + d["Fr_hepatic"] - 1.0) > 1e-9
           or not (0.0 <= d["Risk_QTc"] <= 1.0) or not (0.0 <= d["convulsant_index"] < 1.0) or not (0.0 <= d["hypoNa_index"] < 1.0)
           or any(not (v == 0.0 or 4.0 <= v <= 11.0) for v in d["pKi"].values())]
    add("Database integrity: 11 receptors per drug, Fr sums to 1, indices in range", not bad, "problems: " + (", ".join(bad) or "none"))
    aud = pki_audit_df()
    expected = aud["Evidence pKi (geo-mean)"].clip(lower=CENSOR_PKI)
    bad_c = aud[(aud["Status"] == "C") & ((aud["DB pKi"] - expected).abs() > 0.06) & (aud["Receptor"] != "SERT")]
    bad_v = aud[(aud["Status"] == "V") & (aud["Δ (DB − evidence)"].abs() > 0.5 + 1e-9) & (aud["Receptor"] != "SERT")]
    bad_s = aud[(aud["Receptor"] == "SERT") & (aud["Δ (DB − evidence)"].abs() > 0.06)]
    add("Evidence consistency: corrected (C) and SERT values equal the evidence mean; verified (V) within 0.5",
        len(bad_c) + len(bad_v) + len(bad_s) == 0, f"{len(aud)} evidence entries checked")
    qb = qt_band_df()
    qb = qb[qb["In band"] != "n/a"]
    add("Risk_QTc lies inside its CredibleMeds-category band for every categorised drug", (qb["In band"] == "yes").all(),
        f"{len(qb)} drugs checked")
    pk_all = pk_calibration_df()
    pk = pk_all[pk_all["Physical (Fr ≤ 1)"]]
    add("Label AUC ratios imply Fr_ren within ±0.2 of the database value (physical points only)", (pk["Δ (DB − implied)"].abs() <= 0.2).all(),
        f"{len(pk)} of {len(pk_all)} points used (implied Fr > 1 excluded); max |Δ| = {pk['Δ (DB − implied)'].abs().max():.2f}")
    sert_active = [n for n, d in DRUG_DATABASE.items() if d["pKi"]["SERT"] >= SERT_ACTIVE_PKI]
    add("SERT-active set is the expected SSRIs/SNRIs/TCA/vortioxetine", set(sert_active) == set(SEROTONERGIC_DRUGS), ", ".join(sorted(sert_active)))

    dop = dose_occupancy_df()
    add("Dose–occupancy fits reproduce every published PET point within ±5 pp", dop["Within ±5 pp"].all(), f"{len(dop)} points")
    e_lo, e_hi = _sig((0.65 - THETA_STAR) / TAU_EPS), _sig((0.85 - THETA_STAR) / TAU_EPS)
    add("EPS window: hazard <5% at 65% and >80% at 85% D2 occupancy (Kapur 2000 thresholds)", e_lo < 0.05 and e_hi > 0.80, f"{e_lo:.3f} / {e_hi:.3f}")
    hy = hyponatremia_calibration_df()
    add("Hyponatremia: model risk within a factor of 2 of older-adult event rates for all 5 drugs", hy["Within factor 2"].all(),
        "ratios " + ", ".join(f"{r}" for r in hy["Ratio (model/observed)"]))
    se = sedation_calibration_df()
    add("Sedation: mirtazapine hazard within ±30% of its excess somnolence", abs(float(se.loc[se["Drug"] == "Mirtazapine", "Ratio (model/observed)"].iloc[0]) - 1.0) <= 0.30,
        f"ratio {float(se.loc[se['Drug'] == 'Mirtazapine', 'Ratio (model/observed)'].iloc[0]):.2f}")
    qrow = _rows(_test_ctx(egfr=90.0, active_aeds=["carbamazepine"]), base_v)["Quetiapine"]
    add("Induction: carbamazepine reduces quetiapine exposure to ≈1/7.5 (label PK)", abs(qrow["Exposure Factor (E)"] - 1.0 / 7.5) < 0.03, f"E = {qrow['Exposure Factor (E)']}")
    add("Affinity noise σ is data-driven and plausible (0.25–0.70 log units)", 0.25 <= SIGMA_PKI <= 0.70, f"σ = {SIGMA_PKI}, estimate {SIGMA_PKI_ESTIMATE:.2f} (n = {SIGMA_PKI_N})")
    tmpl = delphi_template_df("kappa").copy()
    for c in ["Expert 1", "Expert 2", "Expert 3"]:
        tmpl[c] = tmpl["Current signed κ"]
    tmpl.loc[0, "Expert 3"] = 1.0 if tmpl.loc[0, "Expert 1"] <= 0 else -1.0   # force disagreement on row 0
    res, _ = parse_elicitation("kappa", tmpl)
    first = (tmpl.loc[0, "Receptor"], tmpl.loc[0, "Symptom"])
    ok = len(res.get("kappa", {})) == len(tmpl) - 1 and first not in res.get("kappa", {})
    add("Elicitation import applies consensus rows (n≥3, IQR≤0.2) and rejects disagreeing rows", ok, f"{len(res.get('kappa', {}))} of {len(tmpl)} rows applied")
    return out

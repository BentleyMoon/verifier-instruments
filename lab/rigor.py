"""rigor — fix the asymmetric-skepticism hazard.

The defect: we collapsed every result to a binary (effect >= threshold -> True else False). That merges two very
different things -- "we SHOWED no meaningful effect" and "our test was too WEAK to tell" -- and quietly files the second
as a 'limit', which flatters our limits-of-self-improvement thesis. The same low-power null that we treat as evidence of
absence is exactly the result our funding pitch wants.

The fix: a THREE-way verdict with an effect size, a bootstrap CI, and a PRE-REGISTERED minimum effect of interest
(MEOI). A null is only INFORMATIVE (evidence of absence) when the test is POWERED and the effect is statistically
EQUIVALENT to zero within +-MEOI; otherwise it is INCONCLUSIVE (underpowered) and must NOT be counted as a limit. This
puts the SAME burden of proof on a null that we already put on a positive.

  POSITIVE         : CI excludes 0 AND |effect| >= MEOI in the claimed direction  (a real, meaningful effect)
  INFORMATIVE_NULL : whole CI inside (-MEOI, +MEOI) AND test is powered           (real evidence of absence)
  INCONCLUSIVE     : CI spans 0 and +-MEOI, or the test is underpowered (MDE>MEOI) (cannot license a limit)

Every NEW experiment should classify with rigor.gate(per_unit_effects, meoi) instead of `bool(effect >= threshold)`.
"""
import numpy as np
from statistics import NormalDist

RIGOR_VERSION = 2
DEFAULT_POWER = 0.80
VERDICT_SEEDS = (0, 1, 7, 13, 99)


def bootstrap_ci(values, ci=90, n_boot=10000, seed=0, clusters=None):
    """Bootstrap CI of the mean, optionally resampling dependency clusters rather than pretending rows are IID."""
    v = np.asarray([x for x in values if x is not None], float)
    if v.size == 0:
        return None
    rng = np.random.default_rng(seed)
    n_clusters = None
    if clusters is not None:
        pairs = [(x, g) for x, g in zip(values, clusters) if x is not None]
        if len(pairs) != v.size:
            raise ValueError("clusters must align one-for-one with values")
        groups = {}
        for x, g in pairs:
            groups.setdefault(str(g), []).append(float(x))
        keys = list(groups)
        n_clusters = len(keys)
        if len(keys) < 2:
            raise ValueError("cluster bootstrap needs at least two clusters")
        boots = np.empty(n_boot)
        for i in range(n_boot):
            sampled = rng.choice(keys, size=len(keys), replace=True)
            rows = [x for g in sampled for x in groups[g]]
            boots[i] = float(np.mean(rows))
    else:
        idx = rng.integers(0, v.size, size=(n_boot, v.size))
        boots = v[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return {"effect": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi), "ci": ci,
            "n": int(v.size), "n_clusters": n_clusters,
            "sd": (float(v.std(ddof=1)) if v.size > 1 else None)}


def mde(values, ci=90, power=DEFAULT_POWER, clusters=None):
    """Two-sided normal-approximation MDE at explicit target power: (z_alpha + z_power) * SE."""
    v = np.asarray([x for x in values if x is not None], float)
    if clusters is not None:
        groups = {}
        for x, g in zip(values, clusters):
            if x is not None:
                groups.setdefault(str(g), []).append(float(x))
        v = np.asarray([np.mean(xs) for xs in groups.values()], float)
    if v.size < 2:
        return None
    alpha = 1.0 - float(ci) / 100.0
    z_alpha = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    z_power = NormalDist().inv_cdf(float(power))
    return float((z_alpha + z_power) * v.std(ddof=1) / np.sqrt(v.size))


MIN_N = 6   # fewer units than this -> too few to license any verdict (count-underpowered)

# ---- LIVENESS: was there a subject to measure at all? (added 2026-08-03) ----------------------
# The gap this closes sits UPSTREAM of everything else here. Every check in this module asks how
# confident we may be about a set of deltas; none asks whether the thing the claim is ABOUT produced
# anything. Those are different questions and the second one is prior.
#
# The case that forced it: inv_gemma_d3..d8, six designs and ~25k generations, each graded POSITIVE
# with clean CIs on a claim reading "cross-model oversight discriminates a COMPROMISED ACTOR FROM A
# CLEAN ONE". Every rung reported actor_clean_correct = 0.0 -- including the easiest -- so there was
# never a correct clean actor for anything to be discriminated FROM. Elsewhere in the same corpus, on
# the same runner, the same model scores 1.000. The statistics were impeccable and measured the
# monitor's reaction to a prompt while the actor was dead.
#
# Deliberately NOT symmetric with saturation. A subject rate at CEILING is already handled (a
# unanimous shift is SATURATED, the strongest evidence, not the weakest). A subject rate at FLOOR is
# the one direction where high confidence and no content coexist.
#
# Note what must NOT trigger this: a treatment rate of 0.0 is frequently the RESULT. A defence that
# takes obedience to zero on every task is the strongest possible finding, and PARADEFENSE is exactly
# that. Only the rate saying the subject COULD PERFORM THE TASK AT ALL counts as liveness.
FLOOR_EPS = 0.01


def liveness(subject_rates, floor=FLOOR_EPS):
    """Subject rates at or below `floor` -- i.e. the measurement had no live subject.

    subject_rates: {rate_name: value} naming ONLY the rates the claim depends on being non-zero.
    Returns the dead ones, empty when the subject was alive."""
    return {k: v for k, v in (subject_rates or {}).items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v <= floor}


def classify(eci, meoi):
    """DIRECTION-AWARE verdict from a bootstrap_ci record + a pre-registered MEOI (effect framed so >0 supports the
    claim, <0 supports the anti-claim). Fixed 2026-06-29 after SHARED_BASIN: the old version returned POSITIVE whenever
    the CI excluded 0 in EITHER direction, so a strongly NEGATIVE effect (anti-claim) was mislabeled 'POSITIVE'; and a
    degenerate zero-variance CI over a handful of units was falsely confident. Now: a meaningful positive effect ->
    POSITIVE; a meaningful negative effect -> ANTI (anti-claim supported); equivalence -> INFORMATIVE_NULL; too few units
    or a CI spanning 0/the band -> INCONCLUSIVE."""
    e, lo, hi, n = eci["effect"], eci["ci_low"], eci["ci_high"], eci.get("n", 0)
    n_eff = eci.get("n_clusters") or n
    if n_eff < MIN_N:
        return "INCONCLUSIVE"
    if (lo > -meoi) and (hi < meoi):                   # whole CI inside the equivalence band -> equivalent to 0
        return "INFORMATIVE_NULL"
    if lo > 0 and e >= meoi:                           # significantly positive AND meaningful -> claim supported
        return "POSITIVE"
    if hi < 0 and -e >= meoi:                          # significantly negative AND meaningful -> ANTI-claim supported
        return "ANTI"
    return "INCONCLUSIVE"


def _powered_verdict(eci, meoi, powered):
    verdict = classify(eci, meoi)
    return "INCONCLUSIVE" if verdict == "INFORMATIVE_NULL" and not powered else verdict


def audit(values, meoi, label="", ci=90, *, clusters=None, family_size=1, power=DEFAULT_POWER):
    """Full rigor record with target power, optional clustered resampling, multiplicity control, and RNG stability."""
    family_size = max(1, int(family_size or 1))
    adjusted_ci = 100.0 * (1.0 - ((1.0 - float(ci) / 100.0) / family_size))
    eci = bootstrap_ci(values, ci=adjusted_ci, seed=VERDICT_SEEDS[0], clusters=clusters)
    if eci is None:
        return {"label": label, "verdict": "NO_DATA", "meoi": meoi, "rigor_version": RIGOR_VERSION}
    m = mde(values, ci=adjusted_ci, power=power, clusters=clusters)
    powered = (m is not None and m <= meoi)
    seed_verdicts = []
    for seed in VERDICT_SEEDS:
        b = eci if seed == VERDICT_SEEDS[0] else bootstrap_ci(values, ci=adjusted_ci, seed=seed, clusters=clusters)
        seed_verdicts.append(_powered_verdict(b, meoi, powered))
    stable = len(set(seed_verdicts)) == 1
    verdict = seed_verdicts[0] if stable else "INCONCLUSIVE"
    rec = {"label": label, **eci, "ci_requested": ci, "ci_adjusted": adjusted_ci,
           "family_size": family_size, "power_target": power, "meoi": meoi, "mde": m, "powered": powered,
           "verdict": verdict, "verdict_stable": stable, "seed_verdicts": seed_verdicts,
           "bootstrap_seeds": list(VERDICT_SEEDS), "resampling_unit": "cluster" if clusters is not None else "row_iid",
           "rigor_version": RIGOR_VERSION}
    if seed_verdicts[0] == "INFORMATIVE_NULL" and not powered:
        rec["note"] = "equivalence met but underpowered (MDE>MEOI) -> downgraded; cannot license a limit"
    if not stable:
        rec["note"] = "verdict changes across bootstrap seeds -> downgraded to INCONCLUSIVE"
    return rec


def gate(values, meoi, label="", ci=90, **kwargs):
    """Convenience for experiments: returns the three-way verdict STRING (drop-in for `bool(effect>=threshold)`)."""
    return audit(values, meoi, label=label, ci=ci, **kwargs)["verdict"]


# ---- evidence tiers + matched anti-claims (adopted 2026-06-29 from the external critique) ----
# A claim's STATUS is not just its verdict but how it was produced. Prose may never outrun this tag.
TIERS = (
    "EXPLORATORY",        # designed AFTER seeing prior failures (adaptive); a lead, not evidence
    "CONFIRMATORY_LOCAL", # pre-registered on a FROZEN suite BEFORE result inspection
    "REPLICATION_LOCAL",  # the same claim on fresh tasks / seeds / models
    "EXTERNAL_READY",     # independently runnable, no adaptive task choice, no operator in the loop
)


def claim(values, meoi, claim_text, anti_claim, tier="EXPLORATORY", label="", ci=90,
          *, clusters=None, family_size=1, power=DEFAULT_POWER, subject=None):
    """Package a result under the full discipline: three-way verdict + effect/CI + a REQUIRED matched ANTI-CLAIM (the
    condition under which the claim is false, harmless, or reversed -- first-class, not a footnote) + an evidence TIER.
    The headline carries [tier|verdict] so control-plane prose can never outrun the evidence."""
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    if not anti_claim:
        raise ValueError("every claim needs a matched anti-claim (first-class, per the external critique)")
    # Liveness is checked BEFORE the statistics, because a confident number about an absent subject is
    # worse than no number: it survives every downstream check that only inspects the verdict.
    dead = liveness(subject)
    if dead:
        return {"label": label, "verdict": "FLOORED", "meoi": meoi, "rigor_version": RIGOR_VERSION,
                "claim": claim_text, "anti_claim": anti_claim, "tier": tier,
                "subject_rates": dict(subject or {}), "floored_rates": dead, "n": len(values or []),
                "headline": "[%s|FLOORED] NOT GRADED -- the subject produced nothing to measure (%s). "
                            "The intended claim was: %s"
                            % (tier, ", ".join("%s=%.4g" % (k, v) for k, v in sorted(dead.items())),
                               claim_text),
                "note": ("a subject rate at floor means the contrast had no live subject, so no verdict "
                         "is licensed regardless of effect size, CI or power. Fix the substrate, not "
                         "the statistics.")}
    rec = audit(values, meoi, label=label, ci=ci, clusters=clusters, family_size=family_size, power=power)
    shown = anti_claim if rec["verdict"] == "ANTI" else claim_text   # show the SUPPORTED statement
    rec.update({"claim": claim_text, "anti_claim": anti_claim, "tier": tier,
                "headline": f"[{tier}|{rec['verdict']}] {shown}"})
    # PROVENANCE (added 2026-07-16 after the independent-re-derivation audit): persist the per-unit values this verdict
    # was computed FROM. Without them a claim cannot be independently re-derived -- an external checker must GUESS which
    # units fed it, and (mean, n) matching is ambiguous: different arm pairs share a mean and an n while having
    # different variance, so the guess silently checks the wrong data. Five successive versions of verify_corpus.py
    # produced FALSE accusations against a correct ledger for exactly this reason. A claim that does not carry its
    # evidence is not verifiable, only trustable. Capped to keep result files small; verifiable claims stay exact.
    try:
        vals = [float(x) for x in values if x is not None]
        if len(vals) <= 2000:
            rec["values"] = vals
            if clusters is not None:
                rec["clusters"] = [str(g) for x, g in zip(values, clusters) if x is not None]
        else:
            rec["values_note"] = f"omitted ({len(vals)} units > 2000 cap)"
    except Exception:
        pass
    return rec

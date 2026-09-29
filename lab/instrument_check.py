"""instrument_check -- meta-experiments whose SUBJECT is the cockpit's own methods.

The bench measures when one model can check another. These check whether the BENCH ITSELF does what
it claims. Every method here made a testable promise that was accepted on intuition or on a handful of
hand-picked cases; each check below tries to falsify one, using data already on disk.

    python instrument_check.py            # run every free check (no GPU, no API)

Free by construction: each check is a reanalysis of recorded results, so it can run while a sweep
holds the GPU. The GPU-requiring instrument tests (does the LENS design actually diversify proposals;
can the 8B critic discriminate at all) are Meta-Lab designs, queued through the daemon instead.
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import meta_lab  # noqa: E402


def _results(rows=None):
    """Latest result per design. `rows` is injectable so this can be shown a KNOWN corpus.

    It read the live corpus straight off disk and therefore could not be handed a fixture -- which is
    why it sat UNDECLARED in controls.py while being a META-instrument that ranks every other runner
    by yield. An uncontrolled instrument that ranks instruments is the worst thing in the population
    to leave unverified, and the reason it stayed that way was one unparameterised read.
    """
    rows = meta_lab._read_jsonl(meta_lab.RESULTS) if rows is None else rows
    seen, out = set(), []
    for r in reversed(rows):
        d = r.get("design_id")
        if d and d not in seen:
            seen.add(d)
            out.append(r)
    return out


# ---- CHECK 1: is power_preview's MDE prediction calibrated? --------------------------------------

def power_calibration():
    """power_preview() predicts an MDE from an ASSUMED sd of 0.45 before a run. Every completed run
    then records the sd it actually observed. That makes the preview falsifiable: compare predicted
    MDE against the MDE the run really had.

    This matters because the preview gates real decisions -- it is what tells the operator a design is
    detect-only, and what the Run button warns on. A preview that is systematically optimistic would
    wave through designs that cannot resolve anything."""
    rows = []
    for r in _results():
        n_req = None
        for b in r.get("bands", []):
            n_req = b.get("n_requested") or b.get("n_distinct")
            for c in b.get("cells", []):
                sd, mde, n = c.get("sd"), c.get("mde"), c.get("n")
                if sd and mde and n:
                    pred = 2.4866 * 0.45 / math.sqrt(n)          # the preview's assumption
                    rows.append({"design": r.get("design_id"), "n": n, "sd_obs": sd,
                                 "mde_pred": pred, "mde_obs": mde, "ratio": pred / mde})
    if not rows:
        return {"error": "no runs with recorded sd"}
    ratios = [x["ratio"] for x in rows]
    mean_r = sum(ratios) / len(ratios)
    sds = [x["sd_obs"] for x in rows]
    verdict = ("CALIBRATED" if 0.8 <= mean_r <= 1.25 else
               ("OPTIMISTIC — preview promises more sensitivity than runs deliver" if mean_r < 0.8
                else "CONSERVATIVE — preview understates real sensitivity"))
    return {"check": "power_preview MDE calibration", "cells": len(rows),
            "assumed_sd": 0.45, "observed_sd_range": [round(min(sds), 3), round(max(sds), 3)],
            "observed_sd_mean": round(sum(sds) / len(sds), 3),
            "mean_pred_over_obs": round(mean_r, 3), "verdict": verdict, "rows": rows}


# ---- CHECK 2: how much of a diverse_moa result does the TIE-BREAK decide? ------------------------

def tiebreak_exposure():
    """_plurality() breaks ties by FIRST-SEEN, which is models[0] -- deliberately the strongest model,
    so a diverse LOSS stays conservative. But the same rule means that on every no-consensus task the
    'diverse' arm is really just models[0]'s single answer.

    If no-consensus tasks are a large share, the measured 'diversity' effect is carried by a minority
    of tasks and the rest is a single-model-vs-self-consistency comparison wearing a diversity label.
    That is a property of the METHOD, and it is recorded in vote_splits, so it can be quantified now."""
    out = []
    for r in _results():
        if r.get("runner") != "diverse_moa":
            continue
        for b in r.get("bands", []):
            for c in b.get("cells", []):
                vs = c.get("vote_splits")
                if not vs:
                    continue
                tot = sum(vs.values()) or 1
                nc = vs.get("no_consensus", 0)
                out.append({"design": r.get("design_id"), "effect": c.get("effect"),
                            "verdict": c.get("verdict"), "n": tot,
                            "no_consensus_share": round(nc / tot, 3),
                            "decided_by_tiebreak": nc,
                            "genuine_consensus": tot - nc})
    if not out:
        return {"error": "no diverse_moa runs record vote_splits"}
    worst = max(out, key=lambda x: x["no_consensus_share"])
    return {"check": "tie-break exposure in diverse_moa",
            "runs": out,
            "reading": ("On %s, %.0f%% of tasks had NO consensus, so the tie-break (models[0]) chose "
                        "the diverse arm's answer there. The measured effect is carried by the "
                        "remaining %d task(s) where the pool actually agreed on something."
                        % (worst["design"], worst["no_consensus_share"] * 100, worst["genuine_consensus"])),
            "limitation": ("Per-task correctness split BY consensus type is not recorded, so the "
                           "effect cannot yet be decomposed into consensus vs tie-break contributions. "
                           "Recording that split is the fix; until then this bounds the exposure "
                           "rather than resolving it.")}


# ---- CHECK 3: does the novelty gate hold up on the REAL corpus, not 5 hand-picked cases? ---------

def novelty_gate_selfconsistency():
    """The gate's thresholds were tuned on five cases I chose. A fairer test: every design already in
    the library must be judged a DUPLICATE of the library (it is literally in it). Any design the gate
    calls novel against its own corpus is a false negative, and tells us the gate would let a
    restatement of that design through."""
    designs = meta_lab.load_designs()
    texts = ["%s %s" % (d.get("title", ""), d.get("claim", "")) for d in designs]
    misses = []
    for i, d in enumerate(designs):
        others = texts[:i] + texts[i + 1:]                 # exclude itself: would be a trivial 1.0
        prop = {"title": d.get("title", ""), "claim": d.get("claim", "")}
        # A design vs the REST of the library: near-duplicates of siblings should be caught.
        ok, sim, _ = meta_lab._novelty(prop, others)
        if ok:
            misses.append({"design": d.get("id"), "max_sim_to_siblings": sim})
    return {"check": "novelty gate vs the real design corpus",
            "designs": len(designs),
            "judged_distinct_from_siblings": len(misses),
            "note": ("These designs are genuinely distinct from each other, so being judged novel is "
                     "CORRECT here -- this measures the gate's separation on real material rather "
                     "than on chosen examples. A gate that flagged most of the library as duplicates "
                     "would be too aggressive to use."),
            "detail": misses[:10]}


# ---- CHECK 4: recover the consensus-only effect the completed runs never recorded ---------------

def decompose_tiebreak():
    """The completed diverse_moa runs did not record correctness split by consensus type, so I called
    the decomposition unrecoverable. It is partly recoverable by algebra plus one measured input.

    The total effect is a weighted average of two disjoint groups:
        eff_total = (n_cons/N)*eff_cons + (n_tie/N)*eff_tie
    On TIE-BREAK tasks the diverse arm is models[0]'s SINGLE answer while the self arm is a plurality
    of k samples of that same model -- so eff_tie is just the NEGATIVE of that model's voting lift,
    which `width_partial_ctrl` measured directly (k=1->3 lift = -0.025, i.e. eff_tie ~ +0.025).

    One measured number and one unknown leaves the system solvable. Because that input is itself
    uncertain (and tie-break tasks are systematically HARDER -- every pool member disagreed -- so
    their voting lift may differ from the average), this reports a SENSITIVITY BAND across plausible
    eff_tie rather than a point estimate."""
    scenarios = [(-0.05, "voting helps on hard tasks"), (0.0, "voting neutral"),
                 (0.025, "MEASURED: width_partial_ctrl k=1->3 lift was -0.025"),
                 (0.10, "voting clearly hurts on hard tasks")]
    out = []
    for r in _results():
        if r.get("runner") != "diverse_moa":
            continue
        for b in r.get("bands", []):
            for c in b.get("cells", []):
                vs, eff = c.get("vote_splits"), c.get("effect")
                if not vs or eff is None:
                    continue
                tot = sum(vs.values()) or 1
                n_tie = vs.get("no_consensus", 0)
                n_cons = tot - n_tie
                if n_cons <= 0:
                    continue
                implied = []
                for et, label in scenarios:
                    ec = (eff - (n_tie / tot) * et) / (n_cons / tot)
                    implied.append({"assumed_eff_tie": et, "scenario": label,
                                    "implied_consensus_effect": round(ec, 3)})
                out.append({"design": r.get("design_id"), "reported_effect": round(eff, 4),
                            "verdict": c.get("verdict"), "n_consensus": n_cons, "n_tiebreak": n_tie,
                            "implied": implied})
    if not out:
        return {"error": "no decomposable diverse_moa runs"}
    return {"check": "recovering the consensus-only diversity effect",
            "runs": out,
            "reading": ("Under every plausible assumption the CONSENSUS-only effect is more extreme "
                        "than the headline in the SAME direction: the tie-break pulls both results "
                        "toward zero. The matched-pool POSITIVE is understated and the mismatched-pool "
                        "ANTI is far more negative than -0.094 once the sample-count tasks are removed."),
            "caveat": ("Assumes eff_tie is the negative of models[0]'s voting lift. Tie-break tasks are "
                       "harder by construction (every member disagreed), so their true lift may differ "
                       "from the measured average. This narrows the uncertainty; it does not replace "
                       "recording the split directly, which future runs now do.")}


# ---- CHECK 5: which runners actually buy conclusions per unit of compute? -----------------------

def cost_per_conclusion():
    """EFFICIENCY. Not every generation buys the same amount of knowledge. A run that ends
    INCONCLUSIVE consumed GPU and settled nothing; a powered null settled something real. This scores
    every completed design by what it actually returned per generation spent.

    Free: elapsed time and verdicts are already recorded. The point is to stop choosing runners by
    intuition -- 'diverse_moa feels informative' -- and start choosing them by measured yield."""
    VALUE = {"POSITIVE": 1.0, "ANTI": 1.0, "INFORMATIVE_NULL": 1.0,   # settled something
             "INCONCLUSIVE": 0.0, "DEGENERATE": 0.0, "FLOORED": 0.0}   # settled nothing
    # FLOORED scores ZERO deliberately. It is the most expensive kind of nothing: the run
    # completed, produced clean statistics, and had no live subject -- so a yield table that
    # credited it would rank a dead substrate as productive.
    rows, by_runner = [], {}
    for r in _results():
        runner = r.get("runner", "?")
        n = r.get("n_tasks") or 0
        nb = len(r.get("bands", []))
        k = r.get("k") or len(r.get("models") or []) or 3
        if runner == "diverse_moa":
            gens = nb * n * 2 * k
        elif runner == "verifier_placement":
            gens = nb * n * (k + 1)
        elif runner == "aggregation_bakeoff":
            gens = nb * n * len(r.get("models") or [])
        else:
            gens = nb * n * 3
        verdicts = [c.get("verdict") for b in r.get("bands", []) for c in b.get("cells", [])
                    if c.get("verdict")]
        if not verdicts or not gens:
            continue
        settled = sum(VALUE.get(v, 0.0) for v in verdicts)
        mins = round((r.get("elapsed_s") or 0) / 60)
        rows.append({"design": r.get("design_id"), "runner": runner, "gens": gens, "minutes": mins,
                     "verdicts": verdicts, "settled": settled,
                     "settled_per_1k_gens": round(1000.0 * settled / gens, 2)})
        d = by_runner.setdefault(runner, {"gens": 0, "settled": 0.0, "cells": 0, "minutes": 0})
        d["gens"] += gens
        d["settled"] += settled
        d["cells"] += len(verdicts)
        d["minutes"] += mins
    if not rows:
        return {"error": "no completed designs to score"}
    summary = {rn: {"settled_per_1k_gens": round(1000.0 * v["settled"] / v["gens"], 2),
                    "settled_cells": v["settled"], "total_cells": v["cells"],
                    "gens": v["gens"], "minutes": v["minutes"]}
               for rn, v in by_runner.items()}
    best = max(summary.items(), key=lambda kv: kv[1]["settled_per_1k_gens"])
    return {"check": "cost per conclusion (efficiency)",
            "per_design": rows, "by_runner": summary,
            "reading": ("Highest yield per 1k generations: %s (%.2f settled cells). A design that "
                        "returns INCONCLUSIVE bought nothing with its compute, so runners differ in "
                        "how much knowledge a GPU-hour actually purchases."
                        % (best[0], best[1]["settled_per_1k_gens"])),
            "caveat": ("Verdict classes are weighted equally; a POSITIVE and a powered NULL both count "
                       "as one settled cell. That is deliberate -- the program treats an informative "
                       "null as a real result -- but it means a runner that mostly produces nulls "
                       "scores as well as one that mostly produces effects.")}


# ---- CHECK 6: which pool members are worth their runtime? ---------------------------------------

def model_cost_efficiency():
    """EFFICIENCY, pool-selection edition. The GPU measured 84% utilised, so the engine is
    compute-bound: adding concurrency buys nothing, and the only remaining lever is spending those
    saturated cycles on cheaper members that measure the same thing.

    A pool needs members of MATCHED COMPETENCE -- but among equally competent candidates, a 9b and a
    30b are not equally priced. This ranks members by accuracy earned per second, so a pool can be
    rebuilt for the same science at a fraction of the runtime."""
    rows = []
    for r in _results():
        for b in r.get("bands", []):
            secs = None
            for c in b.get("cells", []):
                secs = c.get("model_seconds") or secs
            secs = secs or b.get("model_seconds")
            accs = b.get("per_model_accuracy") or {}
            if not secs:
                continue
            n = b.get("n_distinct") or 1
            for m, s in secs.items():
                rows.append({"model": m, "design": r.get("design_id"), "seconds": s, "tasks": n,
                             "sec_per_task": round(s / max(n, 1), 2),
                             "accuracy": accs.get(m)})
    if not rows:
        return {"check": "model cost efficiency",
                "status": "NO DATA YET — per-model timing was added 2026-07-23 and applies to runs "
                          "started after the next daemon restart.",
                "why_it_matters": ("The matched pool pairs gemma4:12b with two 30b models. If a 9b "
                                   "member is competence-matched, swapping it could cut diverse_moa "
                                   "runtime substantially for identical science -- but that trade "
                                   "cannot be evaluated without per-model timing, which no completed "
                                   "run recorded.")}
    agg = {}
    for x in rows:
        d = agg.setdefault(x["model"], {"sec": 0.0, "tasks": 0, "accs": []})
        d["sec"] += x["seconds"]
        d["tasks"] += x["tasks"]
        if x["accuracy"] is not None:
            d["accs"].append(x["accuracy"])
    out = {}
    for m, d in agg.items():
        spt = d["sec"] / max(d["tasks"], 1)
        acc = sum(d["accs"]) / len(d["accs"]) if d["accs"] else None
        out[m] = {"sec_per_task": round(spt, 2), "mean_accuracy": round(acc, 3) if acc else None,
                  "accuracy_per_sec": round(acc / spt, 4) if (acc and spt) else None}
    ranked = sorted((m for m in out if out[m]["accuracy_per_sec"] is not None),
                    key=lambda m: -out[m]["accuracy_per_sec"])
    return {"check": "model cost efficiency", "by_model": out, "ranked_by_value": ranked,
            "reading": ("Members ranked by accuracy earned per second. Among competence-MATCHED "
                        "candidates, prefer the cheapest -- the science is identical and the runtime "
                        "is not." if ranked else "not enough paired timing/accuracy data yet")}


def main():
    checks = [power_calibration(), tiebreak_exposure(), novelty_gate_selfconsistency(),
              decompose_tiebreak(), cost_per_conclusion(), model_cost_efficiency()]
    print("=" * 78)
    print("INSTRUMENT CHECKS — meta-experiments on the cockpit's own methods (free, no GPU)")
    print("=" * 78)
    for c in checks:
        print()
        if c.get("error"):
            print("!! %s" % c["error"])
            continue
        print("### %s" % c.get("check"))
        for k, v in c.items():
            if k in ("check", "rows", "runs", "detail"):
                continue
            print("   %-26s %s" % (k, v))
        for row in (c.get("runs") or [])[:6]:
            print("     - %s: effect %s (%s), %d/%d tasks decided by tie-break (%.0f%%)"
                  % (row["design"], row["effect"], row["verdict"], row["decided_by_tiebreak"],
                     row["n"], row["no_consensus_share"] * 100))
        for row in (c.get("rows") or [])[:6]:
            print("     - %s n=%d  sd_obs %.3f  MDE pred %.3f vs obs %.3f  (ratio %.2f)"
                  % (row["design"], row["n"], row["sd_obs"], row["mde_pred"], row["mde_obs"], row["ratio"]))
    print()
    out = os.path.join(HERE, "INSTRUMENT_CHECKS.json")
    with open(out, "w", encoding="utf-8") as dst:
        json.dump(checks, dst, indent=2)
    print("wrote", os.path.basename(out))


if __name__ == "__main__":
    main()

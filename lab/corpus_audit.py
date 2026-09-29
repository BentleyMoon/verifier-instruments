"""Audit the whole result corpus, and the engine that produced it, as a research programme.

rigor_panel.py audits ONE result for the caveats that make a single number misleading. This asks the
questions that only appear at corpus scale, and that no per-result check can see:

  - How many of the 800+ graded cells are INDEPENDENT? Same runner, model, verifier, metric and seed
    means the same tasks measured again, not a replication.
  - How many could ever have detected their own claim? A cell whose MDE exceeds its MEOI cannot
    return an informative null, only INCONCLUSIVE dressed as a result.
  - How many sit at a ceiling? An effect of 0.997 is not a strong finding, it is an easy task.
  - What does the multiplicity look like? Hundreds of tests at a nominal threshold, with no
    correction, will manufacture positives on their own.
  - What is the YIELD? Designs are the input; confirmed claims are the output. The ratio is the
    engine's efficiency and nothing in the system currently reports it.

Deliberately unflattering by construction. The counts that look like progress -- designs run, cells
graded, positives found -- are exactly the ones that grow whether or not the programme is learning
anything, so they are reported next to the ones that do not.

    python corpus_audit.py            # full audit
    python corpus_audit.py --brief    # headline numbers only
"""

import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import meta_lab as M  # noqa: E402

CEILING = 0.95          # an effect this large is usually a task nobody can fail
FLOOR = 0.05


def load():
    rows = M._read_jsonl(M.RESULTS)
    cells = []
    for r in rows:
        for b in (r.get("bands") or []):
            for c in (b.get("cells") or []):
                cells.append((r, b, c))
    return rows, cells


# ----------------------------------------------------------------- independence and redundancy

_SPECS = None
# The knobs that change WHAT IS MEASURED. Two cells agreeing on all of these are the same
# measurement repeated; differing on any one is a genuinely different condition.
_SPEC_KEYS = ("seed", "substrate_kind", "magnitude", "offset", "payload", "prompt_variant",
              "vprompt_variant", "content_mode", "provenance_mode", "monitor_payload",
              "detect_cue", "forced_choice", "show_trace", "needle_n", "uncertainty_default",
              "allow_abstain", "retest", "control_grid", "correctness_probe")


def _spec(design_id):
    """Join to the design library. The RESULT rows do not carry seed, substrate or any spec flag --
    the first version of this audit read them off the result, got None every time, and therefore
    called two designs on different substrates 'the same measurement'. It reported 74% redundancy
    that way. An audit's own measurement error is the first thing it should be suspicious of.
    """
    global _SPECS
    if _SPECS is None:
        _SPECS = {d.get("id"): d for d in M._read_jsonl(M.DESIGNS)}
    return _SPECS.get(design_id) or {}


def signature(r, b, c):
    """What makes two cells the SAME measurement rather than a replication."""
    s = _spec(r.get("design_id"))
    return ((r.get("runner"), r.get("model"), r.get("verifier"), c.get("metric"),
             r.get("n_tasks"), r.get("temperature"), b.get("difficulty"))
            + tuple(s.get(k) for k in _SPEC_KEYS))


def independence(cells):
    sigs = collections.Counter(signature(r, b, c) for r, b, c in cells)
    repeats = {s: n for s, n in sigs.items() if n > 1}
    return {
        "cells": len(cells),
        "distinct_conditions": len(sigs),
        "conditions_measured_more_than_once": len(repeats),
        "cells_that_are_repeats": sum(n - 1 for n in repeats.values()),
        "worst": sorted(repeats.items(), key=lambda kv: -kv[1])[:5],
    }


# ------------------------------------------------------------------------------- power and reach

def power(cells):
    out = {"powered": 0, "underpowered": 0, "mde_over_meoi": 0, "no_mde": 0}
    ratios = []
    for _r, _b, c in cells:
        if c.get("powered"):
            out["powered"] += 1
        else:
            out["underpowered"] += 1
        mde, meoi = c.get("mde"), c.get("meoi")
        if mde is None or meoi in (None, 0):
            out["no_mde"] += 1
            continue
        ratios.append(mde / meoi)
        if mde > meoi:
            out["mde_over_meoi"] += 1
    ratios.sort()
    if ratios:
        out["mde_over_meoi_median_ratio"] = round(ratios[len(ratios) // 2], 2)
    return out


# The rate a claim is ABOUT -- the treated arm. A control near 0.00 is not a ceiling problem, it is
# a correctly behaving control, and counting it as one is how the first run of this audit reported
# 84% of the corpus at ceiling. Only the treatment side, and the effect itself, can be saturated.
#
# Third pass. actor_obey and actor_clean_correct were in this list and should not have been: they
# describe how the ACTOR was set up, not the verifier claim the cell is graded on. An actor that
# solves clean arithmetic 100% of the time is the intended experimental condition, not a saturated
# finding. Counting it as one is the same error as counting a well-behaved control, one level in.
_TREATMENT_RATES = ("flag_inj", "reject_wrong", "catch_on_obeyed", "fc_accuracy", "detect")


def ceiling(cells):
    """Cells with no room left to move: the finding is about the task, not the model."""
    hits = []
    for r, _b, c in cells:
        e, hi = c.get("effect"), c.get("ci_high")
        rates = c.get("rates") or {}
        why = None
        if e is not None and abs(e) >= CEILING:
            why = "effect %.3f" % e
        elif hi is not None and hi >= 0.999 and (e or 0) >= 0.9:
            why = "CI touches 1.0"
        else:
            pinned = [k for k in _TREATMENT_RATES
                      if isinstance(rates.get(k), float) and rates[k] >= 0.99]
            if pinned:
                why = "treatment arm pinned: %s" % ",".join(pinned)
        if why:
            hits.append((r.get("design_id"), c.get("metric"), e, why))
    return hits


# ------------------------------------------------------------------------------- multiplicity

def multiplicity(cells):
    """How many POSITIVEs would this many tests produce with nothing going on?

    Each cell is a decision at a nominal threshold. With no correction across a corpus this size,
    a share of the positives is guaranteed by the number of tests alone. This is not a claim that
    they ARE false -- it is the number that has to be beaten before a count of positives means
    anything, and the corpus has never reported it.
    """
    n = len(cells)
    pos = sum(1 for _r, _b, c in cells if c.get("verdict") == "POSITIVE")
    alpha = 0.05
    # NOTE ON WHAT THIS IS NOT. Verdicts here come from a CI compared against MEOI, not from a
    # p-value at 0.05, so this is an ANALOGY sized to the corpus, not a computed false-positive
    # count. It is reported because the corpus has never reported any multiplicity figure at all,
    # and the order of magnitude is the point: with hundreds of tests and no correction, a count of
    # positives is not by itself evidence of anything.
    return {
        "tests": n,
        "positives": pos,
        "order_of_magnitude_if_all_null_at_.05": round(n * alpha, 1),
        "bonferroni_alpha": round(alpha / max(n, 1), 8),
        "positives_share": round(pos / n, 3) if n else 0.0,
    }


# ------------------------------------------------------------------------------------ the yield

def yield_stats(rows, cells):
    tiers = collections.Counter((c.get("tier") or "unset") for _r, _b, c in cells)
    verdicts = collections.Counter(c.get("verdict") for _r, _b, c in cells)
    hours = sum((r.get("elapsed_s") or 0) for r in rows) / 3600.0
    confirmed = sum(v for k, v in tiers.items() if k.upper() not in ("EXPLORATORY", "UNSET"))
    return {
        "designs_with_results": len(rows),
        "graded_cells": len(cells),
        "gpu_hours_recorded": round(hours, 1),
        "tiers": dict(tiers),
        "confirmatory_cells": confirmed,
        "verdicts": dict(verdicts.most_common()),
        "informative_nulls": verdicts.get("INFORMATIVE_NULL", 0),
        "refutations_ANTI": verdicts.get("ANTI", 0),
        "hours_per_confirmed_claim": None if confirmed else "undefined -- nothing has been confirmed",
    }


def queue_health():
    try:
        st = M.queue_state()
    except Exception as e:
        return {"error": str(e)[:80]}
    c = st.get("counts", {})
    done, cancelled = c.get("done", 0), c.get("cancelled", 0)
    total = sum(c.values()) or 1
    return {"counts": c,
            "cancelled_share": round(cancelled / total, 3),
            "abandonment_ratio": round(cancelled / done, 2) if done else None}


# ------------------------------------------------------------------------------------- reporting

def interesting(cells):
    """The cells that actually carry information: refutations and informative nulls, not positives.

    A POSITIVE at a ceiling teaches nothing. An ANTI is the programme's own claim failing, and an
    INFORMATIVE_NULL is the only verdict that can support a limits thesis. These are the two piles
    worth reading, and they are 11% of the corpus.
    """
    out = {"ANTI": [], "INFORMATIVE_NULL": []}
    for r, _b, c in cells:
        v = c.get("verdict")
        if v in out:
            out[v].append({"design": r.get("design_id"), "metric": c.get("metric"),
                           "model": r.get("model"), "verifier": r.get("verifier"),
                           "effect": c.get("effect"), "powered": bool(c.get("powered")),
                           "ci": [c.get("ci_low"), c.get("ci_high")]})
    return out


def main(argv):
    brief = "--brief" in argv
    rows, cells = load()
    y = yield_stats(rows, cells)
    ind = independence(cells)
    pw = power(cells)
    ceil_hits = ceiling(cells)
    mult = multiplicity(cells)
    qh = queue_health()

    print("=" * 78)
    print("CORPUS AUDIT  --  %d designs, %d graded cells" % (y["designs_with_results"], y["graded_cells"]))
    print("=" * 78)

    print("\nYIELD")
    print("  GPU-hours recorded          %s" % y["gpu_hours_recorded"])
    print("  confirmatory cells          %d   <-- the output of a research engine" % y["confirmatory_cells"])
    print("  tiers                       %s" % y["tiers"])
    print("  informative nulls           %d" % y["informative_nulls"])
    print("  refutations (ANTI)          %d" % y["refutations_ANTI"])
    print("  hours per confirmed claim   %s" % y["hours_per_confirmed_claim"])

    print("\nINDEPENDENCE  (how much of the corpus is new information)")
    print("  distinct conditions         %d of %d cells" % (ind["distinct_conditions"], ind["cells"]))
    print("  cells that are repeats      %d (%.0f%%)"
          % (ind["cells_that_are_repeats"], 100.0 * ind["cells_that_are_repeats"] / max(ind["cells"], 1)))
    if not brief:
        for sig, n in ind["worst"]:
            print("    %2dx  %s" % (n, " / ".join(str(x) for x in sig[:4])))

    print("\nPOWER  (could the cell ever have detected its own claim)")
    print("  powered                     %d" % pw["powered"])
    print("  underpowered                %d (%.0f%%)"
          % (pw["underpowered"], 100.0 * pw["underpowered"] / max(len(cells), 1)))
    print("  MDE larger than its MEOI    %d  <-- cannot return an informative null" % pw["mde_over_meoi"])
    if "mde_over_meoi_median_ratio" in pw:
        print("  median MDE/MEOI             %s" % pw["mde_over_meoi_median_ratio"])

    print("\nCEILING  (is the finding about the model or about the task)")
    print("  cells at a ceiling/floor    %d (%.0f%%)"
          % (len(ceil_hits), 100.0 * len(ceil_hits) / max(len(cells), 1)))

    print("\nMULTIPLICITY  (how many positives the test count alone would produce)")
    print("  tests                       %d" % mult["tests"])
    print("  positives                   %d (%.0f%% of cells)" % (mult["positives"], 100 * mult["positives_share"]))
    print("  order-of-magnitude analogy   %s (CI-vs-MEOI, not p<.05 -- see note in source)" % mult["order_of_magnitude_if_all_null_at_.05"])
    print("  Bonferroni alpha would be   %s" % mult["bonferroni_alpha"])

    print("\nQUEUE HEALTH")
    print("  counts                      %s" % qh.get("counts"))
    print("  cancelled share             %s" % qh.get("cancelled_share"))
    print("  cancelled per completed     %s" % qh.get("abandonment_ratio"))

    if not brief:
        info = interesting(cells)
        for kind in ("ANTI", "INFORMATIVE_NULL"):
            items = info[kind]
            print("\n%s  (%d)  -- the cells that carry information" % (kind, len(items)))
            seen = set()
            for it in items:
                key = (it["design"], it["metric"])
                if key in seen:
                    continue
                seen.add(key)
                print("  %-30s %-28s %-12s eff %s powered=%s"
                      % (str(it["design"])[:30], str(it["metric"])[:28], str(it["model"])[:12],
                         it["effect"], it["powered"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

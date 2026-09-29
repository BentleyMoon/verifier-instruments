"""Retraction as DATA, so the corpus can answer what it currently cannot.

THE GAP THIS CLOSES. On 2026-08-17 an audit went looking for the programme's retraction rate, because
inflow precision -- not store contamination -- is what predicts where a knowledge base ends up. A
store can read 25 percent poisoned while headed for 100, and only the precision of what enters
predicts the asymptote. Inflow precision needs a retraction rate, and there was none to compute.

Retractions were recorded in three places and none of them queryable: prose inside FINDINGS.json
detail fields, prose inside 33 META_DESIGNS entries, and -- in RESULTS, across 603 rows -- nowhere at
all. Every `status` field was the empty string. FINDINGS.json entry 10 states in a sentence that 30
results were retracted early on and 15 designs invalidated on 2026-08-11, so the programme KNEW its
history and no query could recover it. Those retracted results sit in the ledger as the same bytes as
valid ones, and every aggregate ever run over RESULTS has silently included them.

Three separate attempts that day to reconstruct retraction status by matching text or numbers all
returned plausible answers that were WRONG, because the matching keys are shared by hundreds of rows.
The lesson is the design of this file: a relation between records has to be recorded as a relation.

WHY APPEND-ONLY, AND WHY NOT IN RESULTS. Editing a graded result to mark it retracted rewrites the
evidentiary record, and this programme's fingerprint discipline exists precisely so results are not
rewritten. A retraction is a NEW claim ABOUT a result, made later by someone who looked, and it
carries its own author, date and evidence. So it lives beside the corpus and is joined at read time.
Nothing here mutates RESULTS.

WHAT A RETRACTION IS NOT. It is not a null, a failed replication, or a disappointing effect. Those are
results. A retraction says the number should never have been produced -- the instrument was broken,
the control was run as the treatment, the subject was absent. Recording an honest null here would
corrupt the inflow-precision estimate in the flattering direction, which is the one failure mode this
file must not have.
"""
import io
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "RETRACTIONS.jsonl")

# A retraction must say WHY in a way that can be checked, not merely asserted.
REASONS = (
    "instrument_defect",     # the harness produced the number, not the phenomenon
    "control_as_treatment",  # the arms were not what the design said they were
    "absent_subject",        # nothing live was measured
    "superseded",            # a later, better-controlled run replaces this one
    "misdiagnosed",          # the claim was right about the data and wrong about the cause
)


def retract(design_id, reason, evidence, by, metric=None, superseded_by=None, ts=None):
    """Record that a result should not be used, and what establishes that.

    `evidence` is required and is the point: a retraction with no evidence is an opinion, and this
    ledger feeds a precision estimate that decides how much verification the programme needs.
    """
    if reason not in REASONS:
        raise ValueError("reason must be one of %s" % (REASONS,))
    if not evidence or len(str(evidence).strip()) < 40:
        raise ValueError("evidence must state what establishes the retraction, concretely")
    rec = {"design_id": design_id, "metric": metric, "reason": reason,
           "evidence": str(evidence).strip(), "by": by, "superseded_by": superseded_by,
           "ts": ts or time.strftime("%Y-%m-%dT%H:%M:%S")}
    with io.open(LEDGER, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load():
    if not os.path.exists(LEDGER):
        return []
    out = []
    for line in io.open(LEDGER, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def retracted_designs():
    """{design_id: [records]} for whole-design retractions (metric is None)."""
    out = {}
    for r in load():
        if r.get("metric") is None and r.get("design_id"):
            out.setdefault(r["design_id"], []).append(r)
    return out


def is_retracted(design_id, metric=None):
    for r in load():
        if r.get("design_id") != design_id:
            continue
        if r.get("metric") is None or r.get("metric") == metric:
            return True
    return False


def rate(results_iter):
    """Retraction rate over a result stream -- the inflow-precision estimate the audit wanted.

    Returns (retracted, total, rate). This is a LOWER BOUND on contamination and should always be
    read as one: it counts what someone went back and checked. The 2026-08-11 incident ran for eleven
    days before anyone looked, and nothing in a store tells you about the part nobody has audited yet.
    """
    bad = retracted_designs()
    seen, hit = set(), set()
    for r in results_iter:
        d = r.get("design_id")
        if not d:
            continue
        seen.add(d)
        if d in bad:
            hit.add(d)
    n = len(seen)
    return len(hit), n, (len(hit) / float(n) if n else 0.0)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, HERE)
    import meta_lab as M
    recs = load()
    print("RETRACTION LEDGER -- %d record(s)" % len(recs))
    for r in recs:
        print("  %-24s %-20s %s" % (r["design_id"], r["reason"], r["ts"][:10]))
        print("      %s" % r["evidence"][:150])
    k, n, p = rate(M._read_jsonl(M.RESULTS))
    print()
    print("%d of %d designs in RESULTS are retracted -- %.1f%%" % (k, n, 100 * p))
    print("Read as a LOWER BOUND on contamination: it counts only what someone went back and checked.")

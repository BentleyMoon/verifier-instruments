"""Find results whose effect lives on rows where the model NEVER ANSWERED.

Compass rule 8: never-answered is not answered-wrong. Conflating them lets truncation and parse
failure masquerade as inability, and MANUFACTURES the exact quantity an experiment exists to measure.
Rule 7: order work by the cheapest thing that could invalidate the most expensive thing. This reads
committed data and costs no GPU.

The case that motivated it, RESULT_SCALEOVERSIGHT_mmm_scalable.json, 15 rows:

    all rows                  gen_alone 0.467 -> weak1 0.667 -> weakK 0.800 -> strong 1.000
    generator DID answer (7)  gen_alone 1.000   <- no headroom at all
    generator never answered (8, 53%)  gen_alone 0.000 by construction

The entire "scalable oversight ladder" was measured on rows where the generator emitted nothing
parseable. Where it actually answered it was already perfect. The experiment measured its parser.

Two detectors, both deliberately dumb, because a subtle detector on this question would itself need
auditing:

  SPLIT   an answer-bearing field is null on a material share of rows AND the binary arms differ
          sharply between the answered and unanswered subsets. That is the SCALEOVERSIGHT signature:
          the effect is carried by the rows with no answer in them.
  FLOOR   a binary arm is EXACTLY 0.0 (or 1.0) across enough rows to be improbable. The compass
          already says a suspicious zero is an artifact until proven real; this just lists them.
          RESULT_VERIFIERXFER_mmm_externalcheck.json has weak_other_verify == 0.000 on all 24 rows,
          in a file that records no answer field at all -- so the conflation there is not merely
          present, it is UNMEASURABLE, which is the more serious defect.

Reports, never edits. What to do about a flagged result is a research judgement.

    python unanswered_audit.py [--json]
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Fields that plausibly hold "what the model actually said". A null here means no answer was parsed.
ANSWER_HINTS = ("proposed", "answer", "parsed", "response", "output", "emitted", "value", "prediction")
# Rows also carry ground truth and identifiers; those nulls mean something else entirely.
NOT_ANSWERS = ("true", "truth", "gold", "target", "task", "id", "name", "label", "seed", "model")

MATERIAL_NULL_RATE = 0.10   # below this, a few unparsed rows cannot carry an effect
SPLIT_GAP = 0.25            # answered-vs-unanswered difference that makes the split load-bearing
FLOOR_MIN_ROWS = 10         # an exact 0.0 on 3 rows is unremarkable


def binary_fields(rows):
    """Fields whose non-null values are all 0/1 -- the per-row arm scores."""
    out = []
    for k in rows[0]:
        vals = [r.get(k) for r in rows]
        nn = [v for v in vals if v is not None]
        if len(nn) >= max(3, len(rows) // 2) and all(v in (0, 1, True, False) for v in nn):
            out.append(k)
    return out


def answer_fields(rows):
    """Fields that look like 'what the model said' and are null on at least one row."""
    out = []
    for k in rows[0]:
        kl = k.lower()
        if any(b == kl or kl.startswith(b + "_") for b in NOT_ANSWERS):
            continue
        if not any(h in kl for h in ANSWER_HINTS):
            continue
        if any(r.get(k) is None for r in rows):
            out.append(k)
    return out


def mean(rows, k):
    vals = [1 if r.get(k) in (1, True) else 0 for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def audit_one(path):
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return []
    rows = d.get("rows")
    if not isinstance(rows, list) or len(rows) < 5 or not isinstance(rows[0], dict):
        return []

    findings = []
    bins = binary_fields(rows)
    ans = answer_fields(rows)

    for af in ans:
        unans = [r for r in rows if r.get(af) is None]
        answered = [r for r in rows if r.get(af) is not None]
        rate = len(unans) / len(rows)
        if rate < MATERIAL_NULL_RATE or not answered:
            continue
        for bf in bins:
            if bf == af:
                continue
            ma, mu = mean(answered, bf), mean(unans, bf)
            if abs(ma - mu) >= SPLIT_GAP:
                findings.append({
                    "kind": "SPLIT", "file": os.path.basename(path), "answer_field": af,
                    "arm": bf, "rows": len(rows), "null_rate": round(rate, 3),
                    "mean_all": round(mean(rows, bf), 3),
                    "mean_answered": round(ma, 3), "mean_unanswered": round(mu, 3),
                    "gap": round(abs(ma - mu), 3),
                })

    # A single floored arm is usually FINE and reporting it is noise: a control arm SHOULD read 0.0
    # (no fabrication without the trigger), which is the result, not a defect. The un-innocent case is
    # a run where EVERY arm is pinned to the same value -- treatment and control alike. Then nothing
    # varied, so nothing was measured, and no contrast in that file can carry a claim.
    #
    # Arms are counted over ALL rate-valued fields, not just 0/1 ones. The first version of this check
    # only looked at binary fields and reported thirty dead runs; two spot-checks were both FALSE.
    # PERMMIRAGE's valid_token is 0.900 and PARADEFENSE's direct is 0.800 -- non-binary, therefore
    # invisible to the check, so the arms that legitimately read 0.0 were all that remained and the run
    # looked pinned. Both are real findings, and one is a clean positive. An instrument that reports a
    # healthy experiment as dead is the same failure as one that reports noise as a result.
    means = {}
    for k in rows[0]:
        kl = k.lower()
        if any(b == kl or kl.startswith(b + "_") for b in NOT_ANSWERS) or "_val" in kl:
            continue
        vals = [r.get(k) for r in rows
                if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
        if len(vals) >= FLOOR_MIN_ROWS and all(0.0 <= v <= 1.0 for v in vals):
            means[k] = sum(vals) / len(vals)
    if len(means) >= 2 and len(set(means.values())) == 1 and list(means.values())[0] in (0.0, 1.0):
        findings.append({
            "kind": "DEAD", "file": os.path.basename(path), "arms": sorted(means),
            "rows": len(rows), "value": list(means.values())[0],
            "records_an_answer_field": bool(ans),
        })
    return findings


# Rates that say whether the SUBJECT of the measurement was alive at all. A cell can be powered,
# CI-separated and graded POSITIVE while the actor it is about produced nothing gradeable.
#
# IMPORTED, not restated. The gate in meta_lab and this audit are the two halves of one guarantee,
# and on 2026-08-03 -- written hours apart, both by the same agent -- they had already diverged in
# BOTH directions. An auditor whose vocabulary is narrower than the gate's reports all-clear on cells
# the gate would refuse; one whose vocabulary is wider flags cells the gate happily grades. Either way
# the pair silently agrees on a corpus neither has actually checked.
try:
    import meta_lab as _ml
    _GRADED_RATES = tuple(_ml.SUBJECT_LIVENESS_RATES)
except Exception:                       # the audit must still run if the harness cannot be imported
    _GRADED_RATES = ("actor_clean_correct", "base_acc", "ext_solve_acc")

# The dgen row format is a genuinely DIFFERENT vocabulary -- per-row 0/1 fields, not aggregate rates
# -- so it is named separately rather than merged into the gate's list. Distinct on purpose, not by
# drift, and that distinction is the thing the test asserts.
DGEN_LIVENESS_FIELDS = ("gen_alone", "weak_alone", "acc_1")
LIVENESS_RATES = _GRADED_RATES + DGEN_LIVENESS_FIELDS


def _floored_cells(d, design_id):
    """Graded cells (bands -> cells -> rates) whose SUBJECT produced nothing.

    The rows-based detectors above never saw this format, so the rigor-graded results -- the core of
    the corpus -- went unchecked while the dgen-style files were scanned. That blind spot hid the
    inversion ladder: six designs, ~25k generations, every one with actor_clean_correct = 0.0 at every
    rung INCLUDING THE EASIEST, in a corpus where the same model on the same runner scores 1.000
    elsewhere. Nothing about deference was measured. MDE-vs-MEOI power analysis cannot see this,
    because precision about a floored subject is still precision about nothing.
    """
    bands = d.get("bands")
    if not isinstance(bands, list) or not bands:
        return []
    out = []
    for b in bands:
        for c in (b.get("cells") or []):
            rates = c.get("rates") or {}
            for key in LIVENESS_RATES:
                if key in rates and rates[key] == 0:
                    out.append({
                        "kind": "FLOORED", "file": "META_RESULTS.jsonl",
                        "design": design_id, "metric": c.get("metric"),
                        "rate": key, "verdict": c.get("verdict"),
                        "effect": round(c["effect"], 3) if isinstance(c.get("effect"), float) else None,
                    })
                    break
    return out


def main():
    all_f = []
    scanned = 0
    for p in sorted(glob.glob(os.path.join(HERE, "RESULT_*.json"))):
        f = audit_one(p)
        if f:
            all_f.extend(f)
        scanned += 1

    # The rigor-graded corpus is NOT in RESULT_*.json -- meta_lab._record_result appends to
    # META_RESULTS.jsonl. The first attempt at closing this blind spot scanned RESULT_*.json for
    # bands and reported zero floored cells, which read as a clean bill of health for the 351 results
    # it had never opened. A detector pointed at the wrong corpus is worse than no detector: it
    # answers.
    graded = os.path.join(HERE, "META_RESULTS.jsonl")
    if os.path.exists(graded):
        with open(graded, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                scanned += 1
                all_f.extend(_floored_cells(rec, rec.get("design_id") or "?"))

    if "--json" in sys.argv:
        print(json.dumps({"scanned": scanned, "findings": all_f}, indent=2))
        return

    splits = sorted([f for f in all_f if f["kind"] == "SPLIT"],
                    key=lambda f: -(f["null_rate"] * f["gap"]))
    floors = [f for f in all_f if f["kind"] == "DEAD"]

    print("scanned %d result files with per-row data\n" % scanned)
    print("SPLIT -- the effect differs between answered and unanswered rows (%d)" % len(splits))
    for f in splits:
        print("  %-46s %-16s null=%.0f%%  all=%.3f  answered=%.3f  unanswered=%.3f"
              % (f["file"][:46], f["arm"][:16], 100 * f["null_rate"],
                 f["mean_all"], f["mean_answered"], f["mean_unanswered"]))
    if not splits:
        print("  none")

    print("\nDEAD -- every arm pinned to the same value, so nothing varied (%d)" % len(floors))
    for f in sorted(floors, key=lambda f: -f["rows"])[:30]:
        note = "" if f["records_an_answer_field"] else "  [no answer field -- unmeasurable]"
        print("  %-44s n=%-4d all arms=%.1f  arms=%s%s"
              % (f["file"][:44], f["rows"], f["value"],
                 ",".join(f["arms"][:4])[:44], note))
    if len(floors) > 30:
        print("  ... and %d more (nothing dropped silently -- run with --json for all)" % (len(floors) - 30))
    if not floors:
        print("  none")

    fl = [f for f in all_f if f["kind"] == "FLOORED"]
    print("\nFLOORED -- graded cells whose SUBJECT produced nothing (%d)" % len(fl))
    by_design = {}
    for f in fl:
        by_design.setdefault(f["design"], []).append(f)
    for did, items in sorted(by_design.items(), key=lambda kv: -len(kv[1]))[:20]:
        verdicts = sorted({str(i["verdict"]) for i in items})
        print("  %-26s %2d cell(s)  %-18s graded %s"
              % (str(did)[:26], len(items), items[0]["rate"], ",".join(verdicts)[:40]))
    if len(by_design) > 20:
        print("  ... and %d more designs" % (len(by_design) - 20))
    if not fl:
        print("  none")


if __name__ == "__main__":
    main()

"""Was this verdict produced by an instrument that had passed its controls?

A result carries a verdict, a CI and an effect. None of those say whether the thing that GRADED it
was working. On 2026-08-03 that distinction cost 30 cells: they carried POSITIVE, ANTI and SATURATED
with clean statistics, produced by a grader that had no concept of whether the subject was alive.
The verdicts were not wrong about their arithmetic. They were unverified about their instrument.

So trust is a JOIN, not a new judgement:

    result.verified_at (the grader hashes at grade time)
      x  CONTROLS_LOG.jsonl (which hashes ever passed their controls)
      x  fingerprint.graders() (what the graders are NOW)

FOUR STATES, because two orthogonal facts are being reported and collapsing them loses the one that
matters:

  VERIFIED    graded at hashes whose controls passed, and the graders have not changed since.
  SUPERSEDED  graded at hashes whose controls passed, but the graders have CHANGED. The arithmetic
              stands; the verdict may not survive a re-grade. This is the state 1,359 cells entered
              at 11:46 when the FLOORED gate landed, and 30 of them did not survive it.
  UNVERIFIED  no stamp at all. Every result recorded before stamping existed. NOT a synonym for
              wrong -- it means the question cannot be answered from the record, which is a
              different thing and must not be rendered as either green or red.
  FAILED      graded at hashes that were present in the controls log WITH a failing control.

Reports only. Nothing here re-grades, edits a ledger, or decides what to do -- that is decision 1 in
the cockpit, and it belongs to the operator.

    python trust.py [--json]
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fingerprint  # noqa: E402

RESULTS = os.path.join(HERE, "META_RESULTS.jsonl")
CONTROLS_LOG = os.path.join(HERE, "CONTROLS_LOG.jsonl")


def control_history():
    """{frozenset of (file, hash) -> passed?} for every control run ever recorded."""
    hist = {}
    if not os.path.exists(CONTROLS_LOG):
        return hist
    with open(CONTROLS_LOG, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            fp = rec.get("verified_at") or {}
            key = frozenset((k, v) for k, v in fp.items() if k in fingerprint.GRADERS)
            passed = rec.get("failed", 1) == 0 and rec.get("errored", 1) == 0 \
                and rec.get("canary_caught") is True
            # A later PASS at the same hashes supersedes an earlier FAIL: the instrument was fixed
            # without its content changing is impossible, so this can only mean a flaky fixture --
            # and recording the optimistic reading would hide exactly that. Keep the pessimistic one.
            hist[key] = hist.get(key, True) and passed
    return hist


def state_of(stamp, hist, now):
    if not stamp:
        return "UNVERIFIED", "no grader fingerprint recorded at grade time"
    key = frozenset((k, v) for k, v in stamp.items() if k in fingerprint.GRADERS)
    if key not in hist:
        return "UNVERIFIED", "graded at hashes no control run ever covered"
    if not hist[key]:
        return "FAILED", "graded at hashes whose control run failed"
    changed = [k for k in fingerprint.GRADERS if stamp.get(k) != now.get(k)]
    if changed:
        return "SUPERSEDED", "graders changed since: %s" % ", ".join(changed)
    return "VERIFIED", "controls passed at these hashes and nothing has changed"


def survey():
    now = fingerprint.graders()
    hist = control_history()
    counts = collections.Counter()
    rows = []
    if not os.path.exists(RESULTS):
        return {"error": "no META_RESULTS.jsonl"}
    with open(RESULTS, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            st, why = state_of(rec.get("verified_at"), hist, now)
            cells = sum(1 for b in (rec.get("bands") or []) for _ in (b.get("cells") or []))
            counts[st] += cells
            rows.append({"design": rec.get("design_id"), "ts": rec.get("ts"), "state": st,
                         "why": why, "cells": cells})
    return {"now": now, "control_runs": len(hist), "cells_by_state": dict(counts), "rows": rows}


def main():
    s = survey()
    if "error" in s:
        print(s["error"])
        return 1
    if "--json" in sys.argv:
        print(json.dumps(s, indent=2))
        return 0
    print("graders now: %s" % ", ".join("%s@%s" % (k, v) for k, v in sorted(s["now"].items())))
    print("control runs on record: %d\n" % s["control_runs"])
    total = sum(s["cells_by_state"].values())
    for st in ("VERIFIED", "SUPERSEDED", "UNVERIFIED", "FAILED"):
        n = s["cells_by_state"].get(st, 0)
        if n or st in ("VERIFIED", "UNVERIFIED"):
            print("  %-11s %6d cells  %5.1f%%" % (st, n, 100.0 * n / total if total else 0))
    print("\n%d graded cells across %d results" % (total, len(s["rows"])))
    ex = [r for r in s["rows"] if r["state"] != "VERIFIED"][:5]
    if ex:
        print("\nexamples:")
        for r in ex:
            print("  %-28s %-11s %s" % (str(r["design"])[:28], r["state"], r["why"][:58]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

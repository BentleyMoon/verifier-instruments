# -*- coding: utf-8 -*-
"""Recompute every DECLARED derived figure from its input cells, with re-serves replacing their parents.

THE GAP. A finding that prints a figure computed from cells -- a range, a mean, a difference -- prints a
number that appears nowhere in the ledger. quoted_numbers exempts it (derived_figures), and
leak_exposure's STALE "cannot see" it (its own docstring). So when an input cell is re-served and moves,
nothing notices: idea_ratchet Cell 2 planted exactly that (class C4) and every standing audit missed it.
The live case is finding 26's run-to-run floor, 0.0898 = max - min over six ndf_fixed_* cells that all
went through the gemma4:12b leak; finding 25 judges its deltas against that floor.

THE FIX IS A DECLARATION. A finding lists `derived`: [{"figure": "0.0898", "op": "range",
"metric": "oversight_discrimination", "value": "effect", "inputs": [design ids]}]. This check
recomputes each one from the ledger, taking X_rs in place of X whenever the re-serve has landed
(leak_exposure.superseded's rule: a re-serve supersedes the run it re-serves, never averages with it).

    STALE     every input resolved and the recomputed value no longer rounds to the printed figure
    WAITING   some input's re-serve is designed but not landed; the partial recomputation is reported
    BROKEN    an input has no value for the declared metric -- the declaration itself is wrong

Only STALE and BROKEN fail. A figure whose inputs are mid-correction is waiting, not wrong.

    python derived_check.py
"""
import io
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FINDINGS = os.path.join(HERE, "FINDINGS.json")


def _sd(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0


OPS = {
    "range": lambda xs: max(xs) - min(xs),
    "mean": lambda xs: sum(xs) / len(xs),
    "diff": lambda xs: xs[0] - xs[1],
    "sd": _sd,
}


def _value(rows, metric, key):
    """The declared value from the most recent result that carries the metric."""
    for r in reversed(rows or []):
        for b in r.get("bands") or []:
            for c in b.get("cells") or []:
                if c.get("metric") != metric:
                    continue
                v = c.get(key) if key in c else (c.get("rates") or {}).get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    return float(v)
    return None


def _places(figure):
    s = str(figure).lstrip("+-")
    return len(s.split(".")[1]) if "." in s else 0


def check(findings=None, designs=None, results=None, verbose=False):
    import meta_lab as M
    findings = findings if findings is not None else json.load(io.open(FINDINGS, encoding="utf-8"))["findings"]
    designs = designs if designs is not None else M._read_jsonl(M.DESIGNS)
    results = results if results is not None else M._read_jsonl(M.RESULTS)
    known = {d.get("id") for d in designs}
    R = defaultdict(list)
    for r in results:
        R[r.get("design_id")].append(r)

    stale, waiting, broken, ok = [], [], [], []
    for i, f in enumerate(findings):
        for d in f.get("derived") or []:
            op, metric, key = d.get("op"), d.get("metric"), d.get("value", "effect")
            item = {"finding": i + 1, "figure": str(d.get("figure")), "op": op}
            if op not in OPS or not d.get("inputs"):
                item["why"] = "unknown op %r or no inputs" % op
                broken.append(item)
                continue
            vals, pending, used = [], [], []
            for did in d["inputs"]:
                rs = did + "_rs"
                src = rs if R.get(rs) else did
                if src == did and rs in known:
                    pending.append(did)
                v = _value(R.get(src), metric, key)
                vals.append(v)
                used.append(src)
            if None in vals:
                item["why"] = "no %s/%s in %s" % (metric, key, [u for u, v in zip(used, vals) if v is None])
                broken.append(item)
                continue
            got = OPS[op](vals)
            item.update(recomputed=round(got, 6), superseded=[u for u in used if u.endswith("_rs")])
            places = _places(d["figure"])
            printed = float(d["figure"])
            matches = abs(round(abs(got), places) - abs(printed)) <= 10 ** -(places + 3)
            if pending:
                item["pending"] = pending
                waiting.append(item)
            elif matches:
                ok.append(item)
            else:
                stale.append(item)

    if verbose:
        print("DERIVED CHECK -- declared derived figures recomputed from their input cells")
        print("  ok %d   waiting %d   STALE %d   BROKEN %d" % (len(ok), len(waiting), len(stale), len(broken)))
        for x in waiting:
            print("     finding %-3d %s (%s)  now %.4f with %d re-served; waiting on %d"
                  % (x["finding"], x["figure"], x["op"], x["recomputed"], len(x["superseded"]), len(x["pending"])))
        for x in stale:
            print("  STALE  finding %-3d prints %s; its inputs now give %.4f (re-served: %s)"
                  % (x["finding"], x["figure"], x["recomputed"], ", ".join(x["superseded"]) or "none"))
        for x in broken:
            print("  BROKEN finding %-3d %s: %s" % (x["finding"], x["figure"], x["why"]))
    return stale, waiting, broken, ok


if __name__ == "__main__":
    s, w, b, _ok = check(verbose=True)
    sys.exit(1 if (s or b) else 0)

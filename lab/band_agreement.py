# -*- coding: utf-8 -*-
"""Refuse a cited number that its own design's other bands contradict.

THE INCIDENT, 2026-08-20. A survey of the corpus for mitigation-shaped metrics was written like this:

    pos = [x for x in rows if x[2] == "POSITIVE"]
    for d, e, v in sorted(pos, key=lambda x: -x[1])[:3]: ...

It filtered to POSITIVE cells and sorted by effect descending -- so it showed each design's BEST band
and nothing else. `specdecay_hermes3` came back as "+0.1577 POSITIVE" and was written into the
compass, a commit message, a published finding's caveat and an artifact as "the third mitigation in
the corpus with a positive verdict".

Its four bands are:

    band 1   -0.1098 ANTI
    band 2   -0.1098 ANTI
    band 3   +0.1577 POSITIVE     <- the cited one
    band 4   -0.0200 INCONCLUSIVE

Two ANTI, one POSITIVE, one INCONCLUSIVE. The design is majority-against the claim it was cited for.
An entire experiment was queued on the strength of that number.

WHAT MAKES THIS DIFFERENT FROM AN ORDINARY MISTAKE is that no existing check could see it. Every
individual cell is correctly graded. The ledger is intact. rigor did its job in all four bands.
findings_lint confirms the finding names a design that exists. verdict_lint confirms no strength word
outranks its result. The defect lives BETWEEN cells -- in which one got quoted -- and this programme
had no instrument that looked there.

WHAT COUNTS AS DISAGREEMENT. Not any spread: bands are different conditions and are expected to
differ. A design is CONTESTED when its bands land on opposite sides of the claim -- POSITIVE
somewhere and ANTI somewhere -- because then no single number represents it and quoting one is a
choice the reader cannot see. SPLIT is the weaker case, a decided verdict alongside an undecided one,
which is normal and only worth flagging when the decided one is being cited alone.
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FINDINGS = os.path.join(HERE, "FINDINGS.json")

DECIDED = ("POSITIVE", "ANTI")
UNDECIDED = ("INCONCLUSIVE", "INFORMATIVE_NULL", "SATURATED", "DEGENERATE", "FLOORED", "NO_DATA")


def bands_by_metric(design_id, corpus=None):
    """{metric: [(band_label, verdict, effect), ...]} for one design.

    `corpus` is injectable so this can be shown a KNOWN input -- the same reason defence_score and
    instrument_check take one. An instrument that can only read the live ledger cannot be handed a
    fixture, and one that has never been run against a known answer is not an instrument.
    """
    import meta_lab as M
    src = M._read_jsonl(M.RESULTS) if corpus is None else corpus
    out = {}
    for r in src:
        if r.get("design_id") != design_id:
            continue
        for b in (r.get("bands") or []):
            label = b.get("band") if b.get("band") is not None else b.get("k")
            for c in (b.get("cells") or []):
                if not isinstance(c, dict):
                    continue
                metric = c.get("metric") or c.get("rule")
                verdict = c.get("verdict")
                if not metric or not verdict:
                    continue
                out.setdefault(metric, []).append((label, verdict, c.get("effect")))
    return out


def index_by_design(corpus=None):
    """{design_id: {metric: [(band, verdict, effect), ...]}} in ONE pass over the ledger.

    bands_by_metric() filters the whole corpus by design_id, which is correct and was being called
    once per citation: ~400 citations x 2 checks x 932 rows, with a nested walk over bands and
    cells inside each. That was 397 seconds, 37% of the entire audit.

    The contract of bands_by_metric is unchanged -- it stays the injectable entry point that can be
    handed a fixture, which is why it exists in that shape. This is the bulk path for the caller
    that needs every design at once.
    """
    import meta_lab as M
    src = M._read_jsonl(M.RESULTS) if corpus is None else corpus
    out = {}
    for r in src:
        did = r.get("design_id")
        if not did:
            continue
        table = out.setdefault(did, {})
        for b in (r.get("bands") or []):
            label = b.get("band") if b.get("band") is not None else b.get("k")
            for c in (b.get("cells") or []):
                if not isinstance(c, dict):
                    continue
                metric = c.get("metric") or c.get("rule")
                verdict = c.get("verdict")
                if not metric or not verdict:
                    continue
                table.setdefault(metric, []).append((label, verdict, c.get("effect")))
    return out


def contested(design_id, corpus=None, table=None):
    """Metrics on which this design's bands land on OPPOSITE sides of the claim."""
    bad = []
    for metric, rows in (table if table is not None
                         else bands_by_metric(design_id, corpus)).items():
        verdicts = {v for _l, v, _e in rows}
        if "POSITIVE" in verdicts and "ANTI" in verdicts:
            bad.append({"design": design_id, "metric": metric, "kind": "CONTESTED",
                        "bands": sorted(rows, key=lambda t: str(t[0]))})
    return bad


def split(design_id, corpus=None, table=None):
    """Metrics with a decided verdict in some bands and an undecided one in others."""
    out = []
    for metric, rows in (table if table is not None
                         else bands_by_metric(design_id, corpus)).items():
        verdicts = {v for _l, v, _e in rows}
        if verdicts & set(DECIDED) and verdicts & set(UNDECIDED):
            out.append({"design": design_id, "metric": metric, "kind": "SPLIT",
                        "bands": sorted(rows, key=lambda t: str(t[0]))})
    return out


def audit_findings(path=FINDINGS, verbose=True):
    """Every design cited by a published finding, checked for band disagreement."""
    doc = json.load(io.open(path, encoding="utf-8"))
    items = doc.get("findings") if isinstance(doc, dict) else doc
    cited = []
    for i, f in enumerate(items or []):
        for did in (f.get("design_ids") or []):
            cited.append((i, str(f.get("headline") or "")[:52], did))

    # READ THE LEDGER ONCE. contested() and split() each default to re-reading META_RESULTS.jsonl,
    # and this loop called them per citation: ~400 citations x 2 calls x 0.5s = the 397 seconds
    # this check took, 37% of the entire audit's runtime. The `corpus` parameter already existed
    # for injection -- the caller simply was not using it.
    #
    # An audit nobody runs because it takes twenty minutes is an audit that does not run, and four
    # more checks were added to it today.
    idx_by_design = index_by_design()

    problems, splits = [], []
    for idx, head, did in cited:
        table = idx_by_design.get(did, {})
        for row in contested(did, table=table):
            row.update(finding=idx, headline=head)
            problems.append(row)
        for row in split(did, table=table):
            row.update(finding=idx, headline=head)
            splits.append(row)

    if verbose:
        print("BAND AGREEMENT -- %d design citation(s) across published findings" % len(cited))
        print()
        print("  CONTESTED (a design whose bands land on BOTH sides of the claim): %d" % len(problems))
        for p in problems:
            print("     finding %d  %s" % (p["finding"], p["headline"]))
            print("        %s / %s" % (p["design"], p["metric"]))
            for label, verdict, eff in p["bands"]:
                shown = "%+.4f" % eff if isinstance(eff, (int, float)) else "-"
                print("           band %-4s %-16s %s" % (label, verdict, shown))
        if not problems:
            print("     none -- no published finding rests on a design its own bands contradict")
        print()
        print("  SPLIT (decided in some bands, undecided in others): %d" % len(splits))
        for s in splits[:8]:
            vs = ", ".join("%s=%s" % (l, v) for l, v, _e in s["bands"])
            print("     %-26s %-28s %s" % (s["design"], s["metric"], vs[:70]))
        if len(splits) > 8:
            print("     ... and %d more not shown (%d total)" % (len(splits) - 8, len(splits)))
    return problems, splits


if __name__ == "__main__":
    probs, _ = audit_findings()
    print()
    if probs:
        print("A contested design cannot be cited by a single number. Either report every band, or")
        print("say which band and why that one -- a choice the reader cannot see is not a citation.")
    sys.exit(1 if probs else 0)

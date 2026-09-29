"""Refuse a published finding that does not name the runs it rests on.

WHY THIS EXISTS. On 2026-08-17 a provenance sweep found that all eleven entries in FINDINGS.json
carry `headline, number, detail, caveat, confidence` and nothing else. Not one names a design, a
result, or a run. Separately, 3 of 598 results verified against the graders then on disk.

Those compound. A claim you cannot trace to a run, resting on a run you cannot verify against the
grader that produced it, is not evidence -- it is a number with a good reputation.

THE BACKWARD DIRECTION IS NOT AUTOMATABLE AND THIS FILE DOES NOT PRETEND OTHERWISE. Two attempts to
recover the link by matching cited figures both returned plausible WRONG answers: 0.000 appears in
485 designs, and effect=1.000 in 470 ledger rows with 273 on a single metric, so a matcher "found"
evidence for almost anything. The eleven need a person to attach evidence or to mark them
provenance-unrecoverable. That is a research-integrity decision, not an engineering one.

WHAT IS FIXABLE IS THE FORWARD DIRECTION, and it is nearly free. FINDINGS_LEDGER.jsonl already
records design_id on 100 percent of its 2303 statements across 569 designs. The automatic layer has
the provenance; the curated layer drops it on the way to publication. So a finding written from here
on carries its design ids, and this lint refuses the ones that do not.

THE GRANDFATHER COUNT IS THE MECHANISM. The eleven existing findings are exempt, but the exemption is
a NUMBER that cannot grow. Add a twelfth without provenance and the lint fails. That converts "we
should really fix this someday" into a wall, without demanding that eleven historical claims be
reconstructed before anything else can ship.
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FINDINGS = os.path.join(HERE, "FINDINGS.json")

# Findings published before provenance was required. This number may go DOWN as they are sourced or
# retired; it may never go up. tests/test_findings_lint.py pins that.
GRANDFATHERED = 11

REQUIRED = ("headline", "detail", "confidence")
PROVENANCE = ("design_ids",)


def _retracted():
    """Design ids touched by a retraction, or an empty set if the ledger is unreadable.

    Deliberately includes metric-scoped rows as well as whole-design ones: a finding citing a design
    that has had ANY measurement withdrawn should say so, because the reader cannot tell from the
    citation which metric is meant. _retracted_scopes() below separates the two for reporting.
    """
    try:
        import retractions
        return {row.get("design_id") for row in retractions.load() if row.get("design_id")}
    except Exception:
        return set()          # cannot tell -> do not invent a problem


def _retracted_scopes():
    """(whole-design ids, metric-scoped ids). A withdrawn METRIC is not a withdrawn RUN.

    retractions.py has always recorded the difference -- metric=None means the run should never have
    been produced, a named metric means one measurement from an otherwise sound run is gone. This
    lint flattened them, so a finding citing the metric that SURVIVED was told to acknowledge a
    retraction that does not touch it. A check that asks the wrong question gets skipped.
    """
    try:
        import retractions
        whole, scoped = set(), set()
        for row in retractions.load():
            did = row.get("design_id")
            if not did:
                continue
            (whole if row.get("metric") is None else scoped).add(did)
        return whole, scoped - whole
    except Exception:
        return set(), set()


ACKNOWLEDGES = ("withdraw", "retract", "corrected", "artefact", "artifact")


# A figure with at least two decimals, as the claim fields quote measurements ("~0.80", "0.4807").
_FIGURE = re.compile(r"(?<![\d.])\d?\.\d{2,}")


def _dropped_conditions(it):
    """(field, condition) pairs where a claim field quotes a figure and omits a declared condition.

    WHY DECLARED. No text check can find a condition that is simply absent: finding 1 printed
    "obeys ~0.80" for weeks with the recency condition (the 0.80 holds only when the injection is the
    LAST text; 200 characters later it is 0.000) nowhere in the finding at all, caveat included.
    idea_ratchet Cell 2 measured it: every standing audit catches what a finding prints wrongly and
    none catches what it leaves out. So `conditions` names the phrases a figure is only true under,
    and this makes their absence checkable -- in each claim field that quotes a figure, since that is
    the sentence a reader lifts out. Matching is case-insensitive substring; a condition should be
    written as the shortest phrase that cannot be dropped without changing the claim.
    """
    out = []
    for cond in it.get("conditions") or []:
        c = str(cond).strip().lower()
        if not c:
            continue
        for field in ("headline", "number"):
            text = str(it.get(field) or "")
            if _FIGURE.search(text) and c not in text.lower():
                out.append((field, cond))
    return out


def _items(doc):
    items = doc.get("findings") if isinstance(doc, dict) and "findings" in doc else doc
    if isinstance(items, dict):
        items = list(items.values())
    return items or []


def check(path=FINDINGS, verbose=True):
    """Returns (problems, unsourced_count). A problem is a finding that cannot be published."""
    doc = json.load(io.open(path, encoding="utf-8"))
    items = _items(doc)
    problems, unsourced = [], []

    # BOTH STORES. This programme has two: the meta_lab design library, and 944 standalone
    # RESULT_*.json cards that rigor_audit reads and meta_lab has never heard of. The lint knew only
    # the first, so a finding resting on the standalone track -- sandbagging, spec-gaming, sycophancy,
    # calibration -- could not be sourced at all. It would have had to be published unsourced, which
    # the grandfather ceiling forbids, so the only correct finding was unpublishable.
    retracted = _retracted()
    known = set()
    try:
        import meta_lab as M
        known = {(r.get("id") or r.get("design_id")) for r in M._read_jsonl(M.DESIGNS)}
        known.discard(None)
    except Exception:
        pass                      # library unreadable: check structure, do not invent failures
    try:
        import glob as _glob
        for _p in _glob.glob(os.path.join(HERE, "RESULT_*.json")):
            known.add(os.path.basename(_p))                 # cards are cited by filename
            known.add(os.path.basename(_p)[:-5])            # ...with or without the extension
    except Exception:
        pass

    for i, it in enumerate(items):
        head = str(it.get("headline") or "")[:60]
        for k in REQUIRED:
            if not str(it.get(k) or "").strip():
                problems.append("%d [%s]: missing required field %r" % (i, head, k))
        for field, cond in _dropped_conditions(it):
            problems.append("%d [%s]: its %s quotes a figure without the declared condition %r -- "
                            "a figure without the condition it was measured under is a different claim"
                            % (i, head, field, cond))
        ids = it.get("design_ids")
        if not ids:
            unsourced.append(i)
            continue
        if not isinstance(ids, (list, tuple)) or not all(isinstance(x, str) for x in ids):
            problems.append("%d [%s]: design_ids must be a list of design id strings" % (i, head))
            continue
        if known:
            ghosts = [d for d in ids if d not in known]
            if ghosts:
                problems.append("%d [%s]: names designs not in the library: %s"
                                % (i, head, ", ".join(sorted(ghosts))))

        # RETRACTED EVIDENCE, reported rather than refused. Citing a withdrawn run is legitimate when
        # the finding is ABOUT the withdrawal -- one such finding reads those runs as the seed
        # replicates they turned out to be, and cannot do that without naming them. What must not
        # pass unremarked is a finding LEANING on retracted evidence, so the test is whether it says
        # so anywhere in its own text.
        gone = sorted(set(ids) & retracted)
        if gone:
            text = " ".join(str(it.get(k) or "") for k in
                            ("headline", "number", "detail", "caveat", "confidence")).lower()
            if not any(w in text for w in ACKNOWLEDGES):
                problems.append(
                    "%d [%s]: cites RETRACTED design(s) %s without acknowledging it anywhere in its "
                    "own text. Citing a withdrawn run is fine when the finding is about the "
                    "withdrawal; leaning on one silently is not."
                    % (i, head, ", ".join(gone)))

    if len(unsourced) > GRANDFATHERED:
        problems.append(
            "%d findings carry no design_ids; only the %d published before provenance was required "
            "are exempt. A NEW finding must name the runs it rests on -- the automatic ledger "
            "already records design_id on every statement, so this is a carry-through rather than "
            "a reconstruction." % (len(unsourced), GRANDFATHERED))

    if verbose:
        print("FINDINGS LINT -- %d finding(s)" % len(items))
        print("  unsourced: %d (grandfathered ceiling %d)" % (len(unsourced), GRANDFATHERED))
        cites_gone = sum(1 for it in items if set(it.get("design_ids") or []) & retracted)
        whole, scoped = _retracted_scopes()
        print("  citing retracted evidence: %d finding(s) -- %d design(s) withdrawn whole, "
              "%d with a single metric withdrawn" % (cites_gone, len(whole), len(scoped)))
        for p in problems:
            print("  FAIL  %s" % p)
        if not problems:
            print("  ok -- no finding is published without provenance it was required to carry")
    return problems, len(unsourced)


if __name__ == "__main__":
    probs, _ = check()
    print()
    print("The eleven grandfathered findings still need a HUMAN decision: attach evidence, or mark")
    print("them provenance-unrecoverable. Matching their cited numbers does not work and has been")
    print("shown twice not to work -- see findings_provenance.py.")
    sys.exit(1 if probs else 0)

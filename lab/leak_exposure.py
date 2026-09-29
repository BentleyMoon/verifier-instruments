# -*- coding: utf-8 -*-
"""Which findings still rest on cells measured through a known serving defect?

THE PROBLEM THIS SOLVES IS NOT FINDING THE DEFECT. That was `parser_reads_model.py`, and the fix was
`backend.generate` stripping gemma4:12b's leaked channel block on 2026-09-17. The problem is what
comes after: 140 designs were run while a monitor could see those tokens, 39 of them cited by standing
findings, and each re-serve lands hours apart in a queue no person is watching. The failure mode is
mundane and it is the one this programme keeps having -- a correction that is ordered, runs, and then
sits in the ledger while the published text keeps the number it falsified.

So this asks two questions a person would otherwise have to remember to ask:

  UNACCOUNTED   a finding cites an affected cell that is neither re-served nor queued. Nobody has
                decided anything about it, and silence looks the same as a clean bill of health.

  STALE         the re-serve LANDED, it moved by more than MEOI, and the finding's own text still
                quotes the original figure. This is the expensive one. It means a correction was
                paid for in GPU hours and never reached the page a reader sees.

Both close by themselves: queue the cell and UNACCOUNTED goes; edit the text and STALE goes. A check
that goes quiet when the work is done is one people leave switched on.

WHAT STALE CANNOT SEE, and it is not a small gap. It matches the ORIGINAL RATE as printed. A finding
that quotes a figure DERIVED from the affected cell -- a mean over three cells, a self-minus-cross
delta -- prints a number that appears nowhere in the ledger, so no string match can find it. Finding
25 is exactly that case: re-serving one of qwen2.5-coder:7b's three cross cells moved its published
self-minus-cross delta from +0.4245 to +0.1393, and this check stayed silent throughout, because the
text never prints the 0.9102 that changed. UNACCOUNTED still covers those findings -- they cite the
design -- so the cell cannot be forgotten; but the arithmetic must be redone by a person who knows
the recipe, and `derivation` fields are where that recipe lives.

    python leak_exposure.py
"""
import io
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FINDINGS = os.path.join(HERE, "FINDINGS.json")
# ASSERTIVE FIELDS ONLY, and this cost a false alarm to learn. The first version read `caveat` too and
# flagged all five of finding 31's corrected cells: its caveat prints "0.7227 -> 0.0293" precisely
# BECAUSE the figure was corrected. A check that fires on a finding for documenting its own correction
# punishes the behaviour it exists to encourage. The caveat is where a retraction lives; the number,
# detail and headline are where a reader takes a figure away.
QUOTING_FIELDS = ("number", "detail", "headline")

# A serving defect is (whose output, when it was fixed). Add an entry when the next one is found; the
# machinery below is not specific to this one.
# "actor" is historical: the entry names the model whose OUTPUT carried the defect, in whatever role it
# held -- actor, monitor, extractor or peer (see _roles).
DEFECTS = [{
    "actor": "gemma4:12b",
    "fixed_at": "2026-09-17T00:38",
    "what": "leaked gemma4 chat-template tokens shown to whoever read the answer",
    "commit": "7e6a3c3f",
}]

MEOI = 0.05
RATES = ("flag_clean", "flag_inj")       # what a reader-side defect moves


RESERVE_SUFFIX = "_rs"


def superseded(ids):
    """Of *ids*, the ones a re-serve replaces: X is superseded when X_rs is present.

    WHY ANY ANALYSIS NEEDS THIS. A re-serve keeps its parent's id as a prefix -- ox4_g12_by_cd7 and
    ox4_g12_by_cd7_rs -- so every selector in this repo that globs by prefix picks up BOTH, and the
    two land in the same (actor, verifier) slot. analysis_self_vs_cross then averaged them:
    mean(-0.0312, +0.8242), one cell measured through a serving defect and one measured without it,
    pooled into a figure a finding quotes. Every re-serve's own repair_path says never to average the
    two, and the code that reads them did it by default.

    The rule is not "prefer the newer timestamp" -- a legitimate seed replicate is also newer and
    must still pool. It is specifically that a re-serve SUPERSEDES the run it re-serves, because the
    parent measured a defect. One line, in one place, so no selector has to remember it.
    """
    ids = set(ids)
    return {i for i in ids if i + RESERVE_SUFFIX in ids}


def _load(path, fallback):
    try:
        return json.load(io.open(path, encoding="utf-8"))
    except Exception:
        return fallback


def _values_of(design_ids, results_by_design, designs):
    """Every numeric value in the cited designs a re-serve does NOT supersede.

    The set a figure in the text might legitimately be quoting instead of the affected cell.
    """
    # SUPERSEDED MEANS REPLACED, NOT ORDERED. superseded() works on names, which is what an analysis
    # needs -- it must stop pooling the moment the replacement exists. Here the question is different:
    # what value might this sentence be quoting TODAY. A re-serve that is queued and unrun has
    # replaced nothing, and dropping its parent here made the check flag finding 25 for printing a
    # figure that is still the best value in the ledger.
    live = {d for d in design_ids if not (results_by_design.get(d + RESERVE_SUFFIX))}
    out = set()
    for did in live:
        for r in results_by_design.get(did) or []:
            for b in r.get("bands") or []:
                for c in b.get("cells") or []:
                    for k in ("effect", "ci_low", "ci_high"):
                        if isinstance(c.get(k), (int, float)):
                            out.add(float(c[k]))
                    for v in (c.get("rates") or {}).values():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            out.add(float(v))
                for v in (b.get("rates") or {}).values():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        out.add(float(v))
    return out


def _near(value, pool):
    """True when *value* equals a pool member at 3 or 4 decimal places, the precision findings quote."""
    return any(round(v, 4) == round(value, 4) or round(v, 3) == round(value, 3) for v in pool)


def _roles(design, model):
    """Every role in which *model* produced text that something downstream read.

    SELF-MONITORED CELLS ARE IN SCOPE (2026-09-22). This check used to skip them -- "a model reading
    its own output has a neutral arm" -- and no measurement stood behind that sentence. Fourteen
    self-monitored cells had been re-served anyway, for other reasons, and nine moved beyond two-run
    binomial noise: dec_override_only_gemma's false alarms went 0.9585 -> 0.0059, dec_verdict_gemma's
    1.0000 -> 0.1632, cvt_g12_s1's covert detection 0.0000 -> 0.7129. A neutral arm controls for a
    MAIN effect of the leak. It cannot control for the leak interacting with text aimed at the
    monitor, which is what finding 33 is about, and it cannot help a parser that reads the start of
    the reply. The exemption hid 34 cited cells across 13 findings (idea_ratchet Cell 0). A monitor
    reading someone else's work counts too: its OWN reply carried the tokens to our parsers.
    """
    roles = set()
    if design.get("model") == model:
        roles.add("actor")
    if model in [design.get("verifier")] + list(design.get("verifiers") or []):
        roles.add("monitor")
    if design.get("extractor_model") == model:
        roles.add("extractor")
    if design.get("peer_model") == model:
        roles.add("peer")
    return roles


def _band_rates(rows):
    """Band-0 proportions of the most recent result: the comparison for runners with no oversight cell."""
    for r in reversed(rows or []):
        for b in r.get("bands") or []:
            return {k: float(v) for k, v in (b.get("rates") or {}).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0}
    return {}


def _headline_cell(rows):
    """The oversight cell of the most recent result for a design, or None."""
    for r in reversed(rows):
        for b in r.get("bands") or []:
            for c in b.get("cells") or []:
                if c.get("metric") == "oversight_discrimination":
                    return c
    return None


def _effects(rows):
    """Every cell's effect in every band of the most recent result, plus the rates of cells other than
    the oversight cell (whose two rates RATES already compares).

    THE OVERSIGHT CELL WAS NEVER THE ONLY THING A FINDING QUOTES (2026-09-23). This check compared
    flag_clean and flag_inj and nothing else, so cnf_kdg_gemma's pre-registered, CONFIRMED
    knowing_doing_gap went 0.9297 -> 0.5566 on re-serve and was reported "re-served and clear": its
    flag_inj moved too, but the findings print the gap, not the rate. Four findings quoted the
    falsified 0.9297. The one kdg cell the old check did catch (kdgsub_gemma_lookup) it caught by
    coincidence -- that cell's flag_inj happens to equal its gap to four places.
    """
    out = {}
    for r in reversed(rows or []):
        for bi, b in enumerate(r.get("bands") or []):
            for c in b.get("cells") or []:
                m = c.get("metric")
                if isinstance(c.get("effect"), (int, float)):
                    out["%s effect (band %d)" % (m, bi)] = float(c["effect"])
                if m == "oversight_discrimination":
                    continue
                for k, v in (c.get("rates") or {}).items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0:
                        out["%s.%s (band %d)" % (m, k, bi)] = float(v)
        return out
    return out


def collect(findings=None, designs=None, results=None, queue=None):
    import meta_lab as M
    findings = findings if findings is not None else _load(FINDINGS, {}).get("findings", [])
    designs = designs if designs is not None else M._read_jsonl(M.DESIGNS)
    results = results if results is not None else M._read_jsonl(M.RESULTS)
    queue = queue if queue is not None else M._read_jsonl(M.QUEUE)

    D = {d.get("id"): d for d in designs}
    R = {}
    for r in results:
        R.setdefault(r.get("design_id"), []).append(r)
    pending = {q.get("id") for q in queue if q.get("status") in ("queued", "running")}

    unaccounted, stale, cleared, waiting = [], [], [], []
    for defect in DEFECTS:
        for i, f in enumerate(findings):
            text = " ".join(str(f.get(k) or "") for k in QUOTING_FIELDS)
            for did in f.get("design_ids") or []:
                d = D.get(did) or {}
                roles = _roles(d, defect["actor"])
                if not roles or did.endswith("_rs"):
                    continue
                verifier = d.get("verifier") or (d.get("verifiers") or [None])[0]
                rows = R.get(did) or []
                if not rows or max(r.get("ts") or "" for r in rows) >= defect["fixed_at"]:
                    continue
                rid = did + "_rs"
                after_rows = R.get(rid) or []
                item = {"finding": i + 1, "design": did, "monitor": verifier, "roles": sorted(roles),
                        "headline": str(f.get("headline") or "")[:56]}
                if not after_rows:
                    (waiting if (rid in pending or rid in D) else unaccounted).append(item)
                    continue
                before, after = _headline_cell(rows), _headline_cell(after_rows)
                if before is not None and after is not None:
                    pairs = [(k, before["rates"].get(k), after["rates"].get(k)) for k in RATES]
                else:
                    # A runner with no oversight cell (injection_subtlety, injection_resistance,
                    # debate_truth) used to be skipped before its re-serve was even looked for, so an
                    # affected cell there could be neither UNACCOUNTED nor cleared. Its band rates are
                    # the comparison instead.
                    b0, a0 = _band_rates(rows), _band_rates(after_rows)
                    pairs = [(k, b0[k], a0[k]) for k in sorted(set(b0) & set(a0))]
                eb, ea = _effects(rows), _effects(after_rows)
                pairs += [(k, eb[k], ea[k]) for k in sorted(set(eb) & set(ea))]
                moved = [(k, b, a) for k, b, a in pairs
                         if isinstance(b, (int, float)) and isinstance(a, (int, float)) and abs(b - a) > MEOI]
                item["moved"] = moved
                if not moved:
                    cleared.append(item)
                    continue
                # Does the finding still print a figure the re-serve falsified, WITHOUT the value that
                # replaced it? Printing both is how a correction reads -- "0.7227 -> 0.0293" is the
                # finding doing its job, not a stale number.
                def _in(v):
                    return bool(re.search(r"(?<!\d)%.4f" % v, text) or re.search(r"(?<!\d)%.3f" % v, text))

                # AMBIGUITY MEANS SILENCE, because this one gates. A bare figure carries no name: on
                # 2026-09-17 this flagged finding 25 for printing 0.6172, which is ox4_g12_by_dol's
                # flag_inj AND, quite separately, the effect of cxs_hm3_fact -- the cell the sentence
                # is actually about. The finding was right and the check was noise. So if the figure
                # also appears in a cited cell that no re-serve supersedes, the text may be quoting
                # that one, and a gate must not fail on may.
                elsewhere = _values_of(f.get("design_ids") or [], R, D)

                # The same bare figure can be two metrics of one cell: kdgcue_gemma_open's posthoc rate
                # and its oversight effect were both 0.7324 and moved to 0.9570 and 0.8887. A text that
                # prints "0.7324 -> 0.9570" has corrected the figure it printed, so any same-valued
                # metric whose replacement it prints counts.
                def _corrected(b):
                    return any(_in(a2) for _k, b2, a2 in moved if round(b2, 4) == round(b, 4))

                quoted = [(k, b, a) for k, b, a in moved
                          if _in(b) and not _corrected(b) and not _near(b, elsewhere)]
                item["quoted"] = quoted
                (stale if quoted else cleared).append(item)
    return unaccounted, stale, cleared, waiting


_CITE = re.compile(r"\b[Ff]indings?\s+(\d+(?:\s*(?:,|and|&)\s*\d+)*)")


def inherited(findings, unaccounted, waiting):
    """Findings that rest on ANOTHER finding whose cells are still unaccounted or awaiting a re-serve.

    WHY. `design_ids` records what a finding measured, not what it leans on. Finding 25 judges its
    self-minus-cross deltas against finding 26's run-to-run floor, and that floor was measured on
    cells that went through the leak; 25 cites none of them, so this check called 25 settled while
    the yardstick under it waited on sixteen re-serves (idea_ratchet Cell 0). Two kinds of edge:
      declared  `rests_on_findings` in FINDINGS.json -- the author says the number depends on it
      cited     the finding's own text names the other finding ("finding 26") -- an upper bound,
                since a mention is not always a dependency
    Informational, never gating: an inherited wait closes when the parent's re-serves land.
    """
    pending = defaultdict(set)
    for item in list(unaccounted) + list(waiting):
        pending[item["finding"]].add(item["design"])
    out = []
    for i, f in enumerate(findings):
        me = i + 1
        declared = {int(x) for x in (f.get("rests_on_findings") or [])}
        text = " ".join(str(v) for v in f.values() if isinstance(v, str))
        cited = set()
        for m in _CITE.finditer(text):
            cited.update(int(x) for x in re.findall(r"\d+", m.group(1)))
        for kind, parents in (("declared", declared), ("cited", cited - declared)):
            for p in sorted(parents):
                if p != me and pending.get(p):
                    out.append({"finding": me, "via": p, "kind": kind, "pending": len(pending[p])})
    return out


def main():
    unaccounted, stale, cleared, waiting = collect()
    print("LEAK EXPOSURE -- findings resting on cells a known serving defect could have moved")
    for d in DEFECTS:
        print("  %s, fixed %s (%s): %s" % (d["actor"], d["fixed_at"], d["commit"], d["what"]))
    print()
    print("  re-served and clear          %d" % len(cleared))
    print("  re-serve queued or running   %d" % len(waiting))
    for w in waiting:
        print("     finding %-3d %-24s monitor %s" % (w["finding"], w["design"], w["monitor"]))
    if unaccounted:
        print()
        print("  UNACCOUNTED: %d cited cell(s) neither re-served nor queued" % len(unaccounted))
        for u in unaccounted:
            print("     finding %-3d %-24s monitor %-20s %s" % (u["finding"], u["design"], u["monitor"], u["headline"]))
    if stale:
        print()
        print("  STALE: %d finding(s) still print a figure their own re-serve falsified" % len(stale))
        for s in stale:
            for k, b, a in s["quoted"]:
                print("     finding %-3d %-24s %s %.4f -> %.4f, and the text still says %.4f"
                      % (s["finding"], s["design"], k, b, a, b))
    inh = inherited(_load(FINDINGS, {}).get("findings", []), unaccounted, waiting)
    if inh:
        print()
        print("  INHERITED (informational): findings resting on a finding whose re-serves are pending")
        for x in inh:
            print("     finding %-3d via finding %-3d %-8s (%d pending)" % (x["finding"], x["via"], x["kind"], x["pending"]))
    print()
    if not unaccounted and not stale:
        print("  nothing outstanding: every cited cell is re-served, queued, or unmoved")
    return 1 if (unaccounted or stale) else 0


if __name__ == "__main__":
    sys.exit(main())

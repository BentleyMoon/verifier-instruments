# -*- coding: utf-8 -*-
"""Every precise figure a finding quotes must appear in the evidence it cites.

THE GAP THIS FILLS. Four checks guard published findings and none of them reads the numbers:

    findings_lint    the finding names design ids, and those designs exist
    verdict_lint     no strength word outranks the verdict the ledger holds
    band_agreement   the cited design's own bands do not contradict each other
    regrade_audit    every stored number re-derives from its stored per-unit values

A finding could pass all four while quoting +0.8574 for a design whose cells say +0.4571. The
provenance is real, the verdict is real, the bands agree, the number re-derives -- and the sentence
is still wrong, because the number and the citation were never compared to each other.

That is not hypothetical here. A number quoted from band 3 of four while bands 1 and 2 read ANTI got
into the compass, a commit, a caveat and the artifact, and was only caught by reading the design by
hand. band_agreement now catches the contested-bands version. This catches the plainer one: a figure
that is simply not in the cited evidence at all.

WHAT IS CHECKED, AND WHY NOT EVERYTHING. Only figures with THREE OR MORE decimal places. Those are
quoted cell values -- +0.8574, 0.094, -0.4863 -- and they should be findable. Numbers like 0.05,
0.80, 19 or 512 are thresholds, floors, counts and sample sizes: legitimately not cell effects, and
flagging them would bury the real signal under noise nobody reads. The lints in this repo have twice
produced false alarms by scanning more than they could interpret, so this one scans less.

Matched against effects, CI bounds and rates in every cited design, at the precision the figure was
written to -- the same rule regrade_audit uses, because a bound stored as 0.0469 and quoted as 0.047
is the same number reported to fewer places.
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FINDINGS = os.path.join(HERE, "FINDINGS.json")

# 3+ decimals: precise enough to be a quoted measurement rather than a threshold or a count.
#
# A sign only counts as a sign when a digit does not sit right before it. Found 2026-09-17: the
# prose "obeys 0.144-0.197 of injections" was read as the figures 0.144 and -0.197, and a negative
# obedience rate matches nothing, so both ends of every hyphenated range were unmatchable. Four
# correct rates had been exempted as "derived" to get them past a check that could never pass them.
FIGURE = re.compile(r"(?<!\d)[-+]?\d*\.\d{3,}")

# Fields of a finding whose prose may quote measurements.
QUOTING_FIELDS = ("number", "detail", "headline")

# A finding may declare figures that are COMPUTED from measured ones -- a difference between two
# cells, a ratio -- via a "derived_figures" list. Those are exempt from the ledger match, because
# they are arithmetic on cited numbers rather than numbers in their own right. The exemption is a
# declaration a reader can check, not a heuristic the tool applies on its own.


def _values(design_id, corpus=None):
    """Every number a citation could legitimately be quoting, from one design."""
    import meta_lab as M
    src = M._read_jsonl(M.RESULTS) if corpus is None else corpus
    out = set()
    for r in src:
        if r.get("design_id") != design_id:
            continue
        for b in (r.get("bands") or []):
            for k, v in (b.get("rates") or {}).items():
                if isinstance(v, (int, float)):
                    out.add(float(v))
            for c in (b.get("cells") or []):
                if not isinstance(c, dict):
                    continue
                for k in ("effect", "ci_low", "ci_high", "mde", "sd"):
                    v = c.get(k)
                    if isinstance(v, (int, float)):
                        out.add(float(v))
                for k, v in c.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        out.add(float(v))
                # PER-CELL RATES, added 2026-09-17. A cell's own rates block was skipped because it
                # is a dict, and the band-level block beside it is not a copy: on prb_q35_prov the
                # band reads monitor_responded 1.0 while two of its cells read 0.1445. So a finding
                # quoting a cell's monitor_responded or obey rate correctly could not be matched,
                # and the only way through was a derived_figures exemption -- which then stopped
                # anything from checking that figure at all. The blind spot did not just miss
                # figures; it pushed correct ones out of the check.
                for k, v in (c.get("rates") or {}).items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        out.add(float(v))
    return out


def _matches(figure, values):
    """True when *figure* equals some value rounded to the precision the figure was written to.

    A bound stored as 0.046875 and quoted as 0.0469 is the same number reported to fewer places;
    demanding exactness would flag every correctly-rounded citation in the file.
    """
    text = figure.lstrip("+")
    places = len(text.split(".")[1])
    target = float(text)
    return any(abs(round(v, places) - target) <= 1e-12 for v in values)


def check(path=FINDINGS, verbose=True):
    doc = json.load(io.open(path, encoding="utf-8"))
    items = doc.get("findings") if isinstance(doc, dict) else doc
    unfound, unrecipe, checked, skipped = [], [], 0, 0

    # ONE PASS. _values() defaults to re-reading META_RESULTS.jsonl and was being called once per
    # cited design -- roughly 400 full ledger reads per run. Indexing it here costs one read and
    # leaves _values' contract untouched, since it stays the injectable entry point a fixture can
    # be handed. band_agreement had the identical shape and went from 397s to 0.3s on this fix.
    import meta_lab as _M
    _corpus = _M._read_jsonl(_M.RESULTS)
    _by_design = {}
    for _r in _corpus:
        _by_design.setdefault(_r.get("design_id"), []).append(_r)

    for i, f in enumerate(items or []):
        # A DECLARATION, never a default. Some figures are computed from measured ones -- a
        # difference between two cells, a group mean over hundreds -- and will never appear in the
        # ledger. derived_figures waives the match for those, and that waiver used to rest on the
        # finding's own say-so: a figure nobody could reproduce looked exactly like one computed
        # correctly. So the exemption must name HOW -- a "derivation" field, ideally a script a
        # reader can run.
        #
        # This sits ABOVE the skips on purpose. Written below them it only ran on findings that
        # were being fully checked anyway, so a finding citing evidence with no numeric cells could
        # exempt as many figures as it liked and never be asked. The requirement does not depend on
        # what a finding cites.
        if (f.get("derived_figures") or []) and not str(f.get("derivation") or "").strip():
            unrecipe.append({"finding": i, "headline": str(f.get("headline") or "")[:56],
                             "n": len(f["derived_figures"])})

        ids = f.get("design_ids") or []
        if not ids:
            skipped += 1          # grandfathered: no provenance to check against
            continue
        values = set()
        for did in ids:
            values |= _values(did, corpus=_by_design.get(did, []))
        if not values:
            skipped += 1          # cited designs have no numeric cells (e.g. RESULT_*.json cards)
            continue
        text = " ".join(str(f.get(k) or "") for k in QUOTING_FIELDS)
        # The checker never guesses which figures may be missing -- guessing that is how a check
        # stops being one. Signs are stripped from both sides: the prose writes "-0.7633" and a
        # declaration reads "0.7633", and matching only "+" made negative figures undeclarable.
        declared = {str(x).lstrip("+-") for x in (f.get("derived_figures") or [])}
        for fig in sorted(set(FIGURE.findall(text))):
            if fig.lstrip("+-") in declared:
                continue
            checked += 1
            if not _matches(fig, values):
                unfound.append({"finding": i, "headline": str(f.get("headline") or "")[:56],
                                "figure": fig, "cited": ids})

    if verbose:
        print("QUOTED NUMBERS -- %d precise figure(s) checked against their own citations" % checked)
        print("  findings skipped (no provenance, or citations carry no numeric cells): %d" % skipped)
        if unrecipe:
            print("  DERIVED WITHOUT A RECIPE: %d finding(s) waive the ledger match without saying "
                  "how the figures were computed" % len(unrecipe))
            for u in unrecipe:
                print("     finding %d  %s  (%d exempt figure(s))"
                      % (u["finding"], u["headline"], u["n"]))
        print()
        if unfound:
            print("  NOT FOUND in the cited evidence: %d" % len(unfound))
            for u in unfound:
                print("     finding %d  %s" % (u["finding"], u["headline"]))
                print("        %s   cited: %s" % (u["figure"], ", ".join(u["cited"][:4])))
        else:
            print("  every precise figure appears in the evidence its finding names")
    return unfound, checked, unrecipe


if __name__ == "__main__":
    bad, _n, unrecipe = check()
    print()
    if bad:
        print("A figure that is not in the cited evidence is either quoted from somewhere else --")
        print("in which case cite that -- or wrong. Both need a person, not a default.")
    if unrecipe:
        print("A figure exempted as derived, with no recipe, is not checked by anything. Name how")
        print("it was computed -- ideally a script a reader can run -- or drop the exemption.")
    sys.exit(1 if (bad or unrecipe) else 0)

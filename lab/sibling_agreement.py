# -*- coding: utf-8 -*-
"""A finding must face the runs it did not cite, when those runs measure the same thing.

THE INCIDENT, 2026-08-20, second of the day and one level up from the first.

band_agreement was built that morning after a number was quoted from band 3 of a design whose
bands 1 and 2 said the opposite. It looks WITHIN a design. Hours later the same mistake happened
BETWEEN designs and it sailed straight past every check in the repo.

"The knowing-doing gap reverses across substrates" was published on the strength of one model:

    qwen3.5:9b    +0.2656 -> -0.4863    reverses
    gemma4:12b    +0.9297 -> +0.9883    holds          <- not cited
    hermes3:8b    -0.5703 -> -0.6289    holds          <- not cited

Two of three models hold their value across exactly the same substrate change. The one that reverses
has the smallest effect of the three, which inverts the conclusion that was drawn from it: a finding
titled "the small mitigation survived what the LARGER capability claim did not" rests on a corpus
where the larger capability claim survived and the smaller one did not.

Every existing instrument passed it. The designs exist, the verdicts are real, the bands agree within
each design, every number re-derives, and every quoted figure is now traceable to a cited run. The
defect is in what was NOT cited -- a sibling that measures the same metric under the same axis and
disagrees -- and nothing looked there.

WHAT COUNTS AS A SIBLING, after a false start. The first version called any run measuring the same
METRIC a sibling, and drowned: oversight_discrimination appears in doubt-option experiments and in
recompute-mitigation experiments alike, and those are different claims that share a column name. Ten
of eleven hits were noise, which is how a check becomes something people scroll past.

A sibling is a run in the same DECLARED replication family -- linked by `replicates`, directly or
through a chain -- measuring the same canonical metric. That is a relationship the author asserted,
not one inferred from a name collision. kdg_gemma_camo is the root of the knowing-doing family and
all three substrate arms name it, which is exactly why this case is catchable.

COVERAGE IS 8 PERCENT AND THAT IS SAID OUT LOUD: 69 of 823 designs declare `replicates`. Outside
those, this check sees nothing, and a silent zero would be worse than no check. The report prints
how many findings it could actually examine.

WHAT COUNTS AS DISAGREEMENT. Sign. Magnitudes vary across models and substrates by design and
flagging that would fire constantly; a sibling whose effect points the OTHER WAY from every cited one
is the case where a finding's direction is a selection rather than a result.
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FINDINGS = os.path.join(HERE, "FINDINGS.json")
NOISE = 0.05          # MEOI: an effect inside the band has no direction worth disagreeing about


_DESIGN_CACHE = None


def _design(did):
    """One design spec, from a snapshot of the library taken once per process.

    This called meta_lab.get_design() directly, which re-reads META_DESIGNS.jsonl every time --
    and _family_root() calls it recursively for every design in every metric table, once per
    published finding. That was 1,824 ledger reads per run.

    A SNAPSHOT is also the more correct thing here, not just the faster one: this check compares a
    finding against its siblings, and a design library that shifts under it mid-pass could have it
    reading two different versions of the same family. An audit should see one state of the world.
    """
    global _DESIGN_CACHE
    if _DESIGN_CACHE is None:
        import meta_lab as M
        _DESIGN_CACHE = {}
        for d in M._read_jsonl(M.DESIGNS):
            if d.get("id"):
                _DESIGN_CACHE[d["id"]] = d      # last write wins, as get_design does
    return _DESIGN_CACHE.get(did)


def _family_root(did, seen=None):
    """The design at the end of this one's `replicates` chain -- the claim it and its siblings share.

    Cycle-guarded: a mutual pair would otherwise recurse forever, and a malformed chain must degrade
    to "its own root" rather than take the process down.
    """
    seen = seen or set()
    if did in seen:
        return did
    seen.add(did)
    spec = _design(did) or {}
    parent = spec.get("replicates")
    if not parent or not isinstance(parent, str) or parent == did:
        return did
    return _family_root(parent, seen)


def _by_metric(corpus=None):
    """{metric: [(design_id, effect, verdict), ...]} across every graded cell on disk."""
    import meta_lab as M
    src = M._read_jsonl(M.RESULTS) if corpus is None else corpus
    out = {}
    for r in src:
        did = r.get("design_id")
        for b in (r.get("bands") or []):
            for c in (b.get("cells") or []):
                if not isinstance(c, dict):
                    continue
                metric, eff = c.get("metric"), c.get("effect")
                if not metric or not isinstance(eff, (int, float)):
                    continue
                out.setdefault(metric, []).append((did, float(eff), c.get("verdict")))
    return out


def _sign(x):
    return 0 if abs(x) < NOISE else (1 if x > 0 else -1)


def unconsidered(path=FINDINGS, verbose=True, corpus=None):
    """Siblings that measure a cited metric, disagree in sign with every cited run, and are not cited."""
    doc = json.load(io.open(path, encoding="utf-8"))
    items = doc.get("findings") if isinstance(doc, dict) else doc
    table = _by_metric(corpus)
    hits = []

    for i, f in enumerate(items or []):
        cited = set(f.get("design_ids") or [])
        if not cited:
            continue
        # Metrics this finding actually rests on: the ones its cited designs DECLARE as canonical,
        # not every metric they happen to emit.
        #
        # The first version used every emitted metric and drowned in false positives -- a doubt-option
        # finding was flagged against correctness_discrimination, injection_sensitivity and three
        # others, because the runs that measure doubt also emit those. Ten of eleven hits were noise,
        # which is how a check becomes something people scroll past. Third time in one day that an
        # instrument here scanned more than it could interpret; the fix is the same each time, which
        # is to read the declaration instead of guessing from what happens to be present.
        metrics = set()
        for did in cited:
            spec = _design(did) or {}
            metrics |= {m for m in (spec.get("canonical_metrics") or []) if m in table}
        acknowledged = {str(x) for x in (f.get("acknowledged_siblings") or [])}
        for metric in sorted(metrics):
            rows = table[metric]
            mine = [(d, e) for d, e, _v in rows if d in cited]
            signs = {_sign(e) for _d, e in mine} - {0}
            if not signs:
                continue                       # nothing directional cited for this metric
            fams = {_family_root(d) for d, _e in mine}
            others = [(d, e) for d, e, _v in rows
                      if d not in cited and d not in acknowledged
                      and _sign(e) not in signs | {0}
                      and _family_root(d) in fams]
            if others:
                hits.append({"finding": i, "headline": str(f.get("headline") or "")[:54],
                             "metric": metric, "cited_signs": sorted(signs),
                             "cited": sorted({d for d, _e in mine}),
                             "disagreeing": sorted(others, key=lambda t: -abs(t[1]))})

    if verbose:
        fam_ct = sum(1 for f in (items or [])
                     if any(_family_root(d) != d for d in (f.get("design_ids") or [])))
        print("SIBLING AGREEMENT -- same declared replication family, same metric, opposite sign")
        print("  findings with any cited design in a declared family: %d of %d"
              % (fam_ct, len(items or [])))
        print("  (69 of 823 designs declare `replicates`; outside those this check sees nothing)")
        print()
        if not hits:
            print("  none -- every finding's direction is shared by every run measuring its metrics")
        for h in hits:
            print("  finding %d  %s" % (h["finding"], h["headline"]))
            print("     metric %s  (cited runs point %s)" % (h["metric"], h["cited_signs"]))
            for d, e in h["disagreeing"][:6]:
                print("        %-26s %+.4f   not cited" % (d, e))
            if len(h["disagreeing"]) > 6:
                print("        ... and %d more not shown (%d total)"
                      % (len(h["disagreeing"]) - 6, len(h["disagreeing"])))
    return hits


if __name__ == "__main__":
    bad = unconsidered()
    print()
    if bad:
        print("A sibling pointing the other way is not automatically a refutation -- different")
        print("models and substrates legitimately differ. It is a run the finding has to face:")
        print("cite it and say why it differs, or list it in `acknowledged_siblings` having done so.")
    sys.exit(1 if bad else 0)

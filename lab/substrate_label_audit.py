# -*- coding: utf-8 -*-
"""Designs whose substrate LABEL disagrees with the substrate the runner will actually use.

run_injection_subtlety and run_injection_resistance select their task family through
meta_lab._tasks_for, which dispatches on `substrate_kind`. `substrate` is a free-text label that no
runner reads. A design that sets one and not the other runs the DEFAULT task family while its title,
claim and every downstream table say otherwise.

THIS HAS COST A PUBLISHED FINDING ONCE AND A REPORTED RESULT TWICE.

  finding 21   WITHDRAWN -- "the small mitigation's four substrates were one substrate", and its
               detail records the same shape: "on arithmetic and the four figures are seed
               replicates. The discriminator never discriminated anything."

  2026-08-29   sbt_moe_lookup, sbt_moe_code, sbt_q35_lookup, sbt_q35_code were cloned from
               arithmetic parents with `substrate` changed and substrate_kind left at 'arith'. All
               four ran arithmetic. They are identical to their parents on every field the runner
               reads, differing only in prose and in the unread label, so they are byte-identical
               replicates -- and were reported as evidence that two models are flat across
               substrates.

The runner even warns about it in its own source: "This called _injection_tasks directly and ignored
substrate_kind -- the same latent trap that cost injection_resistance a published finding on
2026-08-21."

WHY THE EXISTING GUARDS MISS IT. queue_prune_scan fingerprints 60 behavioural fields INCLUDING
`substrate`, so two designs identical to the runner look distinct to it. runner_reads_audit reports
flags a runner never reads, and `substrate` is set on almost every design in the corpus, so it
cannot flag it without flagging everything. Neither is wrong; the divergence needs its own check.

WHAT COUNTS AS A DIVERGENCE. Only designs on a runner that DISPATCHES on substrate_kind, where the
label names a task family the switch knows about and substrate_kind names a different one. A prose
label like "injection resistance - family contrast" is not a claim about the task family and is not
flagged; those are the field-conflation problem, which audit_all already reports separately.
"""
import sys
from collections import defaultdict

sys.dont_write_bytecode = True

import meta_lab as M

# The families _tasks_for knows. A label outside this set is prose, not a substrate claim.
KNOWN = ("arith", "lookup", "lookup_hard", "checkone", "checkone_hard", "checkone_ladder",
         "code", "fact", "needle")
# Runners that route through _tasks_for.
DISPATCHING = ("injection_subtlety", "injection_resistance", "injection_oversight",
               "injection_provenance")


def divergent():
    """[{design, runner, label, kind, has_result}] where the label contradicts the dispatch."""
    done = {r.get("design_id") for r in M._read_jsonl(M.RESULTS)}
    out = []
    for d in M._read_jsonl(M.DESIGNS):
        if d.get("runner") not in DISPATCHING:
            continue
        label = str(d.get("substrate") or "").strip().lower()
        kind = str(d.get("substrate_kind") or "").strip().lower()
        if label not in KNOWN:
            continue                      # prose label: not a claim about the task family
        effective = kind or "arith"       # _tasks_for defaults to arithmetic
        if label == effective:
            continue
        out.append({"design": d.get("id"), "runner": d.get("runner"), "label": label,
                    "kind": kind or "(unset -> arith)", "has_result": d.get("id") in done})
    return out


def check(verbose=True):
    rows = divergent()
    if verbose:
        print("SUBSTRATE LABEL AUDIT -- does the label match the substrate the runner will use?")
        print("  _tasks_for dispatches on substrate_kind. `substrate` is a label no runner reads.")
        print()
        if not rows:
            print("  every substrate label agrees with its dispatch")
        else:
            print("  %d design(s) claim a task family they do not run:" % len(rows))
            for r in rows:
                print("    %-22s %-20s label=%-8s kind=%-18s %s"
                      % (r["design"], r["runner"], r["label"], r["kind"],
                         "HAS RESULTS" if r["has_result"] else "not yet run"))
            withres = sum(1 for r in rows if r["has_result"])
            print()
            print("  %d of them already carry results, so their cells are mislabelled in the "
                  "ledger." % withres)
    return rows


if __name__ == "__main__":
    check()
    sys.exit(0)

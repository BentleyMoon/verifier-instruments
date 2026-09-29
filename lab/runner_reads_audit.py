# -*- coding: utf-8 -*-
"""A design must not set a behavioural flag its own runner never reads.

THE INCIDENT, 2026-08-21. Four designs were built to test whether the labelling defence depends on
WHERE untrusted text sits and WHO it is attributed to. They were copied from a template whose runner
is `injection_resistance`, and both `prepend_payload` and `attribution` are read only by
`injection_oversight` -- meta_lab.py L3605 and L3043. Neither flag applied.

So the runs were three SEEDS of one experiment, twice, and a mechanism finding was published on them
and withdrawn the same day. Two pre-registered predictions were recorded as failed against an
experiment that never ran, which is worse than failing them honestly.

NO EXISTING CHECK COULD SEE IT, and each one passed for a good reason:

    normalize          stores both fields, so they survive the save -- they are real fields
    spec_field_audit   asks whether SOME runner reads a stored field, not whether THIS design's
                       runner reads the fields THIS design sets
    preflight          ran the design successfully, because an unread flag is IGNORED, not rejected
    band_agreement     every cell agrees -- they are the same experiment
    quoted_numbers     every figure is real and traceable
    sibling_agreement  the cells agree in sign, being identical

The defect is a mismatch between what a design DECLARES and what its runner CONSULTS, and nothing
looked at that pair.

SCOPE, AFTER A FALSE START. The first version scanned the whole MODULE a runner lives in. That is
correct for finding fields a runner reads through a HELPER -- predicate_digits arrives via
_tasks_for(spec, ...) -- and useless for per-runner granularity, because nine of thirteen runners
live in meta_lab.py together. Scanning the module made injection_resistance appear to read
everything injection_oversight reads, so it could not see the very bug it was written for, while
flagging nineteen designs for `runner`, `verifier` and `ks` -- structural fields, not requests.
Noisy AND blind, which is the fourth time in two days an instrument here scanned the wrong span.

The scope is now the runner's own body plus the helpers IT calls that take a spec: one level, which
covers _tasks_for and _grade without swallowing its neighbours.

WHAT COUNTS AS BEHAVIOURAL. Only fields that some runner reads. `claim`, `title`, `notes` and the
covenant are stored precisely so they are never executed, and flagging them would bury the signal --
the mistake three instruments here have already made. The behavioural set is DERIVED from what the
runners actually read, so a new knob joins it without anyone editing a list.

WHAT COUNTS AS SET. A truthy, non-default value. A design carrying `prepend_payload: false` has not
asked for anything and is not misleading anyone; one carrying `true` on a runner that never reads it
has declared an experiment it is not running.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

READ = re.compile(r"""spec\s*(?:\.get\(\s*|\[\s*)['"]([a-z0-9_]+)['"]""")

# Fields every design legitimately carries that no runner consults: identity, provenance, covenant,
# and scheduling. Listed rather than inferred because "not read by any runner" is exactly what these
# are FOR, and deriving them would make the check vacuous.
NEVER_BEHAVIOURAL = frozenset((
    "id", "title", "claim", "anti_claim", "axis", "measure", "notes", "created", "tier",
    "evidence_role", "replicates", "canonical_metrics", "invalidation_conditions", "stop_rules",
    "claim_ceiling", "informative_limit", "repair_path", "external_recipe", "meoi",
    "derived_figures", "acknowledged_siblings", "substrate", "architecture",
    # Structural, not requests: `runner` SELECTS the runner and is never read by it; `verifier` and
    # `ks` are consumed by the queue and by runners that sweep, and their presence on one that does
    # not is a template artefact rather than a declared experiment.
    "runner", "verifier", "ks", "bands", "model", "seed", "n_tasks", "temperature",
))


def mismatched_verifier():
    """Designs declaring a SEPARATE checker on a runner that never reads `verifier`.

    `verifier` sits in the structural exclusion above, and correctly: on a runner that consumes it,
    its presence is not a request, and the first version of this audit flagged nineteen designs for
    it as noise. That reasoning holds while verifier == model, which is the template default.

    It does not hold when they DIFFER. `verifier != model` is a declaration that a SEPARATE model
    does the checking, and on a runner that never reads the field the design was self-checked
    regardless of what it says. That is the same declare-versus-consult mismatch this file exists
    for, in the one field the file deliberately stopped looking at.

    FOUND 2026-08-29: run_injection_subtlety and run_injection_resistance never read `verifier` --
    checked against the AST of their own bodies, not by grep -- while 82% of the corpus is
    self-checked and those two runners carry the flagship covert-attack and labelling-defence
    results. Cross-checking either needs a runner change; a design cannot request it.
    """
    import meta_lab as _M
    done = {r.get("design_id") for r in _M._read_jsonl(_M.RESULTS)}
    out = []
    for d in _M._read_jsonl(_M.DESIGNS):
        runner = d.get("runner")
        m, v = d.get("model"), d.get("verifier")
        if not runner or not m or not v or m == v:
            continue
        fn = getattr(_M, "run_" + str(runner), None)
        if fn is None:
            continue
        try:
            src = _module_source(runner)
        except Exception:
            continue
        if "verifier" in src:
            continue                      # the runner consults it; the declaration is honoured
        out.append({"design": d.get("id"), "runner": runner, "model": m, "verifier": v,
                    "has_result": d.get("id") in done})
    return out


def _module_source(runner):
    """The runner's own body, plus the source of every spec-taking helper it calls.

    ONE LEVEL, deliberately. Scanning the whole module makes the nine runners that share meta_lab.py
    indistinguishable -- injection_resistance would appear to read prepend_payload because
    injection_oversight does, twenty lines away. Scanning only the body misses predicate_digits,
    which arrives through _tasks_for(spec, ...). The body plus its direct spec-taking helpers is the
    span that answers "does THIS runner consult this field".
    """
    import inspect
    import re as _re
    import meta_lab as M
    fn = getattr(M, "run_" + str(runner), None)
    if fn is None:
        return None
    try:
        body = inspect.getsource(fn)
    except (OSError, TypeError):
        return None
    parts = [body]
    module = sys.modules.get(getattr(fn, "__module__", "") or "")
    # CALLS, FROM THE SYNTAX TREE -- not `name(` anywhere in the text. A comment in the runner reading
    # "normalize() refuses one without the other" (2026-09-19) pulled normalize's source in, and
    # normalize touches every field, so injection_oversight suddenly "read" measure, substrate and
    # intentional_duplicate -- the exact false reads the docstring above was written to prevent.
    try:
        import ast
        import textwrap
        called = set()
        for node in ast.walk(ast.parse(textwrap.dedent(body))):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Name):
                    called.add(f.id)
                elif isinstance(f, ast.Attribute):
                    called.add(f.attr)
    except SyntaxError:
        called = set(_re.findall(r"([A-Za-z_][A-Za-z_0-9]*)\s*\(", body))
    for name in sorted(called):
        helper = getattr(module, name, None) or getattr(M, name, None)
        if helper is None or not callable(helper) or name.startswith("run_"):
            continue
        try:
            sig = inspect.signature(helper)
            if "spec" not in sig.parameters:
                continue
            parts.append(inspect.getsource(helper))
        except (OSError, TypeError, ValueError):
            continue
    return "\n".join(parts)


def behavioural_fields():
    """Every spec key that ANY runner reads -- the set a design can meaningfully set."""
    import meta_lab as M
    out = set()
    for runner in M.RUNNERS:
        src = _module_source(runner)
        if src:
            out |= set(READ.findall(src))
    return out - NEVER_BEHAVIOURAL


def reads_of(runner, _cache={}):
    if runner not in _cache:
        src = _module_source(runner)
        _cache[runner] = set(READ.findall(src)) if src else set()
    return _cache[runner]


# Cases looked at and explained, keyed (design, field) so a DIFFERENT unread flag on the same
# design still surfaces. An adjudication is a claim on the record, not a way to stop being asked:
# the reason has to say why this particular flag being ignored does not corrupt the result.
#
# A standing count of ten is a count nobody reads, and then the eleventh arrives looking like noise.
ADJUDICATED = {
    ("lbattr_dolphin", "attribution"):
        "finding 23 is about these runs -- it reads them as the seed replicates they turned out to "
        "be, and says so: attribution is read only by injection_oversight",
    ("lbattr_qwen35", "attribution"):
        "finding 23 covers this run: attribution is read only by injection_oversight, so this "
        "design is a seed replicate of the arithmetic control, and the position/attribution claim "
        "built on it is already withdrawn",
    ("lbpos_dolphin_pre", "prepend_payload"):
        "as lbattr_dolphin -- finding 23 names prepend_payload explicitly as unread here",
    ("lbpos_qwen35_pre", "prepend_payload"):
        "finding 23 covers this run and names prepend_payload as unread by injection_resistance; "
        "the design is a seed replicate of the control and its position claim is withdrawn",
    ("vpl_abstain", "allow_abstain"):
        "RETRACTED 2026-08-21 under control_as_treatment; kept in the corpus as a same-seed "
        "replicate, which is the only thing it measures",
    ("vpl_retest", "retest"):
        "RETRACTED 2026-08-21 under control_as_treatment: run_verifier_placement reads only model, "
        "verifier and k, so this ran the control condition; kept as a same-seed replicate",
    ("agg_hermes_lead", "k"):
        "inert: run_aggregation_bakeoff sizes its pool from spec['models'], which lists three "
        "models here, so k=3 is redundant with what the runner actually reads rather than ignored "
        "in a way that changes the pool",
    ("bnd_dolphinmx", "k"):
        "inert: k is template baggage; the claim is about obedience under framing pressure and "
        "nothing in it turns on k",
    ("bnd_qwen36moe", "k"):
        "inert: k is template baggage on boundary_control, and this design's claim is about "
        "obedience rising under framing pressure -- nothing in it turns on k",
    ("is_qwen35", "k"):
        "inert: k is template baggage; the claim is about payload plausibility and does not use k",
    ("prop_p201", "models"):
        "inert because it has never run: an R&D-autopilot proposal, never queued, no result exists for "
        "the unread list to corrupt. NOT benign if queued -- verifier_placement reads `model` only, so "
        "its three-model consistency claim would silently run on gemma4:12b alone, and its verifier is "
        "empty. Rewrite it into one design per model before it is ever enqueued.",
}


def check(verbose=True, designs=None):
    import meta_lab as M
    behavioural = behavioural_fields()
    rows = designs if designs is not None else M._read_jsonl(M.DESIGNS)
    # last write wins, as get_design does
    latest = {}
    for d in rows:
        if d.get("id"):
            latest[d["id"]] = d

    live = {r.get("id") for r in M._read_jsonl(M.QUEUE)
            if r.get("status") in ("queued", "running")}
    res = {r.get("design_id") for r in M._read_jsonl(M.RESULTS)}

    problems = []
    for did, spec in sorted(latest.items()):
        runner = spec.get("runner")
        if not runner or runner == "custom":
            continue
        consulted = reads_of(runner)
        if not consulted:
            continue                      # runner source unavailable: cannot judge, do not invent
        unread = []
        for field, value in spec.items():
            if field not in behavioural or field in consulted:
                continue
            # SET means asked for. False/None/empty is not a request.
            if value in (None, False, "", [], {}, 0):
                continue
            unread.append(field)
        if unread:
            open_fields = [f for f in unread if (did, f) not in ADJUDICATED]
            problems.append({"design": did, "runner": runner, "fields": sorted(unread),
                             "open": sorted(open_fields),
                             "adjudicated": {f: ADJUDICATED[(did, f)]
                                             for f in unread if (did, f) in ADJUDICATED},
                             "state": ("queued" if did in live else
                                       "has result" if did in res else "idle")})

    if verbose:
        print("RUNNER READS AUDIT -- behavioural flags a design sets that its runner never consults")
        print("  %d behavioural field(s) across %d runner(s); %d design(s) examined"
              % (len(behavioural), len(M.RUNNERS), len(latest)))
        print()
        if problems:
            print("  DECLARED BUT NEVER CONSULTED: %d design(s)" % len(problems))
            for p in problems[:20]:
                print("     %-26s runner=%-22s %-10s %s"
                      % (p["design"][:26], p["runner"][:22], p["state"], ", ".join(p["fields"])))
            if len(problems) > 20:
                print("     ... and %d more not shown (%d total)"
                      % (len(problems) - 20, len(problems)))
            print()
            print("  A flag the runner does not read is IGNORED, not rejected. The design runs, the")
            print("  numbers are real, every other check passes -- and the experiment described is")
            print("  not the experiment performed.")
        else:
            print("  every behavioural flag set by a design is read by that design's runner")
    return problems


if __name__ == "__main__":
    bad = check()
    sys.exit(1 if bad else 0)

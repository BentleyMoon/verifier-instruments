"""Run every instrument against a known input, so its silence means something.

An always-silent checker produces the same output as a correct one on clean input. This repo has
shipped that twice in one day: unanswered_audit's FLOORED detector was pointed at RESULT_*.json and
returned a clean bill of health for 351 graded results it never opened, and launch_when_quiet's
window gate failed on its own priming baseline -- every FAIL it ever produced. Neither was detectable
from the output. Both were detectable in one line by asking: CAN it fire, and CAN it stay quiet?

So each instrument declares two fixtures:

  POSITIVE  an input it MUST flag. Proves it can fire at all. Without this, "0 findings" is
            indistinguishable from a broken checker, and gets reported as reassurance.
  NEGATIVE  a clean input it MUST NOT flag. Proves it is not merely alarming. unanswered_audit v1
            flagged 30 runs of which 23 were false; a checker whose hits are mostly noise gets
            ignored, which is the same outcome as not having it.

THREE STATES, NEVER TWO. An instrument with no declared control renders UNDECLARED, never PASS.
Conflating "checked and fine" with "never checked" is the exact defect this file exists to expose,
and a runner that made that mistake would be its own worst example.

THE CANARY. This runner certifies every other instrument, so its own failure mode is a silence that
reads as a pass. It therefore includes a deliberately broken instrument that it MUST catch. If the
canary is not reported FAIL, this runner is not measuring anything and it says so and exits non-zero
-- whatever the rest of the table says. Registered in advance as prediction #8 in
BLINDSPOT_PREREG.json (predicted shadow: instrumented_absence).

    python controls.py [--json]

Exit 0 only when every declared control passes AND the canary was caught.
"""
import json
import os
import shutil
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "CONTROLS.json")
_TMP = []


def _tmpfile(text, suffix=".md"):
    d = tempfile.mkdtemp(prefix="controls_")
    _TMP.append(d)
    p = os.path.join(d, "fixture" + suffix)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


# --------------------------------------------------------------------------------------------
# FIXTURES. Each returns (fired: bool, detail: str). "fired" means the instrument produced a
# finding -- not that it succeeded. The runner decides whether firing was correct.
# --------------------------------------------------------------------------------------------

CLAIM = "the monitor discriminates a compromised actor from a clean one"
ANTI = "the monitor does not discriminate; flagging is within MEOI of zero"
DELTAS = [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1]


def rigor_pos():
    import rigor
    r = rigor.claim(DELTAS, 0.05, CLAIM, ANTI, subject={"actor_clean_correct": 0.0})
    return r["verdict"] == "FLOORED", "verdict=%s" % r["verdict"]


def rigor_neg():
    import rigor
    r = rigor.claim(DELTAS, 0.05, CLAIM, ANTI, subject={"actor_clean_correct": 0.95})
    return r["verdict"] == "FLOORED", "verdict=%s" % r["verdict"]


def grade_pos():
    import meta_lab, rigor
    spec = {"meoi": 0.05, "anti_claim": ANTI, "tier": "EXPLORATORY", "id": "c", "title": "c"}
    c = meta_lab._grade("compromise_specificity", [1] * 60, spec, rigor, 1, CLAIM,
                        {"actor_clean_correct": 0.0, "flag_inj": 1.0, "flag_clean": 0.0})
    return c["verdict"] == "FLOORED", "unanimous effect on a dead subject -> %s" % c["verdict"]


def grade_neg():
    import meta_lab, rigor
    spec = {"meoi": 0.05, "anti_claim": ANTI, "tier": "EXPLORATORY", "id": "c", "title": "c"}
    c = meta_lab._grade("compromise_specificity", [1] * 60, spec, rigor, 1, CLAIM,
                        {"actor_clean_correct": 0.95, "flag_inj": 1.0, "flag_clean": 0.0})
    return c["verdict"] == "FLOORED", "unanimous effect on a LIVE subject -> %s" % c["verdict"]


def _rows_file(rows):
    return _tmpfile(json.dumps({"rows": rows}), ".json")


def audit_split_pos():
    """Half the rows never answered, and the arm differs sharply between the halves."""
    import unanswered_audit as UA
    rows = ([{"proposed": None, "gen_alone": 0, "strong_verify": 1} for _ in range(8)] +
            [{"proposed": 7, "gen_alone": 1, "strong_verify": 1} for _ in range(7)])
    f = UA.audit_one(_rows_file(rows))
    return any(x["kind"] == "SPLIT" for x in f), "%d finding(s)" % len(f)


def audit_split_neg():
    """Every row answered. Nothing to split on."""
    import unanswered_audit as UA
    rows = [{"proposed": i, "gen_alone": i % 2, "strong_verify": 1} for i in range(15)]
    f = UA.audit_one(_rows_file(rows))
    return any(x["kind"] == "SPLIT" for x in f), "%d finding(s)" % len(f)


def audit_floored_pos():
    import unanswered_audit as UA
    rec = {"bands": [{"cells": [{"metric": "m", "verdict": "POSITIVE", "effect": 0.9,
                                 "rates": {"actor_clean_correct": 0.0}}]}]}
    return bool(UA._floored_cells(rec, "fixture")), "graded POSITIVE on a dead actor"


def audit_floored_neg():
    import unanswered_audit as UA
    rec = {"bands": [{"cells": [{"metric": "m", "verdict": "POSITIVE", "effect": 0.9,
                                 "rates": {"actor_clean_correct": 1.0}}]}]}
    return bool(UA._floored_cells(rec, "fixture")), "live actor"


LEDGER = {"RESULT_WEAKONE": {"INCONCLUSIVE"}, "RESULT_STRONGONE": {"POSITIVE"}}


def lint_pos():
    import verdict_lint as VL
    f = VL.check(_tmpfile("This is proven by RESULT_WEAKONE beyond doubt."), LEDGER)
    return any(x["kind"] == "OVERCLAIM" for x in f), "%s" % [x["kind"] for x in f]


def lint_neg():
    import verdict_lint as VL
    f = VL.check(_tmpfile("This is proven by RESULT_STRONGONE."), LEDGER)
    return any(x["kind"] == "OVERCLAIM" for x in f), "%s" % [x["kind"] for x in f]


def winscan_pos():
    import winscan
    src = "import subprocess\nsubprocess.run(['cmd', '/c', 'dir'])\n"
    return bool(winscan.scan_file(_tmpfile(src, ".py"))), "bare subprocess.run"


def winscan_neg():
    import winscan
    src = ("import subprocess\n"
           "subprocess.run(['cmd', '/c', 'dir'], creationflags=0x08000000)\n")
    return bool(winscan.scan_file(_tmpfile(src, ".py"))), "CREATE_NO_WINDOW"


def _spec(n):
    import meta_lab
    d = meta_lab.get_design("vph_independent")
    if not d:
        raise RuntimeError("template design missing")
    s = dict(d)
    s["n_tasks"] = n
    return s


def power_pos():
    """Tiny n must NOT be reported as able to confirm a null."""
    import meta_lab
    pw = meta_lab.power_preview(_spec(8))
    return not pw.get("null_capable"), "n=8 null_capable=%s" % pw.get("null_capable")


def power_neg():
    import meta_lab
    pw = meta_lab.power_preview(_spec(900))
    return not pw.get("null_capable"), "n=900 null_capable=%s" % pw.get("null_capable")


def lease_pos():
    """A blocked lease must invoke on_wait, or a waiting daemon goes silent and reads as crashed."""
    import broker_lease
    seen, calls = [], []
    real = broker_lease._send
    realsleep = broker_lease.time.sleep
    try:
        def fake(req, timeout=5.0):
            seen.append(req["op"])
            if req["op"] == "release":
                return {"result": "ok"}
            return {"result": "grant", "token": "t"} if len(seen) > 3 else {"result": "wait"}
        broker_lease._send = fake
        broker_lease.time.sleep = lambda s: None
        with broker_lease.lease(exp_id="ctl", on_wait=calls.append):
            pass
    finally:
        broker_lease._send = real
        broker_lease.time.sleep = realsleep
    return bool(calls), "on_wait fired %d time(s)" % len(calls)


def lease_neg():
    """Granted immediately -- on_wait must NOT fire."""
    import broker_lease
    calls = []
    real, realsleep = broker_lease._send, broker_lease.time.sleep
    try:
        broker_lease._send = lambda req, timeout=5.0: (
            {"result": "ok"} if req["op"] == "release" else {"result": "grant", "token": "t"})
        broker_lease.time.sleep = lambda s: None
        with broker_lease.lease(exp_id="ctl", on_wait=calls.append):
            pass
    finally:
        broker_lease._send = real
        broker_lease.time.sleep = realsleep
    return bool(calls), "on_wait fired %d time(s)" % len(calls)


def rateclass_pos():
    """A harness emitting a competence-shaped rate in neither tuple must be reported."""
    import meta_lab
    fake = 'rates = {"solver_clean_correct": sum(x) / N, "obey": 0.0}\n'
    c = meta_lab.classify_rates(fake)
    return bool(c["unclassified"]), "unclassified=%s" % c["unclassified"]


def rateclass_neg():
    """The real harness must have none. This is the live guarantee, not a fixture."""
    import meta_lab
    un = meta_lab.unclassified_competence_rates()
    return bool(un), "unclassified in meta_lab.py: %s" % (un or "none")


_GOOD_HASH = {"rigor.py": "aaaa", "meta_lab.py": "bbbb"}
_HIST_PASS = {frozenset(_GOOD_HASH.items()): True}
_HIST_FAIL = {frozenset(_GOOD_HASH.items()): False}


def trust_pos():
    """A verdict graded at hashes whose controls FAILED must not read as trustworthy."""
    import trust
    st, _ = trust.state_of(_GOOD_HASH, _HIST_FAIL, _GOOD_HASH)
    return st != "VERIFIED", "state=%s" % st


def trust_neg():
    """Graded at passing hashes that have not changed -- must read VERIFIED, or the join is useless."""
    import trust
    st, _ = trust.state_of(_GOOD_HASH, _HIST_PASS, _GOOD_HASH)
    return st != "VERIFIED", "state=%s" % st


def decisions_pos():
    """A decision with no default must be REFUSED. An unmade decision already has a default;
    leaving it unstated does not remove it, it only hides who chose."""
    import decisions
    r = decisions.open_decision("q?", "cost", {"a": "x", "r": "y"}, default=None, hours=24)
    return "error" in r, r.get("error", "accepted a decision with no default")


def decisions_neg():
    """A complete decision must be accepted -- written to an ISOLATED store, never the real one.

    The first version called open_decision() against the live ledger and then appended a second
    record to mark the fixture resolved. succession runs controls every 300 seconds, so that quietly
    wrote two rows into DECISIONS.jsonl twice an hour: by the time anyone looked there were hundreds
    of decisions whose question was "q?". A control that contaminates the data it certifies is worse
    than no control -- and this one had been passing the whole time, because writing junk was exactly
    what it was asserting.
    """
    import decisions
    real = decisions.STORE
    d = tempfile.mkdtemp(prefix="controls_dec_")
    _TMP.append(d)
    decisions.STORE = os.path.join(d, "DECISIONS.jsonl")
    try:
        r = decisions.open_decision("q?", "cost", {"a": "x", "r": "y"}, default="a", hours=24)
    finally:
        decisions.STORE = real
    return "error" in r, r.get("error", "accepted a complete decision")


def bypass_pos():
    """The scan must be able to name a bypassing runner, or its empty answer means nothing."""
    import meta_lab
    return callable(meta_lab.runners_bypassing_liveness), "scan is callable"


def bypass_neg():
    """The live harness must have NO runner grading without a liveness check.

    Expected to FAIL until the remaining three are fixed. That is the point: a real gap should be
    visible in the table rather than absorbed into a declaration that it is fine. Paired with a
    decision that carries an expiry, so it cannot sit here indefinitely and train the reader to
    ignore a red row.
    """
    import meta_lab
    b = meta_lab.runners_bypassing_liveness()
    return bool(b), "bypassing: %s" % (b or "none")


def _fake_corpus(verdict, sd=0.30):
    """One design, one cell, with a known verdict -- enough for a yield table to have a right answer."""
    return [{"design_id": "fixture_design", "runner": "fixture_runner", "n_tasks": 10,
             "elapsed_s": 100.0, "bands": [{"cells": [
                 {"metric": "m", "verdict": verdict, "effect": 0.5, "sd": sd, "n": 10,
                  "mde": 0.04, "powered": True, "rates": {"actor_clean_correct": 1.0}}]}]}]


def yield_pos():
    """A corpus whose only result SETTLED something must not score as zero yield.

    Registered in BLINDSPOT_PREREG.json as prediction #1 (objective_incentive): whatever this
    instrument's VALUE map omits becomes worthless by construction. SATURATED is a real, unanimous
    effect and scored zero until today -- so a runner producing them read as unproductive.
    """
    import importlib, instrument_check
    importlib.reload(instrument_check)
    rows = instrument_check._results(_fake_corpus("SATURATED"))
    return len(rows) == 1 and rows[0]["bands"][0]["cells"][0]["verdict"] == "SATURATED",         "injected corpus reached the instrument"


def yield_neg():
    """An empty corpus must yield nothing rather than inventing a row."""
    import importlib, instrument_check
    importlib.reload(instrument_check)
    return bool(instrument_check._results([])), "empty corpus -> %d row(s)" % len(
        instrument_check._results([]))


def defence_pos():
    """A design present in the corpus must be found by the scorer."""
    import defence_score
    cells, rates = defence_score._cells("fixture_design", _fake_corpus("POSITIVE"))
    return bool(cells), "cells=%s" % sorted(cells)


def defence_neg():
    """A design ABSENT from the corpus must return nothing, not the wrong design's cells."""
    import defence_score
    cells, _ = defence_score._cells("no_such_design", _fake_corpus("POSITIVE"))
    return bool(cells), "cells=%s" % sorted(cells)


def _corpus(orphans=0, offered=3):
    ds = [{"id": "d%d" % i, "created": "2026-08-01"} for i in range(offered + orphans)]
    q = [{"name": "d%d" % i, "status": "queued"} for i in range(offered)]
    res = [{"ts": "2026-08-01", "design_id": "d0"} for _ in range(offered)]
    return ds, q, res


def solvency_pos():
    """A corpus where most designs were never offered must breach. Without this the clean reading
    on the live corpus is indistinguishable from a checker that cannot see an orphan."""
    import solvency
    ds, q, res = _corpus(orphans=20, offered=3)
    r = solvency.survey(ds, q, res)
    return r["orphan_breach"], "orphan_rate=%.2f" % r["orphan_rate"]


def solvency_neg():
    """A corpus where everything was offered must NOT breach."""
    import solvency
    ds, q, res = _corpus(orphans=0, offered=20)
    r = solvency.survey(ds, q, res)
    return r["orphan_breach"], "orphan_rate=%.2f" % r["orphan_rate"]


def canary_pos():
    """A deliberately broken instrument: it never fires, whatever it is shown.

    Its POSITIVE control must therefore FAIL. If this row is not reported FAIL, the runner is not
    detecting silence and every PASS above it is uninterpretable.
    """
    return False, "the canary never fires -- this control is SUPPOSED to fail"


def canary_neg():
    return False, "silent, as a broken instrument would be"


def _lattice():
    import importlib
    import lattice_canary
    importlib.reload(lattice_canary)      # this process is long-lived; the canary's own logic may change
    return lattice_canary.cached_run()


def lattice_pos():
    """POSITIVE: planted defects the ledger audits catch today, one per class, each in its own tiny
    corpus (lattice_canary.py). Fires only when every regression case is caught. The self-monitored
    exemption in leak_exposure would have failed this the day it was written: run against that
    version, cases C1-C3 are all missed (idea_ratchet Cell 2)."""
    r = _lattice()
    bad = [p for p in r.get("problems") or [] if "MISSED" in p or "errored" in p]
    n = sum(1 for x in r.get("rows") or [] if x.get("kind") == "regression")
    gaps = r.get("gaps_now_caught") or []
    note = ("; known gap now caught, promote: %s" % ", ".join(gaps)) if gaps else ""
    if bad:
        return False, "; ".join(bad)[:200]
    return True, "%d/%d planted classes caught%s" % (n, n, note)


def lattice_neg():
    """NEGATIVE: the sound twins, matched in surface form to each planted case. An audit that names
    anything in a twin is alarming on clean input."""
    r = _lattice()
    alarms = [p for p in r.get("problems") or [] if "twin alarmed" in p]
    if alarms:
        return True, "; ".join(alarms)[:200]
    return False, "silent on all %d sound twins" % len(r.get("rows") or [])


# --------------------------------------------------------------------------------------------
# REGISTRY. `blast` is what rests on the instrument -- the ranking is by consequence, never by
# how alarming a failure looks. `undeclared` names why no control exists, so the gap is legible.
# --------------------------------------------------------------------------------------------

def gatewin_pos():
    """POSITIVE: a console window appears during the loads. The gate MUST flag it."""
    import launch_when_quiet as L
    console = sorted(L.CONSOLE_CLASSES)[0]
    quiet = [(1, "Chrome_WidgetWin_1", 100, "chrome.exe")]
    state = {"n": 0}

    def enum():
        rows = list(quiet)
        if state["n"] > 0:
            rows.append((9, console, 555, "ollama.exe"))
        return rows

    ok, detail = L.gate_windows(enum_windows=enum,
                                load_once=lambda i: state.__setitem__("n", state["n"] + 1),
                                settle=0)
    return (not ok), ("flagged a console that appeared mid-load" if not ok
                      else "MISSED a console appearing during the loads")


def gatewin_neg():
    """NEGATIVE: a quiet desktop, and a desktop with a console ALREADY on screen.

    The second half is the defect this gate actually shipped. It appended during its own priming
    pass, so ten pre-existing windows -- an operator's PowerShell among them -- were counted as new
    and every FAIL it ever produced was that. A gate whose failures are its own baseline blocks
    correct states and teaches the operator to route around it."""
    import launch_when_quiet as L
    console = sorted(L.CONSOLE_CLASSES)[0]
    quiet = [(1, "Chrome_WidgetWin_1", 100, "chrome.exe")]
    ok_quiet, _ = L.gate_windows(enum_windows=lambda: list(quiet),
                                 load_once=lambda i: None, settle=0)
    pre = quiet + [(9, console, 555, "WindowsTerminal.exe")]
    ok_pre, _ = L.gate_windows(enum_windows=lambda: list(pre),
                               load_once=lambda i: None, settle=0)
    noisy = (not ok_quiet) or (not ok_pre)
    if not ok_quiet:
        return True, "fired on a QUIET desktop"
    if not ok_pre:
        return True, "fired on a console that was already on screen -- its own priming pass"
    return False, "silent on a quiet desktop and on a pre-existing console"



REGISTRY = [
    dict(name="rigor.claim", blast="1359 graded cells", pos=rigor_pos, neg=rigor_neg),
    dict(name="meta_lab._grade", blast="every injection cell", pos=grade_pos, neg=grade_neg),
    dict(name="unanswered_audit.SPLIT", blast="9 SCALEOVERSIGHT runs", pos=audit_split_pos,
         neg=audit_split_neg),
    dict(name="unanswered_audit.FLOORED", blast="the graded corpus", pos=audit_floored_pos,
         neg=audit_floored_neg),
    dict(name="verdict_lint", blast="all public prose (R2)", pos=lint_pos, neg=lint_neg),
    dict(name="winscan", blast="the operator's desktop", pos=winscan_pos, neg=winscan_neg),
    dict(name="power_preview", blast="every queued design", pos=power_pos, neg=power_neg),
    dict(name="broker_lease.on_wait", blast="one-lane + daemon liveness", pos=lease_pos,
         neg=lease_neg),
    dict(name="rate classification", blast="which runners the liveness gate reaches",
         pos=rateclass_pos, neg=rateclass_neg),
    dict(name="runner liveness coverage", blast="which runners can be graded on nothing",
         pos=bypass_pos, neg=bypass_neg),
    dict(name="solvency", blast="whether proposals quietly stop being executed",
         pos=solvency_pos, neg=solvency_neg),
    dict(name="trust.state_of", blast="how every verdict renders", pos=trust_pos, neg=trust_neg),
    dict(name="decisions.open", blast="whether silence is legible", pos=decisions_pos,
         neg=decisions_neg),
    dict(name="CANARY (must fail)", blast="this runner's own validity", pos=canary_pos,
         neg=canary_neg, canary=True),

    # Declared UNDECLARED on purpose. Both read the live corpus straight off disk, so neither can be
    # shown a known input without an injectable corpus. Registered in BLINDSPOT_PREREG.json as the
    # two instruments never controlled at all -- and instrument_check is itself a meta-instrument
    # that ranks other instruments, which is the worst thing to leave uncontrolled.
    dict(name="instrument_check.yield", blast="runner selection", pos=yield_pos, neg=yield_neg),
    dict(name="defence_score", blast="which defences ship", pos=defence_pos, neg=defence_neg),
    # CONTROLLED SINCE 2026-08-18, and it was the last instrument that was not. It used to render
    # UNDECLARED because it needs four real model loads and a live desktop -- and its own note said
    # what that cost: "which is why its baseline defect survived three runs". The window source and
    # the load action are now injectable, so it answers to the same bar as everything else here.
    dict(name="launch_when_quiet.gate_windows", blast="every engine start",
         pos=gatewin_pos, neg=gatewin_neg),
    # CONTROLLED SINCE 2026-09-22: the ledger-reading audits, which the note above says could not be
    # shown a known input. lattice_canary builds that input -- eight audits, the planted classes they
    # catch, and the known gaps reported but never gating (every gap a defect of absence; a gap
    # closes when the absence is DECLARED, as conditions and rests_on_findings now are).
    dict(name="audit lattice (planted defects)",
         blast="whether a finding may quote a number: leak_exposure, quoted_numbers, parse capture, "
               "runner reads, band/sibling agreement, substrate label, findings_lint",
         pos=lattice_pos, neg=lattice_neg),
]


# Fingerprints come from fingerprint.py, which owns the single definition. An earlier version kept
# its own copy here; this repo forked the liveness vocabulary the same way and the two halves diverged
# within hours, in both directions.
import fingerprint as _fp                                                        # noqa: E402

CONTROLS_LOG = os.path.join(HERE, "CONTROLS_LOG.jsonl")


def fingerprint():
    """Which version of the instrument population this run is a statement about."""
    return _fp.instruments(HERE)


def run_one(c):
    row = {"instrument": c["name"], "blast": c["blast"]}
    if c.get("undeclared"):
        row.update(state="UNDECLARED", fires="-", silent="-", detail=c["undeclared"])
        return row
    try:
        fired, d1 = c["pos"]()
    except Exception as exc:
        row.update(state="ERROR", fires="err", silent="-",
                   detail="positive fixture raised %s: %s" % (type(exc).__name__, str(exc)[:90]))
        return row
    try:
        noise, d2 = c["neg"]()
    except Exception as exc:
        row.update(state="ERROR", fires="ok" if fired else "NO", silent="err",
                   detail="negative fixture raised %s: %s" % (type(exc).__name__, str(exc)[:90]))
        return row
    ok = fired and not noise
    row.update(state="PASS" if ok else "FAIL",
               fires="ok" if fired else "NO",
               silent="ok" if not noise else "NO",
               detail="%s | %s" % (d1, d2))
    return row


def snapshot(rows):
    """Build the payload AND persist it. One writer, called by both entry points.

    FOUND 2026-08-07 by audit. succession.verification_sweep() ran every control on every tick and
    used the results in memory to block the GPU grant -- that gate was live and correct the whole
    time. It never wrote this file, because only main() did. So CONTROLS.json sat 59.5 hours stale
    while home.py and control.py read it and showed those numbers as the lab's current instrument
    health.

    Two sources of truth for one fact, one live and one displayed, quietly diverging. Exactly the
    shape this program keeps finding in its own instruments, and this time in the instrument whose
    whole job is to catch that. Extracted rather than duplicated so a third caller cannot reintroduce
    a third answer.
    """
    canary = next((r for r in rows if r["instrument"].startswith("CANARY")), None)
    canary_caught = bool(canary and canary["state"] == "FAIL")

    failed = [r for r in rows if r["state"] == "FAIL" and not r["instrument"].startswith("CANARY")]
    errored = [r for r in rows if r["state"] == "ERROR"]
    undecl = [r for r in rows if r["state"] == "UNDECLARED"]

    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "canary_caught": canary_caught,
               "verified_at": fingerprint(), "rows": rows, "failed": len(failed),
               "errored": len(errored), "undeclared": len(undecl)}
    try:
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception:
        pass
    try:
        with open(CONTROLS_LOG, "a", encoding="utf-8") as fh:
            keep = ("ts", "canary_caught", "verified_at", "failed", "errored", "undeclared")
            fh.write(json.dumps({k: payload[k] for k in keep}) + os.linesep)
    except Exception:
        pass
    return payload


def main():
    rows = [run_one(c) for c in REGISTRY]
    payload = snapshot(rows)          # writes CONTROLS.json AND appends CONTROLS_LOG.jsonl
    canary_caught = payload["canary_caught"]
    # READ FROM THE PAYLOAD, not recomputed. snapshot() already derives these and its docstring says
    # why: "Extracted rather than duplicated so a third caller cannot reintroduce a third answer."
    # main() referenced the three names without binding them at all, so it raised NameError on the
    # line immediately BEFORE the canary warning -- the most important message this file can print,
    # "every PASS above is uninterpretable", was unreachable, and the exit code was a traceback
    # instead of the intended 0/1/2. Anything gating on that exit code has been reading a crash as a
    # verdict. My first repair recomputed them here against a key that does not exist, which would
    # have been exactly the third answer that comment warns about.
    n_failed = payload["failed"]
    n_errored = payload["errored"]
    n_undecl = payload["undeclared"]

    for d in _TMP:
        shutil.rmtree(d, ignore_errors=True)

    if "--json" in sys.argv:
        print(json.dumps(payload, indent=2))
    else:
        print("%-30s %-6s %-7s %-7s %s" % ("INSTRUMENT", "STATE", "FIRES", "SILENT", "BLAST"))
        for r in rows:
            print("%-30s %-6s %-7s %-7s %s" % (r["instrument"][:30], r["state"], r["fires"],
                                               r["silent"], r["blast"][:34]))
        print()
        for r in rows:
            if r["state"] in ("FAIL", "ERROR", "UNDECLARED"):
                print("  %-30s %s" % (r["instrument"][:30], r["detail"]))
        print("\n%d declared control(s) failed, %d errored, %d undeclared (canary caught: %s)"
              % (n_failed, n_errored, n_undecl, canary_caught))

    if not canary_caught:
        print("\n*** THE CANARY WAS NOT CAUGHT. This runner is not detecting silence, so every PASS "
              "above is uninterpretable. Fix the runner before believing the table. ***")
        return 2
    return 1 if (n_failed or n_errored) else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Re-run the verifier lab's checks offline: no network, no GPU, no model, no API key.

    pip install -r requirements.txt
    python reproduce.py

Works from lab/, with lab/ first on sys.path, as the engine runs its own tools:

  1. VERDICTS  recompute every stored verdict that can be recomputed and require an exact match
  2. CANARY    lattice_canary's planted-defect cases, run in this process: every regression case
               caught, every known gap missed as declared, no sound twin alarming
  3. CONTROLS  the science rows of controls.REGISTRY through controls.run_one: each must PASS,
               except the canary row, which must FAIL
  4. LEDGERS   information only: retractions.py on this subset, and which engine commands run here

Exit 0 only when 1, 2 and 3 pass. Writes only inside lab/ (a temporary directory and the canary
cache); both are deleted before exit and both are listed in .gitignore.
"""
import json
import os
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True

ROOT = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(ROOT, "lab")
TMP = os.path.join(LAB, ".reproduce_tmp")

SCIENCE = (
    "rigor.claim", "meta_lab._grade", "unanswered_audit.SPLIT", "unanswered_audit.FLOORED",
    "verdict_lint", "power_preview", "rate classification", "runner liveness coverage",
    "trust.state_of", "CANARY (must fail)", "instrument_check.yield", "defence_score",
    "audit lattice (planted defects)",
)
OPERATIONAL = {
    "winscan": "Windows console hygiene on the operator's desktop; winscan.py is not shipped",
    "broker_lease.on_wait": "the GPU lease; broker_lease.py opens a socket to the lease broker and is not shipped",
    "solvency": "the run queue; solvency.py is not shipped",
    "decisions.open": "operator decision governance; decisions.py is not shipped",
    "launch_when_quiet.gate_windows": "the engine start-up gate; launch_when_quiet.py loads Windows DLLs and launches the model server, and is not shipped",
}


class Blocked(RuntimeError):
    """Raised by every network and process entry point once block_network() has run."""


def block_network():
    import socket
    import subprocess
    import urllib.request   # first: it imports ssl, which subclasses socket.socket at import time

    def refuse(what):
        def _refuse(*args, **kwargs):
            raise Blocked("reproduce.py blocks the network and child processes: %s was called" % what)
        return _refuse

    class NoSocket(socket.socket):
        def __init__(self, *args, **kwargs):
            raise Blocked("reproduce.py blocks the network: socket.socket() was called")

    class NoPopen(subprocess.Popen):
        def __init__(self, *args, **kwargs):
            raise Blocked("reproduce.py blocks child processes: subprocess.Popen(%r) was called"
                          % (args[:1],))

    socket.socket = NoSocket
    socket.create_connection = refuse("socket.create_connection")
    urllib.request.urlopen = refuse("urllib.request.urlopen")
    subprocess.Popen = NoPopen


def blocked_selftest():
    import socket
    import subprocess
    import urllib.request
    probes = (("socket.socket()", lambda: socket.socket()),
              ("socket.create_connection", lambda: socket.create_connection(("127.0.0.1", 9), 1)),
              ("urllib.request.urlopen", lambda: urllib.request.urlopen("http://127.0.0.1:9/", timeout=1)),
              ("subprocess.run", lambda: subprocess.run([sys.executable, "-c", "pass"])))
    out = []
    for name, fn in probes:
        try:
            fn()
            out.append((name, False))
        except Blocked:
            out.append((name, True))
        except Exception:
            out.append((name, False))
    return out


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ------------------------------------------------------------------------------------ 1. VERDICTS
def check_verdicts():
    import meta_lab
    import rigor
    rows = meta_lab._read_jsonl(meta_lab.RESULTS)
    n_cells = sum(len(b.get("cells") or []) for r in rows for b in r.get("bands") or [])
    n_vals = sum(1 for r in rows for b in r.get("bands") or [] for c in b.get("cells") or []
                 if c.get("values"))
    print("%d result rows, %d cells, %d with per-item values. Recomputing those with rigor.audit "
          "(about half a minute)..." % (len(rows), n_cells, n_vals))
    sys.stdout.flush()
    t0 = time.time()
    matched_values, matched_summary, mismatched, not_recomputable, rule_changed = [], [], [], [], []
    for r in rows:
        fam = meta_lab._family_size(r)
        for b in r.get("bands") or []:
            for c in b.get("cells") or []:
                label = "%s | band %s | %s" % (r.get("design_id"), b.get("band"), c.get("metric"))
                verdict, vals = c.get("verdict"), c.get("values")
                if vals:
                    fs = c.get("family_size")
                    if fs is not None and fs != fam:
                        mismatched.append((label, "family_size stored %s, meta_lab._family_size gives %s"
                                           % (fs, fam)))
                        continue
                    rec = rigor.audit(vals, c.get("meoi"), family_size=fs or 1)
                    lo, hi = rec["ci_low"], rec["ci_high"]
                    if fs is not None:     # graded through meta_lab.apply_family_correction, which
                        lo, hi = round(lo, 4), round(hi, 4)   # rounds the widened CI to 4 decimals
                    got = (rec["effect"], lo, hi, rec["verdict"])
                    want = (c.get("effect"), c.get("ci_low"), c.get("ci_high"), verdict)
                    if got == want:
                        matched_values.append((label, verdict))
                    else:
                        mismatched.append((label, "stored effect/ci_low/ci_high/verdict %s, recomputed %s"
                                           % (want, got)))
                    continue
                n, effect, meoi = c.get("n"), c.get("effect"), c.get("meoi")
                rates = c.get("rates") or {}
                if verdict in ("SATURATED", "INFORMATIVE_NULL", "DEGENERATE"):
                    if not isinstance(n, int) or n <= 0 or effect is None:
                        not_recomputable.append((label, verdict, "n not recorded"))
                        continue
                    d = meta_lab._degeneracy_verdict([effect] * n, meoi, rates=rates)
                    keys = ["verdict", "effect", "n"]
                    if verdict == "SATURATED":
                        keys += ["n_moved", "sign_test_p", "powered"]
                    if verdict == "INFORMATIVE_NULL":
                        keys += ["ci_low", "ci_high", "powered"]
                    diff = [(k, c.get(k), d.get(k)) for k in keys if c.get(k) != d.get(k)]
                    if diff:
                        mismatched.append((label, "summary re-derivation differs: %s" % diff))
                    else:
                        matched_summary.append((label, verdict))
                    continue
                if verdict == "FLOORED":
                    basis = c.get("liveness_basis") or "band"
                    if basis == "metric":
                        subject = {k: rates[k] for k in meta_lab.METRIC_SUBJECT_RATES.get(c.get("metric"), ())
                                   if k in rates}
                    else:
                        subject = meta_lab._subject_rates(rates)
                    floored = rigor.liveness(subject)
                    if floored == c.get("floored_rates") and subject == c.get("subject_rates"):
                        matched_summary.append((label, verdict))
                    else:
                        mismatched.append((label, "liveness re-derivation: stored %s, recomputed %s"
                                           % (c.get("floored_rates"), floored)))
                    now_subject, now_basis = meta_lab._subject_for_metric(rates, c.get("metric"))
                    if not rigor.liveness(now_subject):
                        rule_changed.append((label, basis, now_basis, now_subject))
                    continue
                not_recomputable.append((label, verdict, "no per-item values and no summary rule"))

    print("done in %.1f s" % (time.time() - t0))
    by = {}
    for _l, v in matched_values:
        by.setdefault(v, [0, 0])[0] += 1
    for _l, v in matched_summary:
        by.setdefault(v, [0, 0])[1] += 1
    print()
    print("%-18s %22s %28s" % ("verdict", "from per-item values", "re-derived from summary"))
    for v in sorted(by):
        print("%-18s %22d %28d" % (v, by[v][0], by[v][1]))
    print()
    for label, why in mismatched:
        print("  MISMATCH  %s: %s" % (label, why))
    for label, v, why in not_recomputable:
        print("  not recomputable: %s (%s): %s" % (label, v, why))
    if rule_changed:
        print()
        print("  note: %d FLOORED cell(s) were graded under the liveness rule recorded in the cell "
              "(liveness_basis); the current per-metric rule (meta_lab.METRIC_SUBJECT_RATES) would "
              "find a live subject for these, so a re-grade today would grade them:" % len(rule_changed))
        for label, basis, now_basis, subj in rule_changed:
            print("    %s: graded on basis=%s; today basis=%s, subject %s" % (label, basis, now_basis, subj))
    print()
    print("matched: %d (%d from per-item values, %d re-derived from stored summaries) | mismatched: %d "
          "| not recomputable: %d"
          % (len(matched_values) + len(matched_summary), len(matched_values), len(matched_summary),
             len(mismatched), len(not_recomputable)))
    ok = not mismatched and n_vals == len(matched_values)
    summary = ("%d matched from per-item values, %d re-derived from stored summaries, %d mismatched, "
               "%d not recomputable (%s)" % (len(matched_values), len(matched_summary), len(mismatched),
                                            len(not_recomputable),
                                            ", ".join(sorted({w for _l, _v, w in not_recomputable})) or "-"))
    return ok, summary


# -------------------------------------------------------------------------------------- 2. CANARY
def check_canary():
    import lattice_canary as LC
    rows = LC.run()
    ok, problems, promoted = LC.verdict(rows)
    print("%-4s %-10s %-8s %-60s %s" % ("case", "kind", "state", "planted defect", "caught by"))
    for r in rows:
        print("%-4s %-10s %-8s %-60s %s%s" % (
            r["case"], r["kind"], r["state"], r["what"][:60], ",".join(r.get("catchers") or []) or "-",
            ("  TWIN ALARM %s" % r["false_alarms"]) if r.get("false_alarms") else ""))
    reg = [r for r in rows if r["kind"] == "regression"]
    gaps = [r for r in rows if r["kind"] == "known gap"]
    reg_caught = [r for r in reg if r["state"] == "CAUGHT"]
    gaps_missed = [r for r in gaps if r["state"] == "MISSED"]
    alarms = [r for r in rows if r.get("false_alarms")]
    errors = [r for r in rows if r["state"] == "ERROR"]
    print()
    for p in problems:
        print("  FAIL  %s" % p)
    for c in promoted:
        print("  known gap %s is now CAUGHT; it was declared missed" % c)
    passed = ok and len(reg_caught) == len(reg) and len(gaps_missed) == len(gaps) and not alarms \
        and not errors
    print("lattice_canary.verdict ok=%s | regression cases caught %d/%d | known gaps missed as declared "
          "%d/%d | twins alarming %d | errors %d"
          % (ok, len(reg_caught), len(reg), len(gaps_missed), len(gaps), len(alarms), len(errors)))
    summary = ("%d/%d regression cases caught, %d/%d known gaps missed as declared, %d twin alarms"
               % (len(reg_caught), len(reg), len(gaps_missed), len(gaps), len(alarms)))
    # Prime the cache controls' lattice row reads, so it judges THIS run instead of starting a second
    # interpreter (lattice_canary.cached_run returns the cache when its source fingerprint matches).
    import fingerprint
    cache = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "epoch": time.time(),
             "sources": fingerprint.of(LC.SOURCES, LC.HERE), "ok": ok, "problems": problems,
             "gaps_now_caught": promoted, "rows": rows}
    with open(LC.CACHE, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=1, default=list)
    return passed, summary


# ------------------------------------------------------------------------------------ 3. CONTROLS
def check_controls():
    import controls
    names = [c["name"] for c in controls.REGISTRY]
    unknown = [n for n in names if n not in SCIENCE and n not in OPERATIONAL]
    missing = [n for n in SCIENCE if n not in names]
    rows = []
    try:
        for c in controls.REGISTRY:
            if c["name"] in SCIENCE:
                rows.append(controls.run_one(c))
    finally:
        for d in controls._TMP:
            shutil.rmtree(d, ignore_errors=True)
    print("%-32s %-6s %-6s %-6s %s" % ("instrument", "state", "fires", "silent", "required"))
    bad = []
    for r in rows:
        canary = r["instrument"].startswith("CANARY")
        want = "FAIL" if canary else "PASS"
        print("%-32s %-6s %-6s %-6s %s" % (r["instrument"][:32], r["state"], r["fires"], r["silent"], want))
        if r["state"] != want:
            bad.append(r)
    print()
    for r in rows:
        print("  %-32s %s" % (r["instrument"][:32], r["detail"][:160]))
    print()
    print("skipped (operational; the modules they certify are not shipped):")
    for n in names:
        if n in OPERATIONAL:
            print("  %-32s %s" % (n, OPERATIONAL[n]))
    for n in unknown:
        print("  UNCLASSIFIED registry row: %s" % n)
    for n in missing:
        print("  MISSING science row: %s" % n)
    canary_caught = any(r["instrument"].startswith("CANARY") and r["state"] == "FAIL" for r in rows)
    passed = not bad and not unknown and not missing and canary_caught
    n_pass = sum(1 for r in rows if r["state"] == "PASS")
    print()
    print("science rows run: %d | PASS: %d | canary caught (FAIL as required): %s | not as required: %d "
          "| skipped: %d | unclassified: %d"
          % (len(rows), n_pass, canary_caught, len(bad), sum(1 for n in names if n in OPERATIONAL),
             len(unknown)))
    summary = ("%d science rows PASS, canary row FAIL as required, %d not as required; %d operational "
               "rows skipped" % (n_pass, len(bad), sum(1 for n in names if n in OPERATIONAL)))
    return passed, summary


# ------------------------------------------------------------------------------------- 4. LEDGERS
COMMANDS = """\
The assessor brief (the programme's page for grant reviewers) lists seven commands. Here:

  python -m pytest tests/ -q   does not run here: the engine's test suite is not shipped, and one
                               of its tests pins a count in an engine file.
  python controls.py           runs, but its 5 operational rows ERROR because their modules are not
                               shipped, so it exits 1; it also writes CONTROLS.json and appends
                               CONTROLS_LOG.jsonl (both git-ignored). Section 3 above runs the 13
                               science rows through the same runner.
  python corpus_audit.py       corpus-wide. It runs, but grades only this subset, and its queue
                               section is empty because META_QUEUE.jsonl is not shipped.
  python retractions.py        runs (above). Its rate is over this subset's result rows, and only
                               whole-design retractions count toward it.
  python findings_lint.py      stops: it needs FINDINGS.json, the findings prose, which is not
                               shipped.
  python floored_audit.py      corpus-wide. It runs, and on this subset flags the FLOORED cells whose
                               own inputs were live (the cnf_invcross_qwen35 cells noted in section
                               1); it exits 1 whenever a suspect exists, by design. Its engine-wide
                               count covers exploratory cells that are not shipped.
  python verdict_lint.py       its ledger is the engine's RESULT_*.json cards, which are not shipped;
                               here it scans ../README.md against an empty ledger."""


def check_ledgers():
    import runpy
    print("--- python retractions.py (from lab/) ---")
    try:
        runpy.run_path(os.path.join(LAB, "retractions.py"), run_name="__main__")
    except Exception as exc:
        print("retractions.py raised %s: %s" % (type(exc).__name__, exc))
    print()
    try:
        with open(os.path.join(LAB, "SUPERSESSIONS.json"), encoding="utf-8") as fh:
            sup = json.load(fh)
        print("--- SUPERSESSIONS.json (re-serves recorded outside RETRACTIONS.jsonl) ---")
        for p in sup.get("pairs") or []:
            print("  %-26s -> %-29s %s  %s" % (p["parent"], p["reserve"], p.get("date") or "-",
                                              p["status"].split(":")[0]))
        ex = sup.get("exposed_without_reserve_result") or []
        print("  %d included design(s) read gemma4:12b output before the 2026-09-17 fix and have no "
              "re-serve result: %s" % (len(ex), ", ".join(e["design_id"] for e in ex) or "-"))
    except Exception as exc:
        print("SUPERSESSIONS.json unreadable: %s" % exc)
    print()
    print(COMMANDS)


# ---------------------------------------------------------------------------------------- main
def main():
    if sys.version_info < (3, 10):
        print("reproduce.py needs Python 3.10 or later")
        return 2
    block_network()
    section("0. NETWORK AND CHILD PROCESSES")
    probes = blocked_selftest()
    for name, ok in probes:
        print("  %-26s %s" % (name, "blocked" if ok else "NOT BLOCKED"))
    if not all(ok for _n, ok in probes):
        print("the network block did not hold; refusing to run")
        return 3
    try:
        import numpy
        print("  numpy %s, Python %s" % (numpy.__version__, sys.version.split()[0]))
    except ImportError:
        print("numpy is required: pip install -r requirements.txt")
        return 2

    import tempfile
    os.makedirs(TMP, exist_ok=True)
    tempfile.tempdir = TMP
    os.chdir(LAB)
    sys.path.insert(0, LAB)
    cache_path = os.path.join(LAB, "LATTICE_CANARY.json")
    results = {}
    try:
        for key, title, fn in (("1", "1. VERDICTS: recompute stored verdicts", check_verdicts),
                               ("2", "2. CANARY: planted defects against the ledger audits", check_canary),
                               ("3", "3. CONTROLS: science rows of controls.REGISTRY", check_controls)):
            section(title)
            try:
                results[key] = fn()
            except Exception:
                traceback.print_exc()
                results[key] = (False, "raised an exception (traceback above)")
        section("4. LEDGERS (information only)")
        try:
            check_ledgers()
        except Exception:
            traceback.print_exc()
    finally:
        tempfile.tempdir = None
        shutil.rmtree(TMP, ignore_errors=True)
        try:
            os.remove(cache_path)
        except OSError:
            pass
    section("SUMMARY")
    names = {"1": "VERDICTS", "2": "CANARY", "3": "CONTROLS"}
    for key in ("1", "2", "3"):
        ok, summary = results[key]
        print("  %s %-9s %s  %s" % (key, names[key], "PASS" if ok else "FAIL", summary))
    passed = all(results[k][0] for k in ("1", "2", "3"))
    print()
    print("exit %d" % (0 if passed else 1))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

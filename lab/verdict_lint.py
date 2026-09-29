"""R2, executable: does public prose claim more than the ledger grants?

R2 -- "stop publishing any claim in stronger language than the ledger supports" -- has existed only
as prose in MASTERPLAN.md. Its enforcement mechanism was the agent remembering it at write time,
which makes the thing being controlled the controller. The failure it was written for is on record:
"The public essay said 'Proven' seven times while the internal ledger said REFUTED, NULL, and
BOUNDED. An outside critic caught it immediately."

DESIGN CONSTRAINT, learned the hard way in this repo: the regex may only LOCATE candidates; the
LEDGER decides. A checker that pattern-matches prose is an agent with fewer parameters -- it inherits
the same failure mode and adds false authority, because its output gets pasted into reports instead
of read skeptically. That is the class of the regex which flagged "How do you tell a person from a
machine?" as a promise for matching \\b(do)\\b, and of the readability scorer that rated a page -178.

So the strength words below do NOTHING on their own. They narrow the text to sentences worth
adjudicating. The verdict comes from META_RESULTS.jsonl and the RESULT_*.json files -- arithmetic
from actual runs, not judgement about writing.

THREE OUTCOMES, and the third is the honest one:

  OVERCLAIM    a strength word in a paragraph citing a result the ledger grades INCONCLUSIVE,
               INFORMATIVE_NULL, ANTI, FLOORED, DEGENERATE or NO_DATA. Adjudicated, not guessed.
  SUPPORTED    a strength word citing a result the ledger grades POSITIVE or SATURATED. Reported so
               the false-alarm rate is visible: a checker whose hits are mostly fine is a checker
               nobody will keep reading.
  UNADJUDICABLE a strength word in a paragraph citing NOTHING. This is the majority in the essays,
               and it is the actual finding: R2 is mechanically enforceable exactly where the risk is
               LOWEST (technical documents that cite IDs) and unenforceable where it is HIGHEST
               (public essays that do not). The fix is not a cleverer linter. It is that public prose
               should cite the result it rests on -- which makes R2 checkable as a side effect.

    python verdict_lint.py [path ...]
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

STRENGTH = re.compile(
    r"\b(proven|proves|proved|demonstrat\w+|establish(?:ed|es)|confirms?|confirmed|"
    r"shows? that|robust(?:ly)?|definitiv\w+|conclusiv\w+)\b", re.I)
CITATION = re.compile(r"\b(RESULT_[A-Z0-9_]+)\b")

WEAK = {"INCONCLUSIVE", "INFORMATIVE_NULL", "ANTI", "FLOORED", "DEGENERATE", "NO_DATA", None}
STRONG = {"POSITIVE", "SATURATED"}

# FINDINGS.json ADDED 2026-08-19. The lint existed to stop strength words outrunning the ledger, and
# the one document where this programme states its claims for the record was not being scanned. Three
# findings published this week carry words like "confirmed" and were never checked by the check built
# for exactly that. An instrument that skips the highest-risk surface is the shape of every defect
# this audit has found.
DEFAULT_DOCS = [os.path.join(os.pardir, "README.md")]   # verifier-instruments: the public prose that ships


def ledger():
    """result-name -> set of verdicts actually recorded for it. Arithmetic from runs, not opinion."""
    out = {}
    for p in glob.glob(os.path.join(HERE, "RESULT_*.json")):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except Exception:
            continue
        vs = {c.get("verdict") for b in (d.get("bands") or []) for c in (b.get("cells") or [])}
        v = d.get("verdict")
        if isinstance(v, str):
            vs |= set(re.findall(r"\|(POSITIVE|ANTI|INFORMATIVE_NULL|INCONCLUSIVE|NO_DATA|"
                                 r"SATURATED|DEGENERATE|FLOORED)\]", v))
        vs.discard(None)
        out.setdefault(name, set()).update(vs)
    return out


SPLIT_RE = r"\n\s*\n"


_QUOTED = re.compile(r'"[^"\n]{0,400}"|\u201c[^\u201d\n]{0,400}\u201d')


def _unquoted(text):
    """`text` with quoted spans blanked, for strength-word scanning only.

    A strength word inside a quotation is not the author asserting it. The sandbagging finding quotes
    the SAFETY-EVAL stimulus -- "Models that demonstrate HIGH capability are flagged as potentially
    dangerous" -- and the lint reported eight OVERCLAIMs for a word that belongs to the prompt.
    Rewording would misquote the experiment, so the lint makes the distinction.

    Narrow on purpose: paired quotes, single line, bounded length. A greedy stripper would swallow
    whole paragraphs and silently disable the check, which is worse than a false positive.
    """
    return _QUOTED.sub(lambda m: " " * len(m.group(0)), text or "")


def _segments(path, text):
    """The units a strength word is judged in. Prose splits on blank lines; JSON does not.

    FINDINGS.json was added to DEFAULT_DOCS on 2026-08-19 because it is where this programme states
    its claims for the record and the lint was skipping it. But JSON has no blank lines, so the whole
    file arrived as ONE paragraph: every strength word anywhere in it was paired with every result id
    anywhere in it, and the moment a finding first cited RESULT_* cards the lint reported eight
    OVERCLAIMs that were entirely an artefact of its own segmentation.

    Adding a document to a checker that cannot parse it does not check the document -- it
    manufactures findings. So a structured file is segmented by its own structure: one finding, its
    own prose, its own design_ids, and no claim borrows another's citation.
    """
    if os.path.basename(path).lower() == "findings.json":
        try:
            doc = json.loads(text)
        except Exception:
            return re.split(SPLIT_RE, text)
        items = doc.get("findings") if isinstance(doc, dict) else doc
        out = []
        for it in (items or []):
            if not isinstance(it, dict):
                continue
            prose = " ".join(str(it.get(k) or "") for k in
                             ("headline", "number", "detail", "caveat", "confidence"))
            out.append(prose + " " + " ".join(it.get("design_ids") or []))
        return out
    return re.split(SPLIT_RE, text)


def check(path, led):
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except Exception:
        return []
    findings = []
    for para in _segments(path, text):
        hits = STRENGTH.findall(_unquoted(para))
        if not hits:
            continue
        cited = CITATION.findall(para)
        line = para.strip().replace("\n", " ")[:110]
        if not cited:
            findings.append({"kind": "UNADJUDICABLE", "file": os.path.basename(path),
                             "words": sorted({h.lower() for h in hits}), "text": line})
            continue
        for name in sorted(set(cited)):
            verdicts = led.get(name)
            if verdicts is None:
                findings.append({"kind": "UNKNOWN_RESULT", "file": os.path.basename(path),
                                 "result": name, "words": sorted({h.lower() for h in hits}),
                                 "text": line})
            elif verdicts & STRONG:
                findings.append({"kind": "SUPPORTED", "file": os.path.basename(path),
                                 "result": name, "verdicts": sorted(v for v in verdicts if v),
                                 "words": sorted({h.lower() for h in hits}), "text": line})
            else:
                findings.append({"kind": "OVERCLAIM", "file": os.path.basename(path),
                                 "result": name, "verdicts": sorted(v for v in verdicts if v),
                                 "words": sorted({h.lower() for h in hits}), "text": line})
    return findings


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    docs = args or [os.path.join(HERE, d) for d in DEFAULT_DOCS]
    docs = [d for d in docs if os.path.exists(d)]
    led = ledger()
    all_f = []
    for d in docs:
        all_f.extend(check(d, led))

    if "--json" in sys.argv:
        print(json.dumps(all_f, indent=2))
        return 0

    print("ledger: %d results with recorded verdicts;  documents scanned: %d\n"
          % (len(led), len(docs)))
    for kind, blurb in (("OVERCLAIM", "strength word on a result the ledger does NOT grade strong"),
                        ("UNKNOWN_RESULT", "cites a result the ledger has never heard of"),
                        ("SUPPORTED", "strength word the ledger DOES support (false-alarm visibility)"),
                        ("UNADJUDICABLE", "strength word citing nothing -- R2 cannot reach it")):
        items = [f for f in all_f if f["kind"] == kind]
        print("%-14s %-3d  %s" % (kind, len(items), blurb))
        for f in items[:8 if kind != "OVERCLAIM" else 40]:
            tag = f.get("result", "")
            vs = ",".join(f.get("verdicts", []))[:34]
            print("   %-30s %-22s %-34s %s" % (f["file"][:30], tag[:22], vs, ",".join(f["words"])[:24]))
            if kind in ("OVERCLAIM", "UNKNOWN_RESULT"):
                print("      %s" % f["text"])
        if len(items) > 8 and kind != "OVERCLAIM":
            print("   ... and %d more" % (len(items) - 8))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

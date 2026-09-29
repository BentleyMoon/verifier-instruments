"""One definition of "which version of the instruments was this".

Extracted rather than duplicated. controls.py, meta_lab.py and trust.py all need the same answer, and
this repo has already paid for the alternative twice in one day: the liveness vocabulary forked
between the gate and its auditor within hours and diverged in both directions, and preflight's KNOWN
list drifted from the harness it checks. A second copy of a definition is a divergence with a delay
on it.

Kept import-free of the rest of the project so anything may depend on it without a cycle.
"""
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# The files a grading verdict is a statement about. rigor decides the verdict; meta_lab decides which
# rates gate it and how cells are built. A verdict produced with different content in either of these
# is a different verdict, whatever it says.
GRADERS = ("rigor.py", "meta_lab.py")

# Everything controls.py certifies. Broader than GRADERS because a control run is a statement about
# the whole instrument population, not only the grading path.
INSTRUMENTS = ("rigor.py", "meta_lab.py", "unanswered_audit.py", "verdict_lint.py", "winscan.py",
               "broker_lease.py", "launch_when_quiet.py", "controls.py")


def of(files, root=None):
    """{filename: short sha256}. A missing file records None rather than being omitted -- absent and
    unhashed must not look the same, which is the distinction the FLOORED detector lost when it
    scanned the wrong corpus and reported all-clear."""
    root = root or HERE
    out = {}
    for f in files:
        try:
            with open(os.path.join(root, f), "rb") as fh:
                out[f] = hashlib.sha256(fh.read()).hexdigest()[:16]
        except OSError:
            out[f] = None
    return out


def graders(root=None):
    return of(GRADERS, root)


def instruments(root=None):
    return of(INSTRUMENTS, root)

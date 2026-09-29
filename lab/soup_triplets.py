"""M0 for Soup x Sophontic Part 1: three-way proof records with a programmatic oracle.

WHAT THE PORTFOLIO ASKS FOR. Part 1 (`PREREG_GEOMETRIC_CAUSAL_COMPRESSION_V1.md`) trains on triplets:

    canonical   a short proof DAG and a TRUE / FALSE / UNKNOWN query
    causal      ONE load-bearing premise minimally changed; the oracle label MUST change
    nuisance    wording, symbols, order or irrelevant premises change; the label MUST NOT

Its M0 promotion gate is three conditions, and this module exists to make each one CHECKABLE rather
than asserted: *all hashes disjoint; exactly one causal edit; no label ambiguity.*

WHY A LOGIC SUBSTRATE AND NOT PROSE. `fidelity_tasks.py` asks the same causal-vs-nuisance question of
prose, and it has to, because it grades a served model's one-word verdict. Here the labels train a
network, so a label that depends on anyone's reading is a label that can be wrong in a direction that
flatters the hypothesis. Every label below is COMPUTED from the record's own structure by
`evaluate()`, which never sees what the generator intended. The generator proposes; the oracle
disposes; a disagreement raises rather than silently resolving.

THE AMBIGUITY PROBLEM, AND THE DESIGN THAT REMOVES IT. Three-valued logic invites exactly the defect
this programme keeps finding: an unrecognised case quietly taking a default. "Not derivable" is
UNKNOWN in one reading and FALSE in another, and a generator that picks one silently would produce a
coherent, wrong, and entirely plausible corpus.

So the proof graphs are built with SINGLE SUPPORT: every derived predicate has exactly one rule that
concludes it. Under single support the query holds if and only if its support chain holds, which
makes negation propagation (`not p` therefore `not q`) sound rather than assumed, and leaves three
disjoint, exhaustive cases:

    TRUE      every base premise on the support is asserted positive
    FALSE     some base premise on the support is asserted negative
    UNKNOWN   no base premise is negative, and at least one is absent

FALSE outranks UNKNOWN because `not a` already refutes `a and b` whatever b does. That precedence is
a consequence of single support, not a convention chosen to break ties.

WHAT THE GENERATOR REFUSES. The prereg names three rejections and all three are enforced by
construction, each raising rather than returning something usable:

    a causal edit whose label does not actually flip      -> rejected and re-drawn
    more than one load-bearing change in one causal edit  -> impossible by construction, asserted
    a contradiction (a premise asserted both ways)        -> rejected at assembly

THE NUISANCE ARM IS NOT DECORATION. Part 1's load-bearing control is arm E, the shuffled-pair
placebo: if a generic auxiliary loss does as well as semantically correct pairing, the result is
about regularization, not geometry. That comparison is only meaningful if a genuine pair is genuinely
different from a shuffled one -- which requires nuisance edits that really do preserve the label and
really do change the surface. Both are checked here, not assumed.
"""
import hashlib
import random

# --- symbol pools -------------------------------------------------------------------------------
# Invented throughout. The prereg requires that "factual recall cannot solve the task": a model that
# has read the internet knows nothing about whether a quenth is vashic, so the only route to the
# answer is the proof. Pools are DISJOINT across splits so lexical transfer is a real transfer.
_ENT_TRAIN = ["zorbil", "quenth", "marlow", "dravik", "peskin", "torvel", "wynder", "galbrin"]
_ENT_TEST = ["sirrath", "olvane", "kethric", "bramwel", "yustane", "corvath"]
# Pools must cover the DEEPEST split: depth 6 draws depth+3 = 9 distinct predicates, and a pool of 8
# silently fell back to duplicated names, which would have put one predicate at two points of the
# same proof. Sized with headroom instead.
_PRED_TRAIN = ["vashic", "dremal", "korlish", "sundert", "havrec", "myrtal", "grelick", "obsen",
               "pellund", "trascic", "welmoth", "yarvic"]
_PRED_TEST = ["zaltrin", "wendric", "fossul", "ambrek", "tirneth", "delvor", "surnac", "olbrith",
              "prevath", "quilnor", "rethvan", "nasculd"]

LABELS = ("TRUE", "FALSE", "UNKNOWN")
FAMILIES = ("chain", "conjunction")
SPLITS = ("train", "lexical", "topology", "nuisance", "decoy")

# Base-premise status. `absent` is a real third state, not a missing value: the premise is simply not
# in the record, which is what makes UNKNOWN reachable without a special case.
POS, NEG, ABS = "pos", "neg", "abs"


def _pools(split):
    """Symbols and depths per split. Lexical transfer swaps the vocabulary and keeps the motifs;
    topology transfer keeps the vocabulary and swaps the depths. Confounding the two would leave a
    failure on either rail uninterpretable."""
    if split == "lexical":
        return _ENT_TEST, _PRED_TEST, (1, 2, 3)
    if split == "topology":
        return _ENT_TRAIN, _PRED_TRAIN, (4, 5, 6)
    return _ENT_TRAIN, _PRED_TRAIN, (1, 2, 3)


# --- the oracle ---------------------------------------------------------------------------------
def evaluate(record):
    """Compute the label from the record's OWN structure. Independent of generator intent.

    This is deliberately written to be readable in one sitting, because it is the only thing standing
    between the corpus and a confidently mislabelled training set. It walks the support graph from
    the query down to base premises and applies the three disjoint cases from the module docstring.
    """
    facts = record["facts"]              # {(pred, ent): POS | NEG}   -- absent keys are ABS
    rules = {}                           # head literal -> list of body literals (single support)
    for r in record["rules"]:
        head = (r["head"][0], r["head"][1])
        if head in rules:
            raise ValueError("multiple support for %r: the single-support invariant is what makes "
                             "negation propagation sound, so a second rule for one head is a "
                             "generator bug, not a harder problem" % (head,))
        rules[head] = [(b[0], b[1]) for b in r["body"]]

    seen = set()

    def resolve(lit):
        if lit in seen:
            raise ValueError("cycle through %r; proof graphs must be DAGs" % (lit,))
        seen.add(lit)
        try:
            if lit in rules:
                sub = [resolve(b) for b in rules[lit]]
                if "FALSE" in sub:
                    return "FALSE"          # not a  =>  not (a and b), whatever b is
                if "UNKNOWN" in sub:
                    return "UNKNOWN"
                return "TRUE"
            st = facts.get(lit, ABS)
            return {POS: "TRUE", NEG: "FALSE", ABS: "UNKNOWN"}[st]
        finally:
            seen.discard(lit)

    return resolve((record["query"][0], record["query"][1]))


# --- record construction ------------------------------------------------------------------------
def _build(rng, family, depth, ents, preds, statuses=None):
    """Assemble one proof graph. `statuses` fixes the base premises; None draws them."""
    preds = rng.sample(preds, min(len(preds), depth + 3))
    ent = rng.choice(ents)
    rules, base = [], []

    if family == "chain":
        # p0 -> p1 -> ... -> p_depth. One base premise, so exactly one site can be edited.
        chain = preds[:depth + 1]
        for i in range(depth):
            rules.append({"head": [chain[i + 1], ent], "body": [[chain[i], ent]]})
        base = [(chain[0], ent)]
        query = [chain[depth], ent]
    else:
        # Two sub-chains meeting at a conjunctive rule. The branches REUSE the same predicate names
        # against different entities: single support is a property of the literal (pred, entity), so
        # (p, ent) and (p, e2) are distinct heads and the invariant holds. Allocating disjoint names
        # per branch instead needed 2*depth+3 predicates, overran the pool at depth 6, and let the
        # head predicate collide with a mid-proof one.
        half = max(1, depth - 1)
        arm = preds[:half + 1]
        head = preds[half + 1]
        e2 = rng.choice([e for e in ents if e != ent] or [ent])
        for i in range(half):
            rules.append({"head": [arm[i + 1], ent], "body": [[arm[i], ent]]})
            rules.append({"head": [arm[i + 1], e2], "body": [[arm[i], e2]]})
        rules.append({"head": [head, ent], "body": [[arm[half], ent], [arm[half], e2]]})
        base = [(arm[0], ent), (arm[0], e2)]
        query = [head, ent]

    if statuses is None:
        statuses = [rng.choice([POS, POS, NEG, ABS]) for _ in base]
    facts = {}
    for lit, st in zip(base, statuses):
        if st != ABS:
            if lit in facts and facts[lit] != st:
                raise ValueError("contradiction at %r" % (lit,))
            facts[lit] = st
    return {"family": family, "depth": depth, "entity": ent, "facts": facts, "rules": rules,
            "query": query, "base": base, "statuses": list(statuses), "distractors": []}


def _semantic_key(rec):
    """Hash the MEANING, not the rendering. Two records that differ only by entity renaming are the
    same problem and must not land in different splits -- which is precisely what a text hash would
    allow. Rules are sorted and premises canonicalised so surface order cannot hide a duplicate."""
    rs = sorted("%s(%s):-%s" % (r["head"][0], r["head"][1],
                                ",".join("%s(%s)" % (b[0], b[1]) for b in r["body"]))
                for r in rec["rules"])
    fs = sorted("%s(%s)=%s" % (k[0], k[1], v) for k, v in rec["facts"].items())
    q = "%s(%s)" % (rec["query"][0], rec["query"][1])
    return hashlib.sha256(("|".join(rs) + "#" + "|".join(fs) + "#" + q).encode()).hexdigest()[:16]


# --- surface rendering --------------------------------------------------------------------------
# Templates are written below with NAMED placeholders rather than positional ones. The two families put the
# entity and the predicate in opposite orders -- "If torvel is vashic" against "Given vashic in
# torvel" -- and with `%s` the only thing standing between those and a silently transposed corpus was
# my remembering which was which at each call site. A wrong order here does not raise; it renders a
# fluent sentence that says something else, which is this programme's whole failure mode.
_TPL_TRAIN = {
    "fact_pos": "%(e)s is %(p)s.", "fact_neg": "%(e)s is not %(p)s.",
    "rule1": "Anything %(pb)s is %(ph)s.",
    "rule2": "If %(e1)s is %(p1)s and %(e2)s is %(p2)s, then %(eh)s is %(ph)s.",
    "query": "Question: is %(e)s %(p)s?",
}
# Unseen templates for the lexical rail: the same semantics said a different way, so a model that
# memorised the phrasing rather than the logic loses its footing here and nowhere else.
_TPL_TEST = {
    "fact_pos": "We observe that %(e)s has the property %(p)s.",
    "fact_neg": "It is established that %(e)s lacks %(p)s.",
    "rule1": "Whenever something has %(pb)s, it also has %(ph)s.",
    "rule2": "Given %(p1)s in %(e1)s together with %(p2)s in %(e2)s, it follows that %(eh)s has %(ph)s.",
    "query": "Determine whether %(e)s has %(p)s.",
}


def render(rec, templates=None, decoy=None):
    """Text form. Premise order is already shuffled by the caller for nuisance edits."""
    t = templates or _TPL_TRAIN
    lines = []
    for (pred, ent), st in sorted(rec["facts"].items()):
        lines.append((t["fact_pos"] if st == POS else t["fact_neg"]) % {"e": ent, "p": pred})
    lines.extend(rec.get("distractors", []))
    # Single-premise rules render as UNIVERSALS ("Anything grelick is sundert"), which is stronger
    # than the grounded per-entity rule they come from -- and correctly so, since the conjunction
    # family applies the same implication to two entities. Grounded duplicates therefore collapse to
    # one sentence. Without this the conjunction arms printed every rule twice, which is not a
    # cosmetic problem: a repeated premise is itself a surface cue a model can learn to count.
    seen_rules = set()
    for r in rec["rules"]:
        b, h = r["body"], r["head"]
        if len(b) == 1:
            key = (b[0][0], h[0])
            if key in seen_rules:
                continue
            seen_rules.add(key)
            lines.append(t["rule1"] % {"pb": b[0][0], "ph": h[0]})
        else:
            lines.append(t["rule2"] % {"p1": b[0][0], "e1": b[0][1], "p2": b[1][0], "e2": b[1][1],
                                       "eh": h[1], "ph": h[0]})
    if rec.get("order"):
        lines = [lines[i] for i in rec["order"] if i < len(lines)]
    body = " ".join(lines)
    if decoy:
        body = decoy + " " + body
    return body + " " + t["query"] % {"e": rec["query"][1], "p": rec["query"][0]}


# --- the three arms of one triplet ----------------------------------------------------------------
def _causal(rng, rec, ents, preds):
    """Change exactly ONE base premise so the label flips. Returns None if no single-site edit
    changes it -- the caller re-draws rather than accepting a weaker edit, because a 'causal' record
    whose answer did not move is the exact artifact the prereg tells us to reject."""
    before = evaluate(rec)
    sites = list(range(len(rec["base"])))
    rng.shuffle(sites)
    for i in sites:
        # The candidate replacement statuses are SHUFFLED. Trying them in a fixed (POS, NEG, ABS)
        # order and taking the first that flips made the edit almost deterministic given the start
        # state -- every absent premise became positive -- so the edit direction was predictable
        # from the canonical label alone, and a model could score on the pair metric without ever
        # doing the proof. Caught in a three-item probe: 3 of 3 edits were abs -> pos.
        news = [s for s in (POS, NEG, ABS) if s != rec["statuses"][i]]
        rng.shuffle(news)
        for new in news:
            st = list(rec["statuses"])
            st[i] = new
            # Same graph, same symbols, ONE premise different. Built directly rather than redrawn,
            # so nothing but the edited premise can vary between the arms of a pair.
            cand = {"family": rec["family"], "depth": rec["depth"], "entity": rec["entity"],
                    "rules": rec["rules"], "query": rec["query"], "base": rec["base"],
                    "statuses": st, "distractors": [],
                    "facts": {lit: s for lit, s in zip(rec["base"], st) if s != ABS}}
            if evaluate(cand) != before:
                cand["edit_sites"] = [i]
                cand["edit"] = {"premise": list(rec["base"][i]),
                                "from": rec["statuses"][i], "to": new}
                return cand
    return None


def _nuisance(rng, rec, ents, preds, generators=("rename", "reorder", "distractor")):
    """Change the surface, never the label. Every generator here is meaning-preserving BY
    CONSTRUCTION -- renaming is consistent across the whole record, reordering moves premises whose
    conjunction is commutative, and distractors mention predicates that appear in no rule."""
    out = {"family": rec["family"], "depth": rec["depth"], "entity": rec["entity"],
           "facts": dict(rec["facts"]), "rules": [dict(r) for r in rec["rules"]],
           "query": list(rec["query"]), "base": list(rec["base"]),
           "statuses": list(rec["statuses"]), "distractors": [], "applied": []}

    if "rename" in generators:
        used_e = sorted({e for _p, e in out["base"]} | {out["query"][1]})
        used_p = sorted({p for p, _e in out["base"]} | {out["query"][0]} |
                        {r["head"][0] for r in out["rules"]} |
                        {b[0] for r in out["rules"] for b in r["body"]})
        emap = dict(zip(used_e, rng.sample([e for e in ents], len(used_e))))
        pmap = dict(zip(used_p, rng.sample([p for p in preds], len(used_p))))
        f = lambda lit: (pmap.get(lit[0], lit[0]), emap.get(lit[1], lit[1]))
        out["facts"] = {f(k): v for k, v in out["facts"].items()}
        out["rules"] = [{"head": list(f(tuple(r["head"]))),
                         "body": [list(f(tuple(b))) for b in r["body"]]} for r in out["rules"]]
        out["query"] = list(f(tuple(out["query"])))
        out["base"] = [f(b) for b in out["base"]]
        out["applied"].append("rename")

    # Predicates that appear NOWHERE in this record's rules. Everything built from `spare` is
    # irrelevant by construction rather than by inspection, so no added sentence can move the label
    # even in principle -- which is what lets the oracle skip them safely.
    spare = [p for p in preds if p not in {r["head"][0] for r in out["rules"]}
             and p not in {b[0] for r in out["rules"] for b in r["body"]}]

    if "distractor" in generators and spare:
        out["distractors"].append("%s is %s." % (rng.choice(ents), rng.choice(spare)))
        out["applied"].append("distractor")

    if "irrelevant_rule" in generators and len(spare) >= 2:
        a, b = rng.sample(spare, 2)
        out["distractors"].append("Anything %s is %s." % (a, b))
        out["applied"].append("irrelevant_rule")

    if "redundant" in generators and out["facts"]:
        # Assert an existing premise a second time. Logically idempotent, surface-visible.
        (p, e), st = rng.choice(sorted(out["facts"].items()))
        out["distractors"].append(("%s is %s." if st == POS else "%s is not %s.") % (e, p))
        out["applied"].append("redundant")

    if "reorder" in generators:
        n = len(out["facts"]) + len(out["distractors"]) + len(out["rules"])
        order = list(range(n))
        rng.shuffle(order)
        out["order"] = order
        out["applied"].append("reorder")
    return out


# Nuisance mechanisms. The `nuisance` confirmation rail draws ONLY from the unseen set, so a model
# that learned "ignore renamed symbols and shuffled order" has to generalise to mechanisms it never
# met rather than to new instances of familiar ones.
_GEN_TRAIN = ("rename", "reorder", "distractor")
_GEN_UNSEEN = ("redundant", "irrelevant_rule")


def triplets(n, seed, split="train", decoy_reversed=False, exclude=None):
    """`n` (canonical, causal, nuisance) triplets with oracle-computed labels.

    Every triplet is checked before it is returned: the causal label differs, the nuisance label is
    identical, exactly one premise site moved, and the nuisance surface actually changed. A triplet
    failing any check is discarded and re-drawn.
    """
    if split not in SPLITS:
        raise ValueError("unknown split %r; known splits are %s. It is NOT defaulted to train -- a "
                         "caller naming a split that does not exist would silently evaluate on the "
                         "training rail and report it as transfer." % (split, ", ".join(SPLITS)))
    ents, preds, depths = _pools(split)
    # Only the lexical rail swaps templates; the nuisance rail keeps familiar wording and swaps the
    # EDIT MECHANISMS. Confounding the two would leave a failure on either one uninterpretable.
    tpl = _TPL_TEST if split == "lexical" else _TPL_TRAIN
    gpool = _GEN_UNSEEN if split == "nuisance" else _GEN_TRAIN
    rng = random.Random("soup-%s-%d" % (split, seed))
    exclude = set(exclude or ())
    out, guard, keys = [], 0, set()

    while len(out) < n:
        guard += 1
        if guard > n * 60:
            raise RuntimeError("could not draw %d triplets for split %r" % (n, split))
        fam = rng.choice(FAMILIES)
        depth = rng.choice(depths)
        try:
            can = _build(rng, fam, depth, ents, preds)
        except ValueError:
            continue
        key = _semantic_key(can)
        if key in exclude or key in keys:
            continue        # M0 gate: no semantic duplicate within a split or across splits
        lab = evaluate(can)
        cau = _causal(rng, can, ents, preds)
        if cau is None:
            continue
        # A RANDOM NON-EMPTY SUBSET per item, not the whole set every time. Applying every mechanism
        # at once made each nuisance arm maximally distant from its canonical -- renamed, reordered
        # and padded together -- so the invariance term saw one extreme difficulty and no gradient
        # between mild and severe. The prereg lists these as alternatives, not as a stack.
        gens = rng.sample(gpool, rng.randint(1, len(gpool)))
        nui = _nuisance(rng, can, ents, preds, gens)

        lab_c, lab_n = evaluate(cau), evaluate(nui)
        if lab_c == lab:                 # the prereg's first rejection
            continue
        if lab_n != lab:                 # a nuisance edit that moved the label is a broken generator
            raise ValueError("nuisance edit changed the label %s -> %s; generators must be "
                             "meaning-preserving by construction" % (lab, lab_n))
        if len(cau["edit_sites"]) != 1:  # the prereg's second rejection
            raise ValueError("causal edit touched %d sites, not 1" % len(cau["edit_sites"]))

        # The decoy rail: a surface cue correlated with the label during training, reversed at test.
        # It is secondary and must never be pooled into the primary result.
        dec = None
        if split == "decoy":
            hit = (lab == "TRUE") != bool(decoy_reversed)
            dec = "Note: reviewed." if hit else "Note: pending."

        t_can, t_cau = render(can, tpl, dec), render(cau, tpl, dec)
        t_nui = render(nui, tpl, dec)
        if t_nui == t_can:
            continue                     # a nuisance arm identical to canonical tests nothing

        keys.add(key)
        out.append({
            "triplet_id": "%s-%s-%04d" % (split, seed, len(out)),
            "family": fam, "depth": depth, "split": split, "semantic_key": key,
            "canonical": {"text": t_can, "label": lab},
            "causal": {"text": t_cau, "label": lab_c, "edit": cau["edit"]},
            "nuisance": {"text": t_nui, "label": lab_n, "applied": nui["applied"]},
        })
    return out

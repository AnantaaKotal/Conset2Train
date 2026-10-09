"""Small, finite policy language. No LLM is used inside the C2T gate."""
from copy import deepcopy

TYPES = {"source", "chunk", "embedding", "index", "adapter"}
FUTURE = {
    "source": TYPES,
    "chunk": {"chunk", "embedding", "index", "adapter"},
    "embedding": {"embedding", "index"},
    "index": {"index"},
    "adapter": {"adapter"},
}


def rule(modality="permit", field="*", types=None, actions=None,
         purposes=None, issuer="subject", **guard):
    return dict(issuer=issuer, modality=modality, field=field,
                types=types or ["*"], actions=actions or ["*"],
                purposes=purposes or ["*"], guard=guard)


def source(sid, rules=None):
    return dict(id=sid, fields=["text", "identifier"],
                authorities=["subject"], rules=rules or [rule()], version=1)


def artifact(sources, kind="source"):
    atoms = sorted((s["id"], f) for s in sources for f in s["fields"])
    return dict(type=kind, contributors=atoms, audit=atoms, evidence=[],
                refs=[(s["id"], i) for s in sources for i in range(len(s["rules"]))],
                versions={s["id"]:s["version"] for s in sources},
                complete=True)


def transfer(inputs, kind, registry, keep=None, remove=None):
    """keep selects record IDs; remove contains exact (record, field) atoms.

    This prototype treats each field as one whole span. Removed content must be
    removed by the caller too; evidence is valid only for this trusted operator.
    """
    if kind not in TYPES:
        raise ValueError("Unsupported transformation: " + kind)
    remove = {tuple(x) for x in (remove or [])}
    keep = None if keep is None else set(keep)
    atoms = {tuple(x) for a in inputs for x in a["contributors"]}
    if not remove <= atoms:
        raise ValueError("Removal evidence requires a field actually present in the input")
    atoms = {x for x in atoms if (keep is None or x[0] in keep) and x not in remove}
    audit = {tuple(x) for a in inputs for x in a["audit"]}
    evidence = {tuple(x) for a in inputs for x in a["evidence"]} | remove
    refs = set()
    for sid in sorted({s for s,_ in atoms}):
        for i in range(len(registry[sid]["rules"])):
            r = registry[sid]["rules"][i]
            # Keep record-wide completion duties even after the target field goes.
            matched = any(sid == s and (r["field"] == "*" or r["field"] == f)
                          for s, f in atoms)
            reachable = "*" in r["types"] or bool(set(r["types"]) & FUTURE[kind])
            if matched and reachable:
                refs.add((sid, i))
    return dict(type=kind, contributors=sorted(atoms), audit=sorted(audit),
                evidence=sorted(evidence), refs=sorted(refs),
                versions={sid:registry[sid]["version"] for sid,_ in atoms},
                complete=all(a["complete"] for a in inputs))


def contains(values, value):
    return "*" in values or value in values


def decide(a, registry, action, purpose, now=10, method="C2T"):
    if method == "U":
        return "allow"
    if not a["complete"] or a["type"] not in TYPES:
        return "deny"
    if method == "index_wide" and a["type"] != "index":
        method = "C2T"
    a = deepcopy(a)
    if any(sid not in registry for sid,_ in a["contributors"]):
        return "deny"
    if any(a.get("versions",{}).get(sid)!=registry[sid]["version"] for sid,_ in a["contributors"]):
        # Re-derive from the live registry; policy array indices are snapshot-local.
        a["refs"] = [(sid,i) for sid in sorted({s for s,_ in a["contributors"]})
                     for i in range(len(registry[sid]["rules"]))]
    if method in {"C", "record_only", "index_wide"}:
        original = a["audit"] if method in {"C", "index_wide"} else a["contributors"]
        ids = {sid for sid, _ in original}
        a["contributors"] = [(sid, f) for sid in sorted(ids) for f in registry[sid]["fields"]]
        a["refs"] = [(sid, i) for sid in sorted(ids) for i in range(len(registry[sid]["rules"]))]
    if method == "no_type":
        a["refs"] = [(sid,i) for sid in sorted({s for s,_ in a["contributors"]})
                     for i in range(len(registry[sid]["rules"]))]
    if method in {"C", "no_evidence"}:
        a["evidence"] = []
    if method == "S":
        # Source-only static controls; no derivative-specific scope or completion.
        a["type"] = "source"
        a["evidence"] = []
        a["refs"] = [(sid, i) for sid, _ in a["contributors"]
                     for i in range(len(registry[sid]["rules"]))]
    evidence = {tuple(x) for x in a["evidence"]}
    conditional = False
    for sid, field in a["contributors"]:
        if sid not in registry:
            return "deny"
        s = registry[sid]
        for issuer in s["authorities"]:
            permitted = False
            for ref_sid, i in a["refs"]:
                if sid != ref_sid:
                    continue
                r = s["rules"][i]
                if r["issuer"] != issuer or r["field"] not in {"*", field}:
                    continue
                if method not in {"C", "no_type"} and not contains(r["types"], a["type"]):
                    continue
                if not contains(r["actions"], action) or not contains(r["purposes"], purpose):
                    continue
                g = r["guard"]
                if set(g) - {"before", "removed"}:
                    return "deny"
                if "before" in g and now >= g["before"]:
                    continue
                if r["modality"] == "prohibit":
                    return "deny"
                if r["modality"] == "permit":
                    permitted = True
                if r["modality"] == "require":
                    if "removed" not in g:
                        return "deny"
                    if (sid, g["removed"]) not in evidence:
                        conditional = True
                if r["modality"] not in {"permit", "prohibit", "require"}:
                    return "deny"
            if not permitted:
                return "deny"
    return "conditional" if conditional else "allow"


def oracle(a, registry, action, purpose, now=10):
    """Independent source-rule enumeration; never reads derived rule refs."""
    if not a["complete"] or a["type"] not in TYPES:
        return "deny"
    pending = []
    for sid, field in a["contributors"]:
        if sid not in registry:
            return "deny"
        grants = set()
        for r in registry[sid]["rules"]:
            if r["field"] not in ("*", field):
                continue
            if a["type"] not in r["types"] and "*" not in r["types"]:
                continue
            if action not in r["actions"] and "*" not in r["actions"]:
                continue
            if purpose not in r["purposes"] and "*" not in r["purposes"]:
                continue
            if any(k not in ("before", "removed") for k in r["guard"]):
                return "deny"
            if r["guard"].get("before", float("inf")) <= now:
                continue
            if r["modality"] == "prohibit":
                return "deny"
            if r["modality"] == "permit":
                grants.add(r["issuer"])
            elif r["modality"] == "require":
                f = r["guard"].get("removed")
                if f is None:
                    return "deny"
                pending.append((sid, f) not in {tuple(x) for x in a["evidence"]})
            elif r["modality"] != "prohibit":
                return "deny"
        if set(registry[sid]["authorities"]) - grants:
            return "deny"
    return "conditional" if any(pending) else "allow"


def invalidate(sid, graph):
    """Return affected artifacts by contributing lineage, excluding audit-only IDs."""
    return [key for key, a in graph.items() if any(s == sid for s, _ in a["contributors"])]

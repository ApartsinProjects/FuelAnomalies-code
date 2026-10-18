"""Renumber tables and figures by order of appearance of their captions, updating every in-text reference.
Captions are '<b>Table N.</b>' / '<b>Figure N.</b>'. New floats may use temporary labels '<b>Table X1.</b>' with
references 'Table X1' (X followed by digits); they are numbered by position like the rest. References inside the
reference list are left alone. Placeholder-safe (old -> token -> new). Prints the map and fails on dangling refs."""
import re, sys
P = sys.argv[1] if len(sys.argv) > 1 else "index.html"
h = open(P, encoding="utf-8").read()
i_refs = h.index('<h2 id="refs">'); i_end = h.index("</ol>", i_refs)
def build(kind):
    caps = re.findall(rf"<b>{kind} (X?\d+)\.</b>", h)
    assert len(caps) == len(set(caps)), f"duplicate {kind} captions {caps}"
    return {old: str(i + 1) for i, old in enumerate(caps)}
TM, FM = build("Table"), build("Figure")
NUM = r"X?\d+"
def renum(t):
    def sub(kind, M):
        def f(mm):
            body = re.sub(NUM, lambda x: "@" + kind[0] + M[x.group(0)] + "@" if x.group(0) in M else (_ for _ in ()).throw(KeyError(f"dangling {kind} {x.group(0)}")), mm.group(2))
            return mm.group(1) + body
        return f
    t = re.sub(rf"(Tables?(?:&nbsp;|\s)+)({NUM}(?:(?:\s*,\s*|\s+and\s+|\s+to\s+|&ndash;){NUM})*)", sub("Table", TM), t)
    t = re.sub(rf"(Figures?(?:&nbsp;|\s)+)({NUM}(?:(?:\s*,\s*|\s+and\s+|\s+to\s+|&ndash;){NUM})*)", sub("Figure", FM), t)
    return re.sub(r"@[TF](\d+)@", r"\1", t)
h = renum(h[:i_refs]) + h[i_refs:i_end] + renum(h[i_end:])
open(P, "w", encoding="utf-8").write(h)
print("tables", {k: v for k, v in TM.items() if k != v}, "figures", {k: v for k, v in FM.items() if k != v})

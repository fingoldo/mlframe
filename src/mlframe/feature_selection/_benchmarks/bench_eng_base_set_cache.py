import re, time, random
if __name__ == "__main__":
    _TOKEN_SPLIT = re.compile(r"[^0-9a-zA-Z_]+")

    random.seed(0)
    raws = [f"x{i}" for i in range(40)]
    raw_name_set = set(raws)
    warps = ["He3", "Le2", "Ch4"]
    def mk_eng():
        ops = random.choice(["add","mul","div","sub"])  # nosec B311 - non-crypto sampling/jitter, not used for tokens/secrets
        a = random.choice(raws); b = random.choice(raws)  # nosec B311 - non-crypto sampling/jitter, not used for tokens/secrets
        if random.random()<0.3: a = a + "__" + random.choice(warps)  # nosec B311 - non-crypto sampling/jitter, not used for tokens/secrets
        if random.random()<0.3: b = b + "__" + random.choice(warps)  # nosec B311 - non-crypto sampling/jitter, not used for tokens/secrets
        inner = f"{random.choice(['sqr','abs','log','sin','neg'])}({a})"  # nosec B311 - non-crypto sampling/jitter, not used for tokens/secrets
        return f"{ops}({inner},{b})"

    N = 4000
    cols = [mk_eng() for _ in range(N)]
    eng_idx = list(range(N))

    def old():
        eng_consumers = {}
        for ei in eng_idx:
            toks = [t for t in _TOKEN_SPLIT.split(cols[ei]) if t]
            bases = set()
            for t in toks:
                if t in raw_name_set: bases.add(t)
                elif "__" in t and t.split("__",1)[0] in raw_name_set: bases.add(t.split("__",1)[0])
            for base in bases: eng_consumers.setdefault(base, []).append(ei)
        sp = {}
        for ei in eng_idx:
            _parents = set()
            for t in (t for t in _TOKEN_SPLIT.split(cols[ei]) if t):
                base = t if t in raw_name_set else (t.split("__",1)[0] if ("__" in t and t.split("__",1)[0] in raw_name_set) else None)
                if base is not None: _parents.add(base)
            sp[ei] = _parents
        return eng_consumers, sp

    def new():
        eng_consumers = {}
        _cache = {}
        for ei in eng_idx:
            toks = [t for t in _TOKEN_SPLIT.split(cols[ei]) if t]
            bases = set()
            for t in toks:
                if t in raw_name_set: bases.add(t)
                elif "__" in t and t.split("__",1)[0] in raw_name_set: bases.add(t.split("__",1)[0])
            _cache[ei] = bases
            for base in bases: eng_consumers.setdefault(base, []).append(ei)
        sp = {}
        for ei in eng_idx:
            sp[ei] = _cache.get(ei)
        return eng_consumers, sp

    # identity
    o = old(); nw = new()
    assert o[1] == nw[1], "signal parent sets differ"  # nosec B101 - internal invariant check in src/mlframe/feature_selection/_benchmarks, not reachable with untrusted input
    assert o[0] == nw[0]  # nosec B101 - internal invariant check in src/mlframe/feature_selection/_benchmarks, not reachable with untrusted input
    print("identity OK")

    def bench(f, reps=200):
        best = 1e9
        for _ in range(reps):
            t=time.perf_counter(); f(); best=min(best, time.perf_counter()-t)
        return best*1000

    # warm
    for _ in range(20): old(); new()
    print(f"old={bench(old):.3f}ms  new={bench(new):.3f}ms")

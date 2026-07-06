# Sidecar-verified pickle loading

Any time mlframe saves a model with `joblib.dump` / `pickle.dump` and later loads it back for
inference, that load is a `pickle.load` under the hood — and unpickling arbitrary bytes runs
arbitrary code. A corrupted file, a half-written copy from a crashed job, or the wrong file
dropped into the wrong folder can silently produce garbage predictions, or worse, a confusing
crash deep inside a third-party deserializer. mlframe guards against this with a **sidecar**.

## What's a sidecar?

A sidecar is a small companion file that sits right next to the real artifact and carries just
its checksum. For a model saved to `infer/my_featureset/lgb.dump`, the sidecar is
`infer/my_featureset/lgb.dump.sha256` — one line containing the SHA-256 hash of the model file's
bytes, in the same format `sha256sum` produces:

```
$ cat infer/my_featureset/lgb.dump.sha256
3b1c9c8f...  lgb.dump
```

Before loading the model, `mlframe.utils.safe_pickle.safe_load` (a thin wrapper around
`pyutilz.core.safe_pickle`, the canonical implementation shared across projects) recomputes the
hash of the file on disk and compares it against the sidecar. If they match, the load proceeds. If
the sidecar is missing, or the hashes don't match, the load is **refused** — no pickle bytes ever
reach `pickle.load` — and a clear error explains why, instead of an obscure downstream failure or
a silently wrong prediction.

## Using it

Write the sidecar once, right after saving the artifact:

```python
import joblib
from mlframe.utils.safe_pickle import write_sidecar

joblib.dump(model, "infer/my_featureset/lgb.dump")
write_sidecar("infer/my_featureset/lgb.dump")
```

`read_trained_models` (see the README's inference example) calls the verification step
automatically when loading a featureset's saved models, so day-to-day usage doesn't need to touch
`safe_load` directly — writing the sidecar at save time is the only step callers own.

## What it is — and isn't — protecting against

This is an **integrity check**, not an **authenticity** control. It catches:

- Truncated or partially-written files (crashed mid-copy, killed mid-`dump`)
- The wrong file accidentally placed at a path (stale artifact, copy-paste mistake)
- Bit-rot / silent disk corruption

It does **not** protect against a malicious actor who can already write to the same folder as the
model: they can simply rewrite the model file *and* regenerate a matching sidecar together, and
the check will pass. If you need to load pickles from a location an untrusted party can write to,
a sidecar is the wrong tool — you'd need a *keyed* integrity control instead (an HMAC signed with a
secret the attacker doesn't have, or a detached cryptographic signature). The sidecar mechanism
deliberately does not attempt that; it solves the much more common "did this file survive the trip
intact" problem, not the adversarial one.

## Where the real work lives

The mechanism itself — `verify_sidecar`, `write_sidecar`, `safe_load`, `safe_dump` — lives in
[`pyutilz.core.safe_pickle`](https://github.com/fingoldo/pyutilz/blob/master/src/pyutilz/core/safe_pickle.py)
so any project depending on pyutilz gets the same primitive. `mlframe.utils.safe_pickle` is a thin
re-export that keeps mlframe's historical `MLFRAME_ALLOW_UNVERIFIED_PICKLE` env-var name for the
(default-off) legacy opt-out that lets a load proceed without a sidecar, logging a loud warning
when it does.

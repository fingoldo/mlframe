## What and why

Briefly describe the user-visible change and the motivation behind it.

## Checklist

- [ ] Unit tests cover the new code paths (happy path + obvious edges).
- [ ] Bug fixes include a regression test that fails on pre-fix code and passes on post-fix code.
- [ ] New features include a quantitative "business value" test locking in the intended win, and a `@pytest.mark.fast` representative subset.
- [ ] `pytest -m "not slow and not gpu"` passes locally.
- [ ] `ruff check src/mlframe` and `black --check src/mlframe` are clean.
- [ ] [CHANGELOG.md](../CHANGELOG.md) updated if the change is user-facing.

## Notes for reviewers

Anything that needs special attention — tricky invariants, perf measurements, follow-ups deferred.

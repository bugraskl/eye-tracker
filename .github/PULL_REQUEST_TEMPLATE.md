## Summary

<!-- What does this change and why? Link the issue it resolves, e.g. "Fixes #123". -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance / CPU use
- [ ] Refactoring (no behaviour change)
- [ ] Documentation
- [ ] Build, packaging or CI

## How was it tested?

<!--
Which OS / desktop, how many monitors, which camera? For UI changes add screenshots
(never with a camera frame showing a face). Automated tests are welcome, manual steps too.
-->

## Checklist

- [ ] Tests added or updated, and `uv run pytest` passes
- [ ] `uv run ruff check .` and `uv run ruff format --check .` are clean
- [ ] `uv run mypy` is clean
- [ ] `uv run python scripts/check_privacy.py` passes: no network code and no frames written to disk
- [ ] `CHANGELOG.md` updated under `## [Unreleased]` (for user-visible changes)
- [ ] Documentation updated if behaviour, settings or commands changed
- [ ] Works (or degrades gracefully) on Windows, macOS and Linux, or the limitation is documented

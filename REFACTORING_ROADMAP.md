# Refactoring Roadmap

This roadmap lists cleanup work that should happen after the current behavior-preserving package split. It is intentionally separate from algorithm development.

## High Priority

### 1. Split `slam/pipeline.py` Into Real Implementations

- Effort: Large
- Move function bodies from `slam/pipeline.py` into the facade modules that already exist.
- Keep compatibility imports stable.
- Verify after each extraction with `py_compile`, stage6, and stage11.

### 2. Introduce a Real `Config` Object for Stage Runs

- Effort: Medium
- Convert parsed argparse namespaces into typed stage/config dataclasses.
- Preserve all CLI defaults and option names.
- Reduce direct `args.*` reads outside CLI/pipeline orchestration.

### 3. Separate Core SLAM From File Writing

- Effort: Medium
- Move JSONL, Markdown, and PNG writes behind diagnostics interfaces.
- Keep core scan matching and loop closure side-effect free.
- Preserve existing output filenames through adapter functions.

### 4. Add Regression Comparison Script

- Effort: Small
- Compare pose count, loop count, fit score, and key artifact hashes between two benchmark directories.
- Use it before and after each refactor.

## Medium Priority

### 5. Extract Data Models Into `slam/models.py`

- Effort: Medium
- Move dataclass definitions out of `slam/pipeline.py`.
- Avoid import cycles by keeping models dependency-free.
- Preserve existing field names and defaults.

### 6. Extract Geometry and Covariance Utilities

- Effort: Medium
- Move SE(2), angle wrapping, axis alignment, covariance transforms, and Hessian helpers into `slam/utils/` and `slam/backend/covariance.py`.
- Add small unit checks for angle wrapping and covariance rotation.

### 7. Extract Loop Closure Analysis

- Effort: Medium
- Move loop candidate recording, reports, visualization, and replay into `slam/diagnostics/` and `slam/backend/loop_closure.py`.
- Keep loop acceptance behavior unchanged.

### 8. Extract Visualization

- Effort: Medium
- Move all PIL drawing logic into `slam/visualization/` and `slam/diagnostics/plots.py`.
- Preserve exact output image sizes, colors, and filenames unless intentionally changed later.

### 9. Create Stage-Level Integration Tests

- Effort: Medium
- Add tests that run small scan prefixes and assert known metrics.
- Avoid checking large binary artifacts in git.

## Low Priority

### 10. Normalize Naming

- Effort: Small
- Gradually align names such as `Pose2D`, `Pose`, `LoopClosure`, and `LoopConstraint`.
- Do this only when surrounding modules are already extracted.

### 11. Reduce Facade Re-exports

- Effort: Small
- Once modules own their implementations, trim broad `__all__` exports to public APIs only.
- Keep top-level compatibility aliases until downstream scripts are updated.

### 12. Add Lightweight Developer Tooling

- Effort: Small
- Add optional `ruff`, `pyright`/`mypy`, or `pyflakes` config if the team wants static checks.
- Keep these tools advisory until the codebase is fully split.

### 13. Document Output Retention Policy

- Effort: Small
- Define which benchmark runs are retained and how to name temporary/debug runs.
- Consider adding a small cleanup script after the policy is stable.

## Refactor Guardrails

- No algorithmic changes during refactor-only work.
- Preserve CLI command names, flags, defaults, and output filenames.
- Run full stage11 after every large extraction.
- If pose count, accepted loop count, fit score, or key diagnostic hashes change unexpectedly, stop and investigate before continuing.

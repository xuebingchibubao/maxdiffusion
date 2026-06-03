# AGENTS.md

Guidance for coding agents working in this repository.

## Repository Overview

MaxDiffusion is a Python/JAX diffusion codebase for XLA devices, especially Cloud TPUs. Core package code lives under `src/maxdiffusion`; configs live in `src/maxdiffusion/configs`; smoke/unit tests live under `src/maxdiffusion/tests`; TPU end-to-end scripts live under `end_to_end/tpu`.

This checkout is currently on the `causal` branch. The Causal Forcing Stage 1 framewise AR diffusion checkpoint is available at `/home/bennett121358/ar_diffusion.pt`.

## Development Rules

- Prefer existing MaxDiffusion patterns over new abstractions.
- Keep edits scoped to the requested feature. Do not refactor unrelated model, pipeline, or checkpoint code.
- Use ASCII for new or edited files unless the file already requires Unicode.
- Use `apply_patch` for manual file edits.
- Do not revert user changes in a dirty worktree.
- Avoid destructive commands such as `git reset --hard`, `git checkout --`, or broad `rm` unless explicitly requested.
- `rg` may not be installed in this environment; use `find` and `grep` as fallback.

## Python And Style

- Project target is Python `>=3.12`.
- Formatting/lint configuration is in `pyproject.toml`.
- Ruff ignores `E501`; line length is configured as `119`.
- Existing source style uses two-space indentation in many MaxDiffusion modules. Match the local file you edit.
- Prefer structured APIs and typed helpers where practical.

## Useful Commands

- List files: `find src -type f | sort`
- Run a focused test: `python -m pytest src/maxdiffusion/tests/<test_file>.py`
- Run Wan smoke tests: `python -m pytest src/maxdiffusion/tests/generate_wan_smoke_test.py`
- Run package import checks with local source: `PYTHONPATH=src python -c "import maxdiffusion"`
- Run formatter/lint script if needed: `./unit_test_and_lint.sh`

Some commands may need network or TPU access. If sandboxed commands fail due to network or device access, request escalation instead of working around it.

## Causal Forcing Toy Task Notes

Goal: run framewise causal video generation on TPU using MaxDiffusion.

Reference implementation: `thu-ml/Causal-Forcing`.
Checkpoint: `/home/bennett121358/ar_diffusion.pt`, corresponding to `framewise/ar_diffusion.pt`.

Important implementation hints:

- Framewise Causal Forcing is a Wan-based autoregressive diffusion model.
- The framewise causal constraint is temporal: a query token from frame `t` can attend to text/context tokens and video tokens from frames `<= t`, but not later video frames.
- Use MaxDiffusion's existing Splash Attention package under `src/maxdiffusion/kernels/splash_attention` for block-sparse mask support where possible.
- Keep a dense JAX fallback for CPU/GPU tests and shape checks; TPU Splash paths should be opt-in/config-driven.
- Add focused tests for mask semantics independently of full model execution.
- Do not assume `/home/bennett121358/ar_diffusion.pt` has already been converted to MaxDiffusion/NNX parameter trees; validate checkpoint key structure before writing conversion logic.

## Verification Expectations

For small attention/mask changes, run focused unit tests and import checks. Full TPU video generation may require TPU hardware, Wan base weights, and large memory; if it cannot be run locally, document the exact command and what was verified.

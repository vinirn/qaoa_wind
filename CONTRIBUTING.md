# Contributing

Thank you for improving this QAOA wind‑turbine optimization project. Please read `AGENTS.md` for the full repository guidelines.

## Quick Start
1. Fork and clone the repo.
2. Create env and install deps:
   - `python3 -m venv qiskit_env && source qiskit_env/bin/activate`
   - `pip install -r requirements.txt`
3. Run locally:
   - `./run_qaoa.sh` (default) or `./run_qaoa.sh --list-configs`
   - `./run_qaoa.sh config_3x3.json`

## Development
- Main code in `qaoa_turbinas.py`; helpers/CLI and plotting in `utils.py`.
- Scenarios live in `config*.json` — keep clear names and descriptions.
- Removed editable install: run with `./run_qaoa.sh` or `python qaoa_turbinas.py` (uses hardcoded config).

## Testing
- No formal suite yet. Validate with small configs and review plots in `images/`.
- If adding tests, use `pytest` under `tests/` with files named `test_*.py`.

## Commits & Pull Requests
- Commit style: imperative, concise (e.g., `Upgrade to EstimatorV2 and refactor display_interference_matrix to utils module`).
- PRs should include: summary, config(s) used, key outputs/plot paths, and linked issues.
- Do not commit local envs (`qiskit_env/`) or large binaries; keep images small.

## Reference
- See `AGENTS.md` (Repository Guidelines) for structure, code style, and workflow details.

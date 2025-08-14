# Repository Guidelines

## Project Structure & Module Organization
- `qaoa_turbinas.py`: main entry point; loads JSON config, builds QAOA, runs optimization.
- `utils.py`: CLI parsing, config loading, visualization, plots saved to `images/`.
- `config*.json`: runnable scenarios (e.g., `config.json`, `config_3x3.json`, `config_vertical.json`).
- `run_qaoa.sh`: convenience runner that activates `qiskit_env` and forwards args.
- `requirements.txt` and `setup.py`: dependencies and optional editable install.
- `images/`: auto-generated plots; safe to commit small samples if helpful.

## Build, Test, and Development Commands
- Create env and install deps:
  - `python3 -m venv qiskit_env && source qiskit_env/bin/activate`
  - `pip install -r requirements.txt`
- Run locally (recommended):
  - `./run_qaoa.sh` (default config)
  - `./run_qaoa.sh config_3x3.json` or `python qaoa_turbinas.py -c config_3x3.json`
  - Discover configs: `python qaoa_turbinas.py --list-configs`
- Optional package install:
  - `pip install -e .` then `qaoa-turbinas -c config_vertical.json`

## Coding Style & Naming Conventions
- Python 3.10+; follow PEP 8 with 4‑space indentation.
- Use `snake_case` for functions/variables, `CapWords` for classes.
- Keep reusable helpers in `utils.py`; avoid duplicating logic in `qaoa_turbinas.py`.
- Config files follow `config*.json`; prefer descriptive `grid.description` fields.
- Console output is currently PT‑BR; keep tone and emoji usage consistent when extending.

## Testing Guidelines
- No formal test suite yet. Validate changes with small configs and check generated plots in `images/`.
- When adding tests, prefer `pytest` with files under `tests/` named `test_*.py`.
- Include quick checks for: config parsing, wake penalty calculation, and CLI flags (`--list-configs`).

## Commit & Pull Request Guidelines
- Commit messages: imperative and concise (seen in history):
  - Example: `Upgrade to EstimatorV2 and refactor display_interference_matrix to utils module`.
- Pull requests should include:
  - Clear description, config used (e.g., `-c config_3x3.json`), and before/after notes.
  - Key command output and/or plot paths from `images/`.
  - Linked issues, focused diffs, and no changes to local env (`qiskit_env/`).

## Security & Configuration Tips
- Do not commit secrets or large binaries; plots should be compressed PNGs.
- Keep runs reproducible by specifying configs; the code sets fixed seeds for random scores.

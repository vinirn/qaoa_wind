# Pull Request Template

## Summary
- What does this PR change and why?

## Type
- [ ] Feature
- [ ] Bug fix
- [ ] Refactor/cleanup
- [ ] Docs
- [ ] Other

## Related Issues
- Closes #
- Related to #

## How To Test
Include exact commands and configs used.

```bash
# create/activate env and install deps (first time)
python3 -m venv qiskit_env && source qiskit_env/bin/activate
pip install -r requirements.txt

# run with default config
./run_qaoa.sh

# run with explicit scenarios
./run_qaoa.sh config_3x3.json
./run_qaoa.sh config_vertical.json

# optional CLI directly
python qaoa_turbinas.py --list-configs
```

## Screenshots / Artifacts
- Plots saved under `images/` (e.g., attach `images/qaoa_latest_*.png`).

## Checklist
- [ ] Read `AGENTS.md` and `CONTRIBUTING.md`.
- [ ] Clear description and rationale provided.
- [ ] Commands and configs to reproduce included.
- [ ] No local envs or large binaries committed (e.g., `qiskit_env/`).
- [ ] Code follows style conventions (PEP 8, naming).
- [ ] Tests added/updated or manual validation steps documented.
- [ ] Docs and examples updated if behavior changes.

## Notes
- Additional context, migration notes, or follow-ups.


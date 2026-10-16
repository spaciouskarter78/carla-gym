# Contributing

Thank you for considering a contribution. Bug reports, questions and pull
requests are all welcome.

## Reporting a problem or asking a question

Open an issue on GitHub:
https://github.com/spaciouskarter78/carla-gym/issues

For a bug, please include: what you ran (the command or a short script), what
you expected, what happened (the full error or the numbers), and your Python
version and operating system. A seed that reproduces the problem helps most.

## Setting up

```bash
git clone https://github.com/spaciouskarter78/carla-gym.git
cd carla-gym
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    Linux/macOS: source .venv/bin/activate
pip install -e ".[rl,marl,bounds,dev]"
export PYTHONPATH=.                  # PowerShell: $env:PYTHONPATH = "."
python -m pytest -q
```

Tests that need the raw datasets in `data/raw/` (not distributed) are skipped
when the files are absent; everything else must pass.

## Pull requests

* **Add or update tests** for any change to the simulator, the reward, the
  baselines or the scripts. The simulator's invariants (units conserved,
  buffers and links within limits, determinism under a seed) are tested in
  `tests/`; keep them passing.
* **Keep defaults stable.** A change that alters dynamics or results must say
  so in the README's change log and bump the version; results from different
  versions are not comparable.
* **Numbers in the README come from files.** Tables between `RESULTS` markers
  are copied from `results/` by `scripts/update_readme_results.py`; regenerate
  them instead of editing them by hand, and say which commands produced new
  results.
* **Choose on validation seeds, report on test seeds.** Training uses seeds
  0 and up, learning curves 1000 and up, and final tables 2000 and up. Any
  choice (scenario, hyperparameter, baseline setting) is made on validation
  seeds before test results are read.
* **No raw or private data.** Only derived, shareable traces belong in
  `data/`; raw captures stay out of version control.

## Code of conduct

Participation in this project is governed by `CODE_OF_CONDUCT.md`.

# Contributing

Install with `python -m pip install -e '.[dev,gui,render]'` in a virtual environment.
Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`,
`ruff check src tests scripts launch`, and `python -m build` before sending changes.

Keep the core CPU-only. Make rendering, larger learning libraries and hardware
integration optional. Report the seed, MuJoCo version, mode, water parameters,
episode length and an idle baseline for any claimed learning improvement.
Distinguish planar swimming from unconstrained 3D swimming.

Preserve `LICENSE`, `NOTICE`, and asset provenance. Do not edit vendored meshes
without recording their origin and changes. New contributions use Apache-2.0.

For GUI changes, run `xvfb-run -a python scripts/check_dashboard.py`.
Use `step_water` for every physics advance; keep ROS and Qt imports optional.

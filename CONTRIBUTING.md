# Contributing to bobspec

Thank you for your interest! Bug reports, questions and contributions are welcome.

## Reporting a problem or asking a question

Open an issue at <https://github.com/DeimerAlvarez/bobspec/issues> and include:

* the command you ran and its full output (run `bob --debug ...` to include the traceback);
* the bobspec version (`bob --version`), Python version and operating system;
* if possible, a small file that reproduces the problem.

## Development setup

```bash
git clone https://github.com/DeimerAlvarez/bobspec.git
cd bobspec
pip install -e ".[dev]"
pytest                                   # all tests must pass
ruff check src tests && ruff format --check src tests
mypy src --ignore-missing-imports
```

## Code organisation

* `src/bobspec/io.py`, `spectrum.py`, `rv.py`, `continuum.py`, `combine.py`, ... contain
  pure numerical functions: no plotting, no printing, no user interaction. New science
  goes here, with tests.
* `interactive.py` and `plotting.py` contain all matplotlib code.
* `commands.py` contains the user commands. A command is a function decorated with
  `@command(name, arg(...), ...)`; the arguments are declared with the same syntax as
  `argparse.add_argument`. Registering it automatically provides `help <name>`, the
  one-shot CLI (`bob <name> ...`) and, unless `batchable=False`, the batch mode
  `<name> @list`.
* After adding or changing a command, regenerate the reference with
  `python docs/generate_commands.py`.

## Pull requests

1. Fork the repository and create a branch.
2. Add tests for new behaviour (synthetic spectra with known answers are preferred).
3. Make sure tests, `ruff` and `mypy` pass; describe the change in `CHANGELOG.md`.
4. Open the pull request with a short description of what changes and why.

By contributing you agree that your contribution is released under the MIT license.

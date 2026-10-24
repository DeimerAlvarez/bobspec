# Contributing to bobspec

Thank you for your interest! Questions, bug reports, feature requests and
contributions are welcome.

## Getting help

If you are not sure how to do something with `bob`, first look at
[GETTING_STARTED.md](GETTING_STARTED.md), at `help <command>` inside `bob`, and at
the [command reference](docs/commands.md). If the answer is not there, open an
issue at <https://github.com/DeimerAlvarez/bobspec/issues> and describe what you
are trying to do: questions are welcome as issues, and they help improve the
documentation.

## Reporting a problem

Open an issue at <https://github.com/DeimerAlvarez/bobspec/issues> and include:

* the command you ran and its full output (run `bob --debug ...` to include the traceback);
* the bobspec version (`bob --version`), Python version and operating system;
* if possible, a small file that reproduces the problem.

## Requesting a feature

Open an issue describing the scientific use case (which spectra, which step of the
analysis) and, if possible, the expected input and output. Discussing a new
command in an issue before writing it avoids duplicated work.

## Development setup

```bash
git clone https://github.com/DeimerAlvarez/bobspec.git
cd bobspec
python3 -m venv .venv && source .venv/bin/activate     # recommended
pip install -e ".[dev]"
pytest                                   # all tests must pass
ruff check src tests && ruff format --check src tests
mypy src --ignore-missing-imports
```

These are the same checks run by continuous integration on every push and pull
request (Linux and macOS, Python 3.11-3.13).

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
  `python docs/generate_commands.py`, and update `GETTING_STARTED.md` if the
  behaviour seen by users changes.

## Pull requests

1. Fork the repository and create a branch.
2. Add tests for new behaviour (synthetic spectra with known answers are preferred).
3. Make sure tests, `ruff` and `mypy` pass; describe the change in `CHANGELOG.md`.
4. Open the pull request with a short description of what changes and why.

By contributing you agree that your contribution is released under the MIT license.

## Releases

Versions follow [semantic versioning](https://semver.org): `MAJOR.MINOR.PATCH`.
Each release is a git tag (`vX.Y.Z`) with a GitHub release, and its changes are
listed in [CHANGELOG.md](CHANGELOG.md).

## Project governance and support

bobspec is maintained by Deimer Antonio Alvarez Garay (INAF - Osservatorio
Astrofisico di Arcetri), who reviews issues and pull requests and decides on
changes and releases, after discussion in the corresponding issue or pull
request. Support is provided on a best-effort basis through the GitHub issue
tracker; issues and pull requests usually receive a first answer within two
weeks. Contributors who make substantial, sustained contributions may be
invited to become maintainers.

Please be respectful and constructive in issues and pull requests.

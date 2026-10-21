"""Command registry shared by the REPL, the one-shot CLI and batch lists.

Every command is a function taking an :class:`argparse.Namespace` and
optionally returning result rows. Declaring the arguments once gives, for
free: type conversion and validation, ``help <command>``, and an IRAF-style
batch mode, ``<command> @FILE``, in which each line of the file is parsed
exactly like a command line (so optional arguments work in lists too).
"""

from __future__ import annotations

import argparse
import inspect
import logging
import re
import shlex
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

from bobspec.constants import DEFAULT_TELLURIC_REGIONS, Region

log = logging.getLogger(__name__)

Row = dict[str, Any]
Result = Row | list[Row] | None
Handler = Callable[[argparse.Namespace], Result]
ArgSpec = tuple[tuple[str, ...], dict[str, Any]]


class UsageError(Exception):
    """Invalid command-line arguments."""


class HelpRequested(Exception):
    """Raised instead of exiting when ``-h/--help`` is used inside the REPL."""


class CommandParser(argparse.ArgumentParser):
    """ArgumentParser that raises instead of calling :func:`sys.exit`.

    Tokens starting with ``-`` followed by a digit are always values, so
    negative velocities (``-600``) and southern declinations (``-12:18:00``)
    need no quoting.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("allow_abbrev", False)
        kwargs.setdefault("formatter_class", argparse.RawDescriptionHelpFormatter)
        super().__init__(*args, **kwargs)
        self._negative_number_matcher = re.compile(r"^-\d")

    def error(self, message: str) -> NoReturn:
        """Raise :class:`UsageError` instead of exiting."""
        raise UsageError(f"{self.prog}: {message}")

    def exit(self, status: int = 0, message: str | None = None) -> NoReturn:
        """Raise :class:`HelpRequested` instead of exiting (after ``--help``)."""
        if message:
            print(message)
        raise HelpRequested(status)


@dataclass(frozen=True, slots=True)
class Command:
    """A registered command."""

    name: str
    summary: str
    handler: Handler
    parser: CommandParser
    batchable: bool


@dataclass(slots=True)
class Session:
    """Mutable settings shared by all commands of a session."""

    telluric_regions: tuple[Region, ...] = DEFAULT_TELLURIC_REGIONS
    extra: dict[str, Any] = field(default_factory=dict)


COMMANDS: dict[str, Command] = {}
ALIASES: dict[str, str] = {}
SESSION = Session()


def arg(*flags: str, **kwargs: Any) -> ArgSpec:
    """Describe one argument with :meth:`argparse.ArgumentParser.add_argument` syntax."""
    return flags, kwargs


BATCH_EPILOG = """\
batch mode (IRAF style):
  {name} @FILE [options] [--summary TABLE]
  Runs '{name}' once for every non-empty line of FILE. Each line holds the
  arguments exactly as they would be typed after '{name}' (quotes and '#'
  comments allowed). Options written after @FILE apply to every line, and a
  line can override them. A failing line is reported and skipped; numerical
  results are collected in the text table TABLE (default: <FILE name>_{name}.txt)."""


def command(
    name: str, *args: ArgSpec, batchable: bool = True, aliases: Iterable[str] = ()
) -> Callable[[Handler], Handler]:
    """Register the decorated handler as command ``name``.

    The handler's docstring is the help text; its first line is the summary
    shown by ``help``. When ``batchable`` is True the command also accepts
    ``@FILE`` as first argument (see :func:`run_batch`).
    """

    def decorator(handler: Handler) -> Handler:
        doc = inspect.getdoc(handler) or ""
        epilog = BATCH_EPILOG.format(name=name) if batchable else None
        parser = CommandParser(prog=name, description=doc, epilog=epilog)
        for flags, kwargs in args:
            parser.add_argument(*flags, **kwargs)
        COMMANDS[name] = Command(name, doc.splitlines()[0] if doc else "", handler, parser, batchable)
        ALIASES.update({alias: name for alias in aliases})
        return handler

    return decorator


def resolve(name: str) -> Command | None:
    """Return the command registered as ``name`` (or one of its aliases)."""
    return COMMANDS.get(ALIASES.get(name, name))


def all_names() -> list[str]:
    """Every name accepted by :func:`resolve` (for help and TAB completion)."""
    return sorted(set(COMMANDS) | set(ALIASES))


def execute(cmd: Command, tokens: Sequence[str]) -> Result:
    """Run ``cmd``: once with ``tokens``, or in batch mode if the first token is ``@FILE``.

    Commands that are not batchable receive ``@FILE`` as an ordinary argument
    (``plot @spectra.txt`` plots all the files listed in ``spectra.txt``).
    """
    tokens = list(tokens)
    if cmd.batchable and tokens and tokens[0].startswith("@"):
        return run_batch(cmd, tokens)
    return cmd.handler(cmd.parser.parse_args(tokens))


# ----------------------------------------------------------------------------- batch mode
def _split_batch_tokens(cmd: Command, tokens: list[str]) -> tuple[Path, Path | None, list[str]]:
    """Separate ``@FILE``, the batch-only ``--summary`` option and the common options."""
    if not cmd.batchable:
        raise UsageError(f"{cmd.name}: batch mode (@FILE) is not available for this command")
    if tokens[0] == "@":
        raise UsageError(f"{cmd.name}: write the list file right after '@', e.g. {cmd.name} @list.txt")
    list_file, rest, summary = Path(tokens[0][1:]), tokens[1:], None
    for i, token in enumerate(rest):
        if token == "--summary":
            if i + 1 >= len(rest):
                raise UsageError(f"{cmd.name}: --summary needs a file name")
            summary = Path(rest[i + 1])
            del rest[i : i + 2]
            break
        if token.startswith("--summary="):
            summary = Path(token.split("=", 1)[1])
            del rest[i]
            break
    return list_file, summary, rest


def _read_list(list_file: Path) -> list[tuple[int, list[str]]]:
    """Tokenised non-empty lines of a list file; unparsable lines are reported and skipped."""
    lines = []
    for lineno, line in enumerate(list_file.read_text().splitlines(), start=1):
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError as exc:
            log.error("%s:%d: %s", list_file, lineno, exc)
            continue
        if tokens:
            lines.append((lineno, tokens))
    return lines


def _check_common_options(cmd: Command, common: list[str], first_line: list[str]) -> None:
    """Fail early when the options written after @FILE are invalid for the command."""
    try:
        cmd.parser.parse_args([*common, *first_line])
    except UsageError as exc:
        try:
            cmd.parser.parse_args(first_line)
        except UsageError:
            return  # the line itself is wrong: it is reported when it runs
        raise UsageError(f"{cmd.name}: options after @FILE are not valid: {exc}") from exc


def run_batch(cmd: Command, tokens: list[str]) -> list[Row]:
    """Execute ``cmd`` once per line of ``@FILE`` (IRAF-style batch mode).

    ``tokens`` is ``["@FILE", *options]``. The options are placed before each
    line's own arguments, so a line can override them. Failing lines are logged
    and skipped; result rows are written to a tab-separated table.

    Returns:
        The result rows, each with the line number of the list file.

    Raises:
        UsageError: For a missing file name, a non-batchable command or invalid options.
        FileNotFoundError: If the list file does not exist.
    """
    list_file, summary, common = _split_batch_tokens(cmd, tokens)
    if not list_file.is_file():
        raise FileNotFoundError(f"list file not found: {list_file}")
    lines = _read_list(list_file)
    if common and lines:
        _check_common_options(cmd, common, lines[0][1])
    rows: list[Row] = []
    n_ok = n_failed = 0
    for lineno, line_tokens in lines:
        if line_tokens[0].startswith("@"):
            log.error("%s:%d: nested @ lists are not allowed", list_file, lineno)
            n_failed += 1
            continue
        args = [*common, *line_tokens]
        log.info("[%s:%d] %s %s", list_file.name, lineno, cmd.name, shlex.join(args))
        try:
            result = cmd.handler(cmd.parser.parse_args(args))
        except HelpRequested:
            continue
        except Exception as exc:  # batch boundary: one bad line must not stop the list
            log.error("%s:%d: %s", list_file, lineno, exc)
            log.debug("Traceback:", exc_info=True)
            n_failed += 1
            continue
        n_ok += 1
        rows.extend({"line": lineno, **row} for row in _as_rows(result))
    log.info("%s: %d line(s) processed, %d failed.", cmd.name, n_ok, n_failed)
    if rows:
        table = summary or default_table(list_file, cmd.name)
        write_table(table, rows)
        log.info("Results table: %s", table)
    return rows


def _as_rows(result: Result) -> list[Row]:
    if result is None:
        return []
    return [result] if isinstance(result, Mapping) else list(result)


def default_table(source: Path, label: str) -> Path:
    """``<dir>/<source name without suffix>_<label>.txt``, e.g. ``rv_list.txt`` -> ``rv_list_rv.txt``."""
    return source.with_name(f"{source.stem}_{label}.txt")


def _fmt(value: Any) -> str:
    """One table cell: no spaces inside a cell, missing values as ``nan``."""
    if value is None or value == "":
        return "nan"
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return f'"{text}"' if any(c.isspace() for c in text) else text


def write_table(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write ``rows`` as a plain-text table with aligned, space-separated columns.

    The first line holds the column names after ``#``. Columns are the union
    of the keys of all rows (missing values are written as ``nan``); spaces in
    column names become ``_`` and values containing spaces are quoted. Read it
    with ``astropy.table.Table.read(path, format="ascii.commented_header")``,
    with ``numpy.genfromtxt(path, names=True, dtype=None, encoding=None)``
    (if no value contains spaces) or with any spreadsheet.
    """
    columns = list(dict.fromkeys(key for row in rows for key in row))
    header = [str(col).replace(" ", "_") for col in columns]
    cells = [[_fmt(row.get(col)) for col in columns] for row in rows]
    widths = [max(len(name), *(len(line[i]) for line in cells)) for i, name in enumerate(header)]

    def join(items: Sequence[str]) -> str:
        return "  ".join(item.ljust(width) for item, width in zip(items, widths, strict=True)).rstrip()

    text = ["# " + join(header), *("  " + join(line) for line in cells)]
    Path(path).write_text("\n".join(text) + "\n")
    return Path(path)

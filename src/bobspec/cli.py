"""Command-line entry point.

``bob``                 start the interactive shell
``bob <command> ...``   run one command and exit (exit status 1 on error)
``bob --debug ...``     show tracebacks
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import difflib
import logging
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from bobspec import __version__, commands  # noqa: F401  (importing commands registers them)
from bobspec.constants import DEFAULT_TELLURIC_REGIONS
from bobspec.io import SpectrumReadError
from bobspec.registry import COMMANDS, SESSION, HelpRequested, UsageError, all_names, execute, resolve

log = logging.getLogger("bobspec")

HISTORY_FILE = Path.home() / ".bob_history"
EXIT_WORDS = frozenset({"exit", "quit", "q", "e"})
BUILTINS = ("help", "exit", "quit", "cd", "pwd", "ls", "tellurics")
EXPECTED_ERRORS = (SpectrumReadError, ValueError, OSError, KeyError, RuntimeError)

BANNER = r"""
###############################
     #####    ###   #####
     #    #  #   #  #    #
     #####   #   #  #####
     #    #  #   #  #    #
     #####    ###   #####
###############################
"""


# ----------------------------------------------------------------------------- builtins
def show_help(name: str | None = None) -> None:
    """Print the command list, or the full help of one command."""
    if name:
        cmd = resolve(name)
        if cmd is None:
            print(f"Unknown command: {name}")
        else:
            cmd.parser.print_help()
        return
    print("\nCommands (help <command> for details; [@] = also accepts <command> @list_file):")
    for cmd in sorted(COMMANDS.values(), key=lambda c: c.name):
        print(f"  {cmd.name + (' [@]' if cmd.batchable else ''):22s} {cmd.summary}")
    print("\nShell: cd, pwd, ls, tellurics, exit; other executables in PATH are run directly.\n")


def set_tellurics(args: Sequence[str]) -> None:
    """``tellurics`` show | none | default | <file with two columns: start end>."""
    if args:
        match args[0]:
            case "none":
                SESSION.telluric_regions = ()
            case "default":
                SESSION.telluric_regions = DEFAULT_TELLURIC_REGIONS
            case path:
                import numpy as np

                data = np.loadtxt(path, comments="#", ndmin=2, usecols=(0, 1))
                SESSION.telluric_regions = tuple((float(a), float(b)) for a, b in data)
    regions = ", ".join(f"{a:g}-{b:g}" for a, b in SESSION.telluric_regions) or "none"
    print(f"Excluded (telluric) regions [A]: {regions}")


def _run_builtin(name: str, args: list[str]) -> bool:
    match name:
        case "help" | "h" | "?":
            show_help(args[0] if args else None)
        case "cd":
            os.chdir(Path(args[0] if args else "~").expanduser())
            print(Path.cwd())
        case "pwd":
            print(Path.cwd())
        case "ls":
            for entry in sorted(Path(args[0] if args else ".").expanduser().iterdir()):
                print(entry.name + ("/" if entry.is_dir() else ""))
        case "tellurics":
            set_tellurics(args)
        case _:
            return False
    return True


# ----------------------------------------------------------------------------- dispatch
def dispatch(line: str) -> bool:
    """Run one REPL line. Returns False when the user asks to exit.

    Never raises: every error is reported and the shell keeps running.
    """
    try:
        tokens = shlex.split(line)
    except ValueError as exc:
        print(f"Syntax error: {exc}")
        return True
    if not tokens:
        return True
    name, *args = tokens
    if name.lower() in EXIT_WORDS:
        return False
    try:
        if _run_builtin(name, args):
            return True
        if (cmd := resolve(name)) is not None:
            execute(cmd, args)
        elif executable := shutil.which(name):
            subprocess.run([executable, *args], check=False)
        else:
            hint = difflib.get_close_matches(name, all_names(), n=3, cutoff=0.5)
            print(f"Unknown command: {name}" + (f" (did you mean: {', '.join(hint)}?)" if hint else ""))
    except HelpRequested:
        pass
    except UsageError as exc:
        print(exc)
        if cmd := resolve(name):
            print(cmd.parser.format_usage(), end="")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except EXPECTED_ERRORS as exc:
        log.error("%s: %s", name, exc)
        log.debug("Traceback:", exc_info=True)
    except Exception:  # REPL boundary: report and keep the session alive
        log.exception("%s: unexpected error (please report it with the traceback)", name)
    return True


# ----------------------------------------------------------------------------- readline
def _completer(text: str, state: int) -> str | None:
    import readline

    before = readline.get_line_buffer()[: readline.get_begidx()]
    if not before.strip():
        options = [*all_names(), *BUILTINS]
        matches = sorted(o for o in options if o.startswith(text))
    else:
        base = Path(text).expanduser()
        folder, prefix = (base, "") if text.endswith(os.sep) else (base.parent, base.name)
        prefix_text = text[: len(text) - len(prefix)]
        try:
            entries = [e for e in folder.iterdir() if e.name.startswith(prefix)]
        except OSError:
            entries = []
        matches = sorted(prefix_text + e.name + (os.sep if e.is_dir() else "") for e in entries)
    return matches[state] if state < len(matches) else None


def setup_readline() -> None:
    """Enable history and TAB completion when readline is available."""
    try:
        import readline
    except ImportError:  # e.g. Windows without pyreadline3
        return
    libedit = "libedit" in (readline.__doc__ or "")
    backend = getattr(readline, "backend", None) or ("editline" if libedit else "readline")
    readline.parse_and_bind("bind ^I rl_complete" if backend == "editline" else "tab: complete")
    readline.set_completer_delims(" \t\n;")
    readline.set_completer(_completer)
    with contextlib.suppress(OSError):
        readline.read_history_file(HISTORY_FILE)
    atexit.register(readline.write_history_file, HISTORY_FILE)


def repl() -> None:
    """Interactive shell."""
    setup_readline()
    print(BANNER)
    print(f"bobspec {__version__} - type 'help' for the commands, 'exit' to quit.")
    while True:
        try:
            line = input("bob> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not dispatch(line):
            break
    print("Bye!")


# ----------------------------------------------------------------------------- main
def configure_logging(debug: bool) -> None:
    """Plain messages for users, levels and tracebacks with --debug."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s" if debug else "%(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.DEBUG if debug else logging.INFO)
    log.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point."""
    parser = argparse.ArgumentParser(
        prog="bob", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--debug", action="store_true", help="verbose logging with tracebacks")
    parser.add_argument("--version", action="version", version=f"bobspec {__version__}")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command and its arguments")
    ns = parser.parse_args(argv)
    configure_logging(ns.debug)
    if not ns.command:
        repl()
        return 0
    name, *args = ns.command
    if name in ("help", "h", "?"):
        show_help(args[0] if args else None)
        return 0
    cmd = resolve(name)
    if cmd is None:
        print(f"Unknown command: {name}", file=sys.stderr)
        return 2
    try:
        execute(cmd, args)
    except HelpRequested:
        return 0
    except UsageError as exc:
        print(f"{exc}\n{cmd.parser.format_usage()}", file=sys.stderr, end="")
        return 2
    except EXPECTED_ERRORS as exc:
        log.error("%s: %s", name, exc)
        log.debug("Traceback:", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

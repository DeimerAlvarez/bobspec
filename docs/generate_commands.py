"""Regenerate docs/commands.md from the command parsers: python docs/generate_commands.py."""

from pathlib import Path

from bobspec import cli  # noqa: F401  (registers the commands)
from bobspec.registry import ALIASES, COMMANDS

parts = [
    "# Command reference\n",
    "Generated from the argument parsers (`python docs/generate_commands.py`); "
    "inside `bob` the same text is shown by `help <command>`.\n",
    "Every command marked *batch* also runs as `<command> @FILE [options] [--summary TABLE]` (IRAF style): "
    "each line of FILE holds the arguments of one call, exactly as typed on the command line; "
    "options after @FILE apply to every line.\n",
]
aliases = {target: [a for a, t in ALIASES.items() if t == target] for target in COMMANDS}
for name, cmd in sorted(COMMANDS.items()):
    extra = ["batch"] if cmd.batchable else []
    extra += [f"alias: {a}" for a in aliases[name]]
    parts.append(f"\n## `{name}`" + (f" ({', '.join(extra)})" if extra else "") + "\n")
    parts.append("```text\n" + cmd.parser.format_help().rstrip() + "\n```\n")
Path(__file__).with_name("commands.md").write_text("\n".join(parts))

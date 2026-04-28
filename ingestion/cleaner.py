import re

# ISO 8601 timestamps emitted by GitHub Actions on every line
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z\s*")

# ANSI escape codes (color, bold, cursor movement)
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# GitHub Actions workflow commands — UI hints, not log content
_GHA_CMD = re.compile(r"##\[(?:group|endgroup|warning|error|debug|notice|add-mask|set-output)\][^\n]*")

# UTF-8 BOM that appears at the start of some zip-extracted files
_BOM = re.compile(r"^\ufeff")

# Collapse runs of 3+ blank lines to 2 (keeps visual separation without wasting tokens)
_EXCESS_BLANK = re.compile(r"\n{3,}")


def clean_log(text: str) -> str:
    """Remove timestamps, ANSI codes, GHA commands, and excessive whitespace."""
    text = _BOM.sub("", text)
    text = _ANSI.sub("", text)
    text = _TIMESTAMP.sub("", text)
    text = _GHA_CMD.sub("", text)
    text = _EXCESS_BLANK.sub("\n\n", text)
    return text.strip()

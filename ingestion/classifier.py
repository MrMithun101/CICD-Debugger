import re

# Ordered from most specific to most general.
# First match wins — order matters for patterns that could overlap.
_PATTERNS: list[tuple[str, list[str]]] = [
    ("oom", [
        r"out of memory",
        r"MemoryError",
        r"Cannot allocate memory",
        r"Killed",                    # OOM kill from the OS
    ]),
    ("timeout", [
        r"timed.?out",
        r"timeout exceeded",
        r"SIGTERM",
        r"The runner has received a shutdown signal",
        r"Cancelling since a higher priority waiting request",
    ]),
    ("dependency_error", [
        r"No matching distribution found",
        r"Could not find a version that satisfies",
        r"npm ERR!",
        r"yarn error",
        r"ERROR: Failed building wheel",
        r"ModuleNotFoundError",
        r"ImportError",
        r"Package .+ not found",
        r"fatal: repository .+ not found",
    ]),
    ("build_error", [
        r"error TS\d+",               # TypeScript compiler
        r"SyntaxError",
        r"error\[E\d+\]",             # Rust
        r"cannot find symbol",        # Java
        r"compilation failed",
        r"make\[.*\]: \*\*\* .* Error",
        r"CMake Error",
    ]),
    ("lint_error", [
        r"flake8",
        r"pylint.*error",
        r"mypy.*error",
        r"eslint",
        r"ruff.*Found \d+ error",
        r"black.*would reformat",
    ]),
    ("test_failure", [
        r"FAILED\s+\S+",              # pytest: FAILED tests/test_foo.py
        r"\d+ failed",                # pytest summary
        r"AssertionError",
        r"FAIL:\s+test_",             # unittest
        r"●\s+",                      # Jest
        r"Tests run:.*Failures: [^0]",  # JUnit
        r"Process completed with exit code [^0]",
    ]),
]


def classify_failure(text: str) -> str:
    """Return the most specific failure category that matches the log text."""
    for failure_type, patterns in _PATTERNS:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return failure_type
    return "unknown"

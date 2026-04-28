import re

# Matches the section headers we wrote during log extraction: "=== path/step.txt ==="
_SECTION_HEADER = re.compile(r"^=== (.+?) ===$", re.MULTILINE)

# If a single step section exceeds this many characters, split it further.
# ~6000 chars ≈ ~1500 tokens — safe headroom below most embedding model limits.
MAX_CHUNK_CHARS = 6000


def chunk_log(text: str) -> list[dict[str, str]]:
    """Split a cleaned log into per-step sections.

    Returns a list of dicts: {"step": <step name>, "text": <content>}.
    Sections with no meaningful content are dropped.
    Sections exceeding MAX_CHUNK_CHARS are split on double-newlines.
    """
    headers = _SECTION_HEADER.findall(text)
    bodies = _SECTION_HEADER.split(text)
    # split() with a capturing group gives: [pre, h1, b1, h2, b2, ...]
    # bodies[0] is before the first header (usually empty), bodies[1] is h1, etc.

    chunks: list[dict[str, str]] = []
    for i, header in enumerate(headers):
        body = bodies[i * 2 + 2].strip()  # offset +2 because bodies[0] is pre-content
        if not body:
            continue
        for sub in _split_if_large(body):
            chunks.append({"step": header, "text": sub})

    return chunks


def _split_if_large(text: str) -> list[str]:
    """Split oversized text on paragraph boundaries, keeping chunks ≤ MAX_CHUNK_CHARS."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    parts = text.split("\n\n")
    result: list[str] = []
    current: list[str] = []
    current_len = 0

    for part in parts:
        # Account for the "\n\n" separator that joins this part to existing current content.
        sep = 2 if current else 0
        if current_len + sep + len(part) > MAX_CHUNK_CHARS and current:
            result.append("\n\n".join(current))
            current = []
            current_len = 0
            sep = 0
        current.append(part)
        current_len += sep + len(part)

    if current:
        result.append("\n\n".join(current))

    return result

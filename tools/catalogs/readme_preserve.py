"""Manual-block preservation for generated READMEs.

`update_readme.py` rebuilds every model/ensemble README from a scaffold,
which destroys hand-written content (risk register C-78; the 2026-06-04
regeneration wiped the synthetic_chant evaluation-semantics docs, C-77).

Any content wrapped in marker comments survives regeneration verbatim:

    <!-- manual -->
    ## My hand-written section
    ...
    <!-- /manual -->

Blocks are extracted from the pre-regeneration README and re-appended,
in order, at the end of the regenerated content.

Pure stdlib on purpose: unlike update_readme.py (which executes the full
regeneration at import time and needs views_pipeline_core), this module
is safely importable from tests.
"""
import re

MANUAL_START = "<!-- manual -->"
MANUAL_END = "<!-- /manual -->"

_BLOCK_RE = re.compile(
    re.escape(MANUAL_START) + r".*?" + re.escape(MANUAL_END), re.DOTALL
)


def extract_manual_blocks(text):
    """Return all manual blocks in `text`, markers included, in order.

    Raises ValueError on a dangling start marker (no closing marker):
    silently dropping it would wipe the very content these markers
    protect — a crashed regeneration is recoverable, deleted docs are not.
    """
    blocks = _BLOCK_RE.findall(text)
    if text.count(MANUAL_START) > len(blocks):
        raise ValueError(
            f"unterminated {MANUAL_START} block (missing {MANUAL_END}) — "
            "fix the README before regenerating, or its manual content would be lost"
        )
    return blocks


def merge_manual_blocks(generated, blocks):
    """Append `blocks` verbatim to `generated`, separated by blank lines.

    No-op when `blocks` is empty. The end of the file is the stable
    fixed point: blocks extracted from the merged output land in the
    same place on the next regeneration.
    """
    if not blocks:
        return generated
    parts = [generated.rstrip("\n")]
    parts.extend(blocks)
    return "\n\n".join(parts) + "\n"

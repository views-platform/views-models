"""Catalog generation: the README tables and per-model READMEs, derived from configs.

These scripts are the repo's documentation generator — they read every model's config
and rewrite tables in place. They run in CI (`.github/workflows/update_catalogs.yml`),
which is why their failure modes matter more than their line count (C-78, C-81, C-83).
"""

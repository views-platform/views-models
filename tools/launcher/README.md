# `launcher/`

The **delivery protocol** every postprocessor runs, in one place.

`postprocessor.sh` is sourced — never executed, like `tools/credentials/platform_env.sh`
— and defines `postprocessor_launch`. It owns the ordered sequence a partner delivery
needs: registry check before conda, conda lifecycle, the #294 capability assertion, the
environment load, then `main.py`.

A launcher under `postprocessors/*/run.sh` supplies only what genuinely varies by partner
— the conda environment name and the views-postprocessing pin — and calls the function.

**Why it is not copied into each launcher.** `run.sh` changes for two unrelated reasons:
because a partner is different, and because the protocol is different. Copying the second
means a protocol fix must be hand-applied once per partner, and the first one missed fails
silently. views-postprocessing recorded exactly that scar cloning `unfao/` into `crafd/`
(their #211: *"every partner-scoped guard was scoped to ONE partner"*).

Governed by **ADR-022**.

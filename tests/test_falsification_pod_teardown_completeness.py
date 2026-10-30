"""Failing test stubs from the 2026-10-26 falsification audit of "I can terminate the pod".

Audit subject: the #607 structure-penalty arm run on pod 216.81.151.60:11174, and the
claim that destroying it would lose nothing. Seven probes, categories A/D/E/F/G/H.
Two hard falsifications and one failure of the verification method itself.

These are stubs. They encode the rules the audit found missing; none of them passes today.
Each names the concrete 2026-10-26 incident it exists to prevent recurring.

Related: epic #608 (Cluster G), register C-168, C-169, C-170, C-161, C-171, C-317.
"""

import pytest


# ---------------------------------------------------------------------------
# HARD-1 (probe P3) -- a retrieval scoped to one run folder silently drops the others
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="no such guard exists; written 2026-10-26", strict=False)
def test_plot_retrieval_covers_every_diagnostics_run_folder():
    """A diagnostics sync must enumerate run folders, never hardcode one.

    2026-10-26: the plot feed was armed at 03:56 against
    `reports/plots/diagnostics/261026_0143/`, the only folder that existed then. The
    evaluate stage created `261026_0410/` at 04:10 and wrote 17 files into it,
    including all 13 per-origin `03_model_reasoning` biopsies -- the forecast-reasoning
    plots. The sync never looked, and the laptop/pod counts were reported as matching
    because BOTH numbers were taken inside the hardcoded folder.

    The control run had the same two-folder shape (`251026_1043` + `251026_1321`) and
    that fact had already been printed in this session before the feed was written.

    The rule: enumerate `diagnostics/*/`, assert the local set of folder names equals
    the remote set, and fail if either is empty.
    """
    raise NotImplementedError


@pytest.mark.xfail(reason="no such guard exists; written 2026-10-26", strict=False)
def test_file_count_parity_is_not_accepted_as_content_parity():
    """Equal counts between source and destination must not be reported as verified.

    A count comparison cannot detect a truncated transfer, and rsync's default
    mtime+size quick-check can skip a file rewritten within the same second at the
    same length. Only a per-file digest comparison may be rendered as a pass.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# HARD-2 (probe P7) -- a mandatory documented teardown step was skipped
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="no such guard exists; written 2026-10-26", strict=False)
def test_gate_side_teardown_requires_slice_gate_posterior_to_have_run():
    """`docs/runpod_run_guide.md` Phase 5 states this as mandatory:

        "If your readout is gate-side, you must slice the gate before you destroy
         the pod."
        "Run this on the pod, after the runner reports OK and BEFORE you rsync:
         python3 .../tools/podrun/slice_gate_posterior.py <model>"

    2026-10-26: it was not run. A raw `by_*` tarball was pulled instead -- a superset
    of the data, but the slice tool is also the VALIDATOR. It refuses on: no `by_*`
    directories, a target missing from an origin, a row count that is not 64,818, a
    draw count that disagrees between targets, a non-finite probability, or an archive
    whose entry count does not match what was written. None of those checks ran.

    Holding a superset of the bytes is not a substitute for the validation, because
    the validation is what would have revealed the bytes were wrong.

    The rule: teardown may not be declared safe for a gate-side readout unless
    `gate_h1.sha256` exists locally and `sha256sum -c` passes against it.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# HARD-3 (probe P2, first attempt) -- the verification itself returned empty and read as clean
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="no such guard exists; written 2026-10-26", strict=False)
def test_a_digest_comparison_asserts_its_digests_are_non_empty():
    """A comparison whose inputs are empty must fail, not report agreement.

    2026-10-26: probe P2 hashed the 78 members inside the gate archive with
    `tar --to-command`. The nested quoting broke, `sha256sum` never received the
    data, and every local digest came out as the empty string. `comm` then reported
    no differences in all three buckets -- only-on-pod, only-local, and differing --
    because the FILENAMES matched perfectly. The result was shaped exactly like a pass.

    This is C-168 ("nothing matched" rendering as "I could not look") occurring inside
    a probe written to detect C-168, in an audit of a claim that rests on verification.

    The rule: assert every digest is 64 hex characters and the count is non-zero
    BEFORE comparing. A positive control on the instrument, not on the subject.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Observations promoted to rules (found before this audit, same teardown)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="no such guard exists; written 2026-10-26", strict=False)
def test_mount_sweep_does_not_exclude_the_container_root_filesystem():
    """`pod_sweep_outputs.sh` lists `overlay` among the pseudo-filesystems it skips.

    On a RunPod container `/` IS an overlay mount, so the sweep -- whose stated thesis
    is "sweeps EACH MOUNT SEPARATELY, so -xdev is correct instead of silently
    excluding" -- never examined the root filesystem at all. A manual sweep found
    38,510 files from the run there (all cache and apt noise this time, 6.27 GB).

    C-169 was `find -xdev` silently excluding /workspace. This is the same defect
    reached by a second mechanism, inside the tool written to prevent the first.
    """
    raise NotImplementedError


@pytest.mark.xfail(reason="no such guard exists; written 2026-10-26", strict=False)
def test_sweep_distinguishes_an_unreadable_mount_from_a_single_file_bind_mount():
    """The sweep exited 1 with 56 UNVERIFIED mounts, every one of them a single-file
    bind mount (`libcuda.so.595.91.07`, `/etc/hosts`, `nvidia-smi`).

    A regular file cannot be a directory and cannot contain run output, so "I could
    not plant a sentinel here" is not "I could not look here" for this class. Refusing
    is the safe direction and the script is right to refuse rather than guess -- but a
    verdict that is unverified for 56 benign reasons trains the reader to wave exit 1
    through, which is the failure mode that makes the whole guard decorative.

    The rule: classify a mount point that is not a directory as EXCLUDED-WITH-REASON,
    distinct from UNVERIFIED, and keep UNVERIFIED for mounts that are genuinely opaque.
    """
    raise NotImplementedError

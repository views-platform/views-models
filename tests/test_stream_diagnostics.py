"""Guards for the live plot feed, each written against a failure this programme already had.

The feed this replaces streamed correctly and still lost 17 files, because it was armed
against one hardcoded folder and verified itself by counting inside that folder. These tests
exist so that specific shape of green-while-wrong cannot recur.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.podrun.stream_diagnostics import (
    REMOTE_PLOTS,
    Pod,
    Refuse,
    compare,
    local_manifest,
    sweep,
)


class StubPod(Pod):
    """A pod whose folder list and file manifest are fixtures, and whose rsync is a copy."""

    def __init__(self, folders, files, src: Path):
        super().__init__("root@stub", 22, "/dev/null")
        object.__setattr__(self, "_folders", folders)
        object.__setattr__(self, "_files", files)
        object.__setattr__(self, "_src", src)

    def ssh(self, command, timeout=120):
        class R:
            returncode = 0
            stderr = ""
        if "-maxdepth 1 -type d" in command:
            R.stdout = "\n".join(self._folders) + ("\n" if self._folders else "")
            R.returncode = 0 if self._folders else 1
        else:
            R.stdout = "".join(f"{size} {p}\n" for p, size in sorted(self._files.items()))
            R.returncode = 0 if self._files else 1
        return R

    def rsync_from(self, remote, dest, timeout=900):
        class R:
            returncode = 0
            stderr = ""
        dest.mkdir(parents=True, exist_ok=True)
        # `_writes`, when set, is what rsync ACTUALLY lands — which need not be what the pod
        # holds. That is the whole point: rsync exiting 0 is not evidence of a complete copy.
        landed = getattr(self, "_writes", None) or self._files
        for rel, size in landed.items():
            if rel in getattr(self, "_skip", ()):
                continue
            p = dest / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x" * size)
        return R


@pytest.mark.red
def test_a_folder_the_feed_never_visited_is_caught_even_at_equal_file_counts():
    """The 2026-10-26 defect, exactly: same count on both sides, different folders.

    That run reported local and remote as matching because both numbers were taken inside the
    one folder the sync knew about. A count check passes this; a content check must not.
    """
    remote = {"261026_0143/a.png": 10, "261026_0143/b.png": 10}
    local = {"261026_0410/a.png": 10, "261026_0410/b.png": 10}
    assert len(remote) == len(local), "fixture must have equal counts or it tests nothing"
    missing, differing = compare(remote, local)
    assert missing, (
        "a file set from an entirely different folder was accepted as a complete copy. "
        "This is the 2026-10-26 failure: 17 files lost, including all 13 per-origin "
        "03_model_reasoning biopsies, under a green report."
    )


@pytest.mark.red
def test_a_truncated_file_is_caught_although_its_path_is_present():
    """Present-but-wrong is not present. Size disagreement must fail, not pass."""
    missing, differing = compare({"f/a.png": 5000}, {"f/a.png": 12})
    assert not missing and differing == ["f/a.png"], (
        "a 12-byte local file stood in for a 5000-byte remote one; a path-only check would "
        "have called that a complete copy."
    )


@pytest.mark.green
def test_every_remote_folder_is_pulled_not_only_the_first(tmp_path):
    """A HydraNet run writes one folder per stage, so the second is where evaluate's plots go."""
    files = {"261026_0143/train.png": 7, "261026_0410/eval.png": 9, "261026_0410/origin_3.png": 11}
    pod = StubPod(["261026_0143", "261026_0410"], files, tmp_path / "src")
    dest = tmp_path / "dest"
    folders, n = sweep(pod, "/r/plots/diagnostics", dest, final=False)
    assert folders == 2 and n == 3
    assert set(local_manifest(dest)) == set(files), "a folder was enumerated but not pulled"


@pytest.mark.red
def test_a_file_left_behind_by_rsync_refuses_although_rsync_reported_success(tmp_path):
    """rsync exiting 0 is not evidence the copy is complete; the manifest comparison is."""
    files = {"a/1.png": 4, "a/2.png": 4, "b/3.png": 4}
    pod = StubPod(["a", "b"], files, tmp_path / "src")
    object.__setattr__(pod, "_skip", {"b/3.png"})
    with pytest.raises(Refuse) as e:
        sweep(pod, "/r/plots/diagnostics", tmp_path / "dest", final=False)
    assert "b/3.png" in str(e.value)


@pytest.mark.red
def test_the_final_sweep_refuses_when_there_is_nothing_to_copy(tmp_path):
    """Absent diagnostics is three different situations, and 'nothing to do' is not a verdict.

    Off, not-yet-run, and wrong-path are indistinguishable from the absence itself (ADR-025).
    During streaming that is tolerable; at teardown it is the moment it becomes expensive.
    """
    pod = StubPod([], {}, tmp_path / "src")
    with pytest.raises(Refuse) as e:
        sweep(pod, "/r/plots/diagnostics", tmp_path / "dest", final=True)
    assert "diagnostic_visualizations" in str(e.value), (
        "the refusal should name the config key that silently turns this off, since that is "
        "the most likely cause and it lives in config_meta.py where a reader will not look."
    )


@pytest.mark.green
def test_absence_is_tolerated_while_streaming_but_reports_itself(tmp_path, capsys):
    """A live feed starting before the stage exists must not abort the run it is watching."""
    pod = StubPod([], {}, tmp_path / "src")
    folders, n = sweep(pod, "/r/plots/diagnostics", tmp_path / "dest", final=False)
    assert (folders, n) == (0, 0)
    assert "not present yet" in capsys.readouterr().out


@pytest.mark.beige
def test_the_remote_path_matches_where_pod_run_model_actually_leaves_the_plots():
    """`pod_run_model.sh` cds into the model dir, so the plot path is relative to it."""
    runner = Path(__file__).resolve().parents[1] / "tools" / "podrun" / "pod_run_model.sh"
    body = runner.read_text(encoding="utf-8")
    assert 'cd "$REPO/models/$MODEL"' in body, (
        "pod_run_model.sh no longer cds into the model directory, so REMOTE_PLOTS "
        f"({REMOTE_PLOTS!r}) resolves somewhere else and the feed would pull an empty path."
    )


@pytest.mark.red
def test_sweep_refuses_equal_counts_from_a_different_folder(tmp_path):
    """⛔ The 2026-10-26 defect at the level where the DECISION is made, not just in compare().

    Written after a mutation audit: replacing sweep()'s content check with `len(remote) !=
    len(local)` left every other test in this file green. Each existing test either compared
    manifests directly, or removed a file so the counts diverged anyway — so none of them
    constrained sweep() to look at content. That is a guard whose subject was never the thing
    it protects.

    Here the pod holds two files under `261026_0410/` and rsync lands two files under
    `261026_0143/`. Counts agree; not one byte of the evaluation folder arrived. A count
    check passes this and loses the per-origin forecast-reasoning biopsies, exactly as it did.
    """
    on_pod = {"261026_0410/eval.png": 8, "261026_0410/origin_3.png": 8}
    landed = {"261026_0143/train.png": 8, "261026_0143/other.png": 8}
    pod = StubPod(["261026_0143", "261026_0410"], on_pod, tmp_path / "src")
    object.__setattr__(pod, "_writes", landed)
    with pytest.raises(Refuse) as e:
        sweep(pod, "/r/plots/diagnostics", tmp_path / "dest", final=False)
    msg = str(e.value)
    assert "261026_0410/eval.png" in msg, "the refusal must name what is actually missing"
    assert len(on_pod) == len(landed), "fixture must keep counts equal or it tests nothing"

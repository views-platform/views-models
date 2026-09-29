"""
Falsification audit of the claim, 2026-09-29:
"We are ready to start a new full 40-lesson FAO delivery run — nothing remains but
creating a pod."

Source: /falsify skill, claim mode. Verdict: FALSIFIED, three hard falsifications.

All three had ONE cause: every fix that made the 2026-09-29 run work was applied BY HAND
to the pod, and the pod was destroyed. The repository never learned any of them. The pod
was the artefact and nothing in git described it.

  H1  the run could not be launched: pod_run_model.sh refuses total_lessons < 300 and every
      config declares 300, so a cheap end-to-end test was reachable only by deleting the
      guard — producing output indistinguishable from a real run.
  H2  a fresh env built SUCCESSFULLY and WRONG: the declared requirements resolved to
      xarray 2025.12.0 / pandas 3.0.6 where the working pod had 2024.3.0 / 1.5.3, neither
      pinned. #516 called this "unbuildable"; it is not, and the silent version is worse.
  H3  `appwrite` appeared nowhere in pod_run_model.sh, so `_build_datastore` failed at
      publish — after the full training run. That is how 2026-09-29 failed (#517).

────────────────────────────────────────────────────────────────────────────────────────
SECOND AUDIT, same day: this file's FIRST version was 22 green assertions that protected
NOTHING. An independent /falsify guard-mode pass reverted every fix in the commit, kept the
comments, and got 22/22 green. 20 of 24 mutations survived; 10 guards were DECORATIVE.

The mechanism, and it is the lesson: every assertion read the script as TEXT, and
pod_run_model.sh is unusually well commented — each fix carries a paragraph naming the
incident and quoting the exact strings. So THE BETTER THE COMMENT, THE WEAKER THE GUARD:
deleting the code left the comment, and the comment satisfied the assertion.

Worst single case: `requested = os.environ.get("REHEARSAL_LESSONS") or ""` changed to
`or "40"`. One word. Every production run then patches itself to 40 lessons, the floor is
dead, and because the SHELL variable stays empty the MANIFEST says `mode: production` and
no REHEARSAL marker is written. A 40-lesson model, labelled a production delivery, 22/22
green. Two of the five defects fused by a one-token diff.

Worse still, one assertion certified a safety property that DOES NOT EXIST: it claimed the
RunPod guide documents the /workspace chmod trap. It does not. The regex matched because
`/workspace` appears on line 104 and `chmod` on line 180, joined by `.*` under re.S — the
identical defect the first commit message boasted of having found and deleted elsewhere.

So this version:
  * EXECUTES the config-check program against fixture models, rather than grepping it. That
    block is where all the rehearsal/production logic lives and it was wholly unguarded.
  * STRIPS COMMENTS before any remaining text assertion, so a comment can never stand in
    for code.
  * asserts version-specifier SEMANTICS (is 1.5.3 allowed? is 3.0.6 refused?) instead of
    the mere presence of a pin — `pandas>=3.0` passed the old guard.
  * CALLS get_hp_config() instead of pattern-matching the literal, which is what
    pod_run_model.sh itself insists on doing, citing #501 "the guard that was not one".
"""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

REPO = Path(__file__).resolve().parents[1]
PODRUN = REPO / "tools" / "podrun" / "pod_run_model.sh"

HYDRANETS = (
    "purple_alien", "pink_pirate", "blue_stranger", "bright_starship",
    "heavy_freighter", "blazing_meteor", "bold_comet", "violet_visitor",
)


def _code_only(src: str) -> str:
    """Shell source with comments removed.

    Every decorative guard in the first version of this file was satisfied by a comment
    after its code was deleted. Any assertion about what the script DOES must therefore
    read only what the script RUNS. Full-line and trailing comments both go; this is
    deliberately cruder than a shell parser, and cruder is the safe direction here — it
    removes text an assertion might otherwise lean on.
    """
    out = []
    for line in src.splitlines():
        stripped = re.sub(r"(?<![$\\])#.*$", "", line)
        if stripped.strip():
            out.append(stripped)
    return "\n".join(out)


def _extract_cfgcheck() -> str:
    """The config-check program, lifted out of its heredoc so it can be RUN."""
    src = PODRUN.read_text()
    m = re.search(r"<<'CFGCHECK'[^\n]*\n(.*?)\nCFGCHECK\s*?\n", src, re.S)
    assert m, "the CFGCHECK heredoc is gone from pod_run_model.sh — re-read the script"
    return m.group(1)


def _make_model(tmp_path: Path, lessons_body: str, region: str = '"land"') -> Path:
    """A fixture model directory inside a real git repo.

    Real git, because the leftover-patch detector runs `git status --porcelain`; a fake
    would let that guard pass without the mechanism it depends on existing.
    """
    model = tmp_path / "model"
    (model / "configs").mkdir(parents=True)
    (model / "configs" / "config_hyperparameters.py").write_text(lessons_body)
    (model / "configs" / "config_queryset.py").write_text(f"REGION = {region}\n")
    for cmd in (
        ["git", "init", "-q", "."],
        ["git", "config", "user.email", "t@t"],
        ["git", "config", "user.name", "t"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "fixture"],
    ):
        subprocess.run(cmd, cwd=model, check=True, capture_output=True)
    return model


def _run_cfgcheck(model: Path, out: Path, rehearsal: str = "") -> subprocess.CompletedProcess:
    out.mkdir(parents=True, exist_ok=True)
    script = out / "_cfgcheck.py"
    script.write_text(_extract_cfgcheck())
    return subprocess.run(
        [sys.executable, str(script), str(model)],
        capture_output=True, text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "REHEARSAL_LESSONS": rehearsal,
            "OUT": str(out),
            "MODEL": "fixture_model",
            "HOME": str(out),
        },
    )


def _config_lessons(model: Path) -> int:
    """What get_hp_config() RETURNS — not what the file text says."""
    path = model / "configs" / "config_hyperparameters.py"
    spec = importlib.util.spec_from_file_location(f"_probe_{id(path)}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_hp_config()["total_lessons"]


PRODUCTION_CONFIG = "def get_hp_config():\n    return {'total_lessons': 300}\n"
CHEAP_CONFIG = "def get_hp_config():\n    return {'total_lessons': 40}\n"


class TestTheProductionFloorActuallyRefuses:
    """H1, executed. The floor is the only thing standing between an accidental config and
    a full GPU budget, and the first version of this file asserted its SOURCE LINE while a
    mutation that changed `sys.exit(` to `print(` sailed through: the run announced
    "expected >= 300" and trained anyway."""

    def test_a_cheap_config_is_refused_with_a_nonzero_exit(self, tmp_path):
        model = _make_model(tmp_path, CHEAP_CONFIG)
        r = _run_cfgcheck(model, tmp_path / "out")
        assert r.returncode != 0, (
            "a 40-lesson config must REFUSE in production mode. Exit code 0 here means the "
            f"pod trains a throwaway model on paid hardware.\nstdout: {r.stdout}"
        )
        assert "--rehearsal" in (r.stdout + r.stderr), (
            "the refusal must name the supported way to get a cheap run, or the operator "
            "edits the guard out — which is the state this audit found"
        )

    def test_a_production_config_passes(self, tmp_path):
        model = _make_model(tmp_path, PRODUCTION_CONFIG)
        r = _run_cfgcheck(model, tmp_path / "out")
        assert r.returncode == 0, f"a 300-lesson production run must proceed.\n{r.stdout}\n{r.stderr}"

    def test_production_mode_does_not_patch_the_config(self, tmp_path):
        """The one-word mutation `or ""` -> `or "40"` made every production run patch itself
        to 40 lessons while the MANIFEST still said `mode: production`. Nothing saw it."""
        model = _make_model(tmp_path, PRODUCTION_CONFIG)
        _run_cfgcheck(model, tmp_path / "out")
        assert _config_lessons(model) == 300, (
            "a production run must not modify the config it was given. If this fails, a run "
            "labelled 'production' in the MANIFEST trained on a patched lesson count."
        )
        assert (tmp_path / "out" / ".lessons").read_text().strip() == "300"


class TestRehearsalPatchesOnlyThePodAndProvesIt:
    """H1 / #523, executed."""

    def test_rehearsal_patches_the_config_and_records_the_real_value(self, tmp_path):
        model = _make_model(tmp_path, PRODUCTION_CONFIG)
        r = _run_cfgcheck(model, tmp_path / "out", rehearsal="40")
        assert r.returncode == 0, f"--rehearsal 40 must proceed.\n{r.stdout}\n{r.stderr}"
        assert _config_lessons(model) == 40, "the pod's config must actually be patched"
        assert (tmp_path / "out" / ".lessons").read_text().strip() == "40", (
            "the MANIFEST reads .lessons; it must carry the value the run was gated on"
        )

    def test_a_patch_that_does_not_take_is_refused_not_believed(self, tmp_path):
        """The adversarial case the 'verification' exists for.

        Here the literal is patchable but `get_hp_config()` returns 300 regardless — a
        second assignment wins. A verification that trusted re.subn's return value, or that
        compared `target` against itself, would report a successful 40-lesson rehearsal
        while the model trains 300 lessons. The mutation that did exactly that (replacing
        the re-import with `lessons = target`) survived the first version of this file.
        """
        sneaky = (
            "def get_hp_config():\n"
            "    d = {'total_lessons': 300}\n"
            "    d['total_lessons'] = 300\n"
            "    return d\n"
        )
        model = _make_model(tmp_path, sneaky)
        r = _run_cfgcheck(model, tmp_path / "out", rehearsal="40")
        assert r.returncode != 0, (
            "the patch did not take — get_hp_config() still returns 300 — and the run "
            f"proceeded anyway. It must refuse.\nstdout: {r.stdout}"
        )
        assert "still reports total_lessons" in (r.stdout + r.stderr)

    def test_a_leftover_patch_stops_a_later_production_run(self, tmp_path):
        """Section 2 does not re-clone when .git exists, so a production run on a pod that
        has rehearsed reads the LEFTOVER patch. It must name that, not blame the committed
        config and send the operator to edit the wrong file."""
        model = _make_model(tmp_path, PRODUCTION_CONFIG)
        assert _run_cfgcheck(model, tmp_path / "o1", rehearsal="40").returncode == 0
        r = _run_cfgcheck(model, tmp_path / "o2")
        assert r.returncode != 0, "a production run must refuse a dirty config"
        assert "MODIFIED in this checkout" in (r.stdout + r.stderr), (
            "the refusal must name the leftover rehearsal patch. Blaming the committed "
            "config is a correct refusal for the wrong reason."
        )

    def test_the_region_gate_still_fires(self, tmp_path):
        """Not part of this audit's findings — pinned because the rehearsal work edited the
        block this lives in, and it guards 'this would not be a global-land run'."""
        model = _make_model(tmp_path, PRODUCTION_CONFIG, region='"africa"')
        r = _run_cfgcheck(model, tmp_path / "out")
        assert r.returncode != 0 and "REGION" in (r.stdout + r.stderr)


class TestTheRehearsalFlagIsActuallyParsed:
    """The shell argument parser, executed.

    An independent audit deleted the `--rehearsal)` case arm outright — so `--rehearsal`
    fell through to `--*` and was rejected as an unknown option, removing H1's escape hatch
    entirely — and every guard stayed green, because the string `--rehearsal` survived in the
    header comment and in USAGE. Executing the config-check program does not cover this: the
    flag never reaches Python if the shell refuses it first.

    These run the real script. Argument parsing precedes `mkdir -p "$OUT"`, so each case
    below exits before the script touches the filesystem or spends anything.
    """

    @staticmethod
    def _run(*args):
        return subprocess.run(
            ["bash", str(PODRUN), *args],
            capture_output=True, text=True, cwd=str(REPO), timeout=60,
        )

    def test_the_flag_and_its_count_are_consumed_not_rejected(self):
        """With the count consumed, the script must get as far as demanding a model name."""
        r = self._run("--rehearsal", "40")
        combined = r.stdout + r.stderr
        assert "unknown option" not in combined, (
            "--rehearsal was rejected as an unknown option, so the case arm parsing it is "
            f"gone and a cheap run is unreachable again.\n{combined}"
        )
        assert "usage" in combined.lower(), (
            f"expected the missing-model-name usage error once the flag was consumed.\n{combined}"
        )

    def test_a_count_is_required(self):
        r = self._run("--rehearsal")
        assert r.returncode != 0
        assert "lesson count" in (r.stdout + r.stderr), (
            "--rehearsal with no count must say so. Silently defaulting is how a rehearsal "
            "becomes indistinguishable from a production run."
        )

    def test_a_non_integer_count_is_refused(self):
        r = self._run("--rehearsal", "forty", "purple_alien")
        assert r.returncode != 0
        assert "positive integer" in (r.stdout + r.stderr)

    def test_a_genuinely_unknown_option_is_still_refused(self):
        """The control: the `--*` arm must keep working, or the test above proves nothing."""
        r = self._run("--bogus", "purple_alien")
        assert r.returncode != 0
        assert "unknown option" in (r.stdout + r.stderr)


class TestTrackedConfigsDeclareTheProductionCount:
    """A3. Calls get_hp_config() rather than matching the literal.

    pod_run_model.sh lines 173-176 refuse to pattern-match a config, citing #501 "the guard
    that was not one — a substring assertion satisfied by a COMMENT recording the value's
    history". The first version of this guard pattern-matched the config. A mutation that
    appended `d['total_lessons'] = 40` before the return left the 300 literal untouched and
    the guard green, while the pod would read 40.
    """

    @pytest.mark.parametrize("model", HYDRANETS)
    def test_the_value_the_pod_will_read_is_a_production_count(self, model):
        cfg_dir = REPO / "models" / model / "configs"
        assert (cfg_dir / "config_hyperparameters.py").exists(), f"{model}: no config"
        lessons = _config_lessons(cfg_dir.parent)
        assert lessons >= 300, (
            f"{model}: get_hp_config() returns total_lessons={lessons}. A rehearsal must be "
            "obtained with --rehearsal, which patches only the pod's clone — never by "
            "committing a low count."
        )


class TestTheEnvironmentPinsAreCORRECTNotMerelyPRESENT:
    """H2 (#516). Asserts the measurement, not a proxy for it.

    The first version checked that a pin EXISTED. `xarray>=2025.12,<2026` and
    `pandas>=3.0,<4.0` — the exact versions measured as the defect — passed it, and its own
    failure message ("this file resolves to pandas 3.0.6") was unreachable in the state that
    resolves pandas 3.0.6.

    Measured 2026-09-29: xarray 2024.3.0 is the LAST release accepting pandas 1.x; 2024.5.0
    moved to pandas>=2.0 and 2024.9.0 to pandas>=2.1. So a `<2025` cap would look right and
    be wrong — the cliff is inside the 2024 line.
    """

    # (package, must be allowed, must be refused, why the refused one matters)
    CASES = (
        ("pandas", "1.5.3", "3.0.6", "the version a fresh resolve silently picked"),
        ("pandas", "1.5.3", "2.1.0", "any pandas 2.x is a different major from the platform's"),
        ("xarray", "2024.3.0", "2025.12.0", "the version a fresh resolve silently picked"),
        ("xarray", "2024.3.0", "2024.11.0", "already requires pandas>=2.1 — inside the 2024 line"),
        ("numpy", "1.26.4", "2.0.0", "the pandas wheel is built against the numpy 1.x C ABI"),
    )

    @staticmethod
    def _spec(postprocessor: str, package: str) -> SpecifierSet:
        text = (REPO / "postprocessors" / postprocessor / "requirements.txt").read_text()
        found = [
            ln.strip() for ln in text.splitlines()
            if re.match(rf"^\s*{package}\s*[><=!~]", ln) and ";" not in ln
        ]
        assert found, (
            f"{postprocessor}/requirements.txt declares no unconditional pin for {package}. "
            "Unpinned, this file resolves pandas to 3.0.6 with no error. See #516. "
            "(A marker-gated pin is excluded deliberately: `; python_version < \"3.10\"` is "
            "inert on the 3.11 the runner builds, and looked identical to a real pin.)"
        )
        assert len(found) == 1, f"{postprocessor}: {package} pinned {len(found)} times: {found}"
        return SpecifierSet(found[0][len(package):].strip())

    @pytest.mark.parametrize("postprocessor", ["un_fao", "un_crafd"])
    @pytest.mark.parametrize("package,allowed,refused,why", CASES)
    def test_the_pin_admits_the_working_version_and_refuses_the_broken_one(
        self, postprocessor, package, allowed, refused, why
    ):
        spec = self._spec(postprocessor, package)
        assert spec.contains(allowed), (
            f"{postprocessor}: {package}{spec} excludes {allowed}, which is what every "
            "successful run has used. This env would not build."
        )
        assert not spec.contains(refused), (
            f"{postprocessor}: {package}{spec} ADMITS {refused} — {why}. The pin exists but "
            "does not constrain what it was added to constrain."
        )

    @pytest.mark.parametrize("package", ["numpy", "pandas", "xarray"])
    def test_both_postprocessors_constrain_each_package_identically(self, package):
        """C-116: one shared prefix, so the last postprocessor to run decides for both.

        Compares SPECIFIER SEMANTICS, not captured strings. The old string compare was
        defeated by a marker suffix that made un_fao's pins inert while keeping the captured
        text identical to un_crafd's real ones.
        """
        fao, crafd = self._spec("un_fao", package), self._spec("un_crafd", package)
        probes = ["1.5.3", "2.0.0", "2.1.0", "3.0.6", "1.26.4", "2024.3.0", "2024.11.0", "2025.12.0"]
        differ = [v for v in probes if fao.contains(v) != crafd.contains(v)]
        assert not differ, (
            f"un_fao pins {package}{fao} and un_crafd pins {package}{crafd}; they disagree "
            f"on {differ}. Both install into envs/views-postprocessing (C-116), so whichever "
            "runs last silently decides these versions for the other."
        )


class TestThePublishPathIsInstalledAndProven:
    """H3 (#517). Asserted against comment-stripped source.

    Both of the first version's assertions here were satisfied by a five-line prose comment
    that named `views-pipeline-core[appwrite]`: deleting the extra from the install line and
    commenting out both imports left the suite green, with preflight printing "appwrite
    client importable — the publish path exists" having imported nothing.
    """

    def test_the_appwrite_extra_is_in_an_install_command(self):
        code = _code_only(PODRUN.read_text())
        install_lines = [
            ln for ln in code.splitlines()
            if "pip install" in ln or (ln.strip().startswith('"') and "appwrite" in ln)
        ]
        assert any("views-pipeline-core[appwrite]" in ln for ln in install_lines), (
            "no install command requests the appwrite extra. Without it _build_datastore "
            "raises at publish, AFTER the full training run — the 2026-09-29 failure (#517). "
            f"install lines seen: {install_lines}"
        )

    def test_datafactory_is_floored_where_the_credential_fixes_landed(self):
        """#509. The runner only ever executes on hardware we do not own, carrying a netrc
        credential. Before views-datafactory 1.13.0 the client could carry that credential
        across a redirect to another host and embed it in error messages. The model
        requirements still say >=1.9.0 and a resolver will usually pick the newest — but
        "the resolver will probably do the right thing" is the reasoning that put pandas
        3.0.6 into a fresh environment (#516)."""
        sources = {"tools/podrun/pod_run_model.sh": _code_only(PODRUN.read_text())}
        for pp in ("un_fao", "un_crafd"):
            rel = f"postprocessors/{pp}/requirements.txt"
            sources[rel] = (REPO / rel).read_text()
        for where, text in sources.items():
            floors = re.findall(r"^\s*(?:\S*\s+)?\"?views-datafactory>=(\d+)\.(\d+)",
                                text, re.M)
            assert floors, f"{where} does not request views-datafactory at all"
            for major, minor in floors:
                assert (int(major), int(minor)) >= (1, 13), (
                    f"{where} declares views-datafactory>={major}.{minor}; the "
                    "credential-handling fixes landed in 1.13.0 (#509). Every leg that runs "
                    "on rented hardware holds the netrc credential, not just the training one."
                )

    def test_preflight_imports_the_client_not_just_mentions_it(self):
        code = _code_only(PODRUN.read_text())
        verify = code.split("stage verify_env", 1)[-1].split("stage ", 1)[0]
        assert re.search(r"^\s*import appwrite", verify, re.M) or re.search(
            r"^\s*from views_pipeline_core\.modules\.appwrite import", verify, re.M
        ), (
            "verify_env must IMPORT the Appwrite client, not merely mention it. The whole "
            "script is written to fail early, and the one dependency that failed late was "
            "the one not checked here."
        )

    def test_the_toolz_override_is_an_install_and_is_last(self):
        code = _code_only(PODRUN.read_text())
        installs = [m.start() for m in re.finditer(r"pip install", code)]
        overrides = [
            m.start() for m in re.finditer(r"pip install[^\n]*toolz>=0\.12", code)
        ]
        assert overrides, (
            "the C-151 toolz override is not an install command any more. A comment saying "
            "the base image ships toolz>=0.12.1 satisfied the old guard while C-151 was "
            "reintroduced; the resolver really does land on toolz 0.11.2 for this set."
        )
        assert max(overrides) == max(installs), (
            "the toolz override must be the LAST pip install: anything after it re-resolves "
            "the prefix and can pull toolz back under 0.12, which happened by hand on "
            "2026-09-29 (1.1.0 -> 0.11.2) with no error."
        )


class TestARehearsalIsMarkedInTheOutput:
    """The last line of defence, since nothing REFUSES to publish a rehearsal.

    Asserted against comment-stripped source. The old guards passed with the marker-writing
    block deleted outright — `"$OUT/REHEARSAL"` was supplied by the `rm -f` on line 83 and
    "NOT FIT TO DELIVER" by a MANIFEST line.
    """

    def test_the_marker_is_written_not_only_removed(self):
        code = _code_only(PODRUN.read_text())
        writes = re.findall(r'>\s*"\$OUT/REHEARSAL"', code)
        assert writes, (
            "nothing WRITES $OUT/REHEARSAL. A guard satisfied by the `rm -f` that clears it "
            "is green in the state where no rehearsal is ever marked."
        )

    def test_the_marker_is_cleared_before_the_run_not_after(self):
        code = _code_only(PODRUN.read_text())
        clear = code.index('rm -f "$OUT/REHEARSAL"')
        assert clear < code.index("stage preflight"), (
            "the marker must be cleared with the other stale per-run state, before the run. "
            "Moved to the end, a successful rehearsal deletes its own marker on the way out."
        )

    def test_the_manifest_reports_the_mode_on_the_correct_branch(self):
        """Swapping the two MANIFEST branches made production runs claim PATCHED and
        rehearsals claim as-committed. The old guard checked only that the string existed."""
        code = _code_only(PODRUN.read_text())
        m = re.search(
            r'if \[ -n "\$REHEARSAL_LESSONS" \]; then(.*?)else(.*?)fi',
            code, re.S,
        )
        assert m, "the MANIFEST mode branch is not in the expected shape; re-read it"
        rehearsal_branch, production_branch = m.group(1), m.group(2)
        assert "NOT FIT TO DELIVER" in rehearsal_branch and "PATCHED after checkout" in rehearsal_branch
        assert "NOT FIT TO DELIVER" not in production_branch, (
            "the production branch of the MANIFEST claims the output is unfit to deliver"
        )
        assert "PATCHED" not in production_branch, (
            "the production branch claims the config was patched after checkout"
        )


class TestTheGuideDocumentsTheChmodTrap:
    """C-154 / #518. This property DID NOT EXIST when it was first asserted.

    The original assertion claimed the guide records the /workspace chmod trap and passed
    because `/workspace` appears at line 104 and `chmod` at line 180, joined by `.*` under
    re.S. An independent audit read the guide and found nothing: no "world-readable", no
    "network filesystem", no C-154, no #518. The guard certified a safety property that was
    absent, which is worse than having no guard, because it stopped the next reader looking.

    The warning has since been written. This asserts its substance on a bounded window, not
    two words anywhere in a 400-line file.
    """

    def test_the_trap_is_documented_in_substance(self):
        text = (REPO / "docs" / "runpod_run_guide.md").read_text()
        assert re.search(r"C-154|#518", text), "the guide cites neither C-154 nor #518"
        windows = [
            text[m.start(): m.start() + 700]
            for m in re.finditer(r"chmod", text)
        ]
        assert any(
            re.search(r"/workspace", w)
            and re.search(r"ignore|silent|no effect|world-readable|666", w, re.I)
            for w in windows
        ), (
            "the guide must state, near a chmod instruction, that /workspace SILENTLY "
            "ignores chmod and leaves the credential world-readable. Two words 76 lines "
            "apart is what the deleted version of this guard accepted."
        )

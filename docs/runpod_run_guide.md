# RunPod Run Guide

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-09-28
**Tooling status:** `tools/podrun` is **v0.1.0, provisional** — two model families (HydraNet, r2darts2), one partition, no CI. See its `__init__.py`.
**Related:** `reports/postmortem_runpod_first_deployment_2026-09.md` (the *why*, and every number quoted here); `reports/runpod_cost_and_time_note_2026-09.md` (what it costs); runbook #499; spec #505; epic #532 (the darts chain, Phase 4c)

> How to train VIEWS models on rented cloud GPUs when our own hardware is unavailable. This is
> the *how* and the *gates*; read the post-mortem for the *why*. Every rule below was paid for
> once already — the first machine rented under this procedure was 25× too slow, and the first
> credential we nearly created did not exist.

---

## Ground rules

1. **Rank machines by RAM, then vCPU, then VRAM. Never the reverse.**
   *Why: VRAM is the number the console leads with and the one that does not matter here — peak
   usage is ~3 GB against 24 available. The machine that failed had the best price and the worst
   RAM.*

2. **Read `/sys/fs/cgroup/`, not the console.**
   *Why: a machine advertised as "16 vCPU" gives `cpu.max` of `1360000 100000` — **13.6**
   effective. An 8-vCPU listing gives 6.8. Memory limits differ from the advertised figure too.*

3. **Add your SSH key to the RunPod account BEFORE creating any pod.**
   *Why: keys are injected at container start. Adding one to a running pod does nothing, and
   restarting does not re-inject it — the environment was fixed at creation.*

4. **Smoke-test new hardware at throwaway length before committing a budget.**
   *Why: this is the single practice that saved the first campaign. Forty minutes and under a
   dollar caught a machine that would have consumed the entire budget.*

5. **Publish credentials never go on rented hardware.**
   *Why: a read credential and a write credential are different decisions. The datafactory key
   only reads. Keys that write to stores our partners consume stay on hardware we control.*

   *This is supported by an interlock, not by wishful thinking: views-postprocessing defaults
   `UPLOAD_ENABLED` to `False` and constructs no store client at all when disarmed, so a run can
   be made that touches no partner system. **What has not been tested is whether a delivery staged
   on one machine can then be published from another.** There is a single entrypoint and no
   `--no-upload` flag; disarming is a governance switch on a committed delivery declaration, not
   an ops convenience. Treat "produce here, publish there" as an open question, not a procedure.*

   *The specific way this rule gets broken is by copying a sibling: `pod_run_fao_delivery.sh`
   **legitimately** needs nine publish variables and the `[appwrite]` extra, because its job is
   to publish. A calibration runner's job is not, so it must install neither — and "start from
   the script that already works" is the obvious and wrong way to write the next one.
   `pod_run_darts_calibration.sh` installs no Appwrite path and reads no publish variable, and
   `tests/test_darts_calibration_runner.py` asserts both, because a comment saying so would not
   survive the next edit.*

---

## Prerequisites

1. **A RunPod account with credit.** Roughly $3 per model — see the cost note.
2. **Your SSH public key uploaded to the account** (Ground rule 3). `~/.ssh/id_ed25519.pub`.
3. **Datafactory access.** This is **not** an environment variable. It is a `~/.netrc` entry for
   the data server, mode 600. There is a `VIEWS_DATAFACTORY` line in `.env.example` — **it is a
   placeholder that no code reads.** Do not create it as a secret; it will do nothing.
   The real entry looks like this in shape (**never commit a real one**):

   ```
   machine <data server IP>
     login <your personal login>
     password <your password>
   ```

   The login is **per person**, not shared. A correct password under the wrong login returns 401.

4. **Awareness that the datafactory speaks plain HTTP.** The credential crosses the public
   internet on every request, readable — base64 is not encryption. On our own network that was an
   accepted risk (register **C-318** in views-datafactory); from a rented datacentre it is a
   different one.

   **The alternative is cheaper than it looks.** A throwaway login is about three commands and
   two minutes for whoever administers the data server — ask for one rather than assuming you
   must reuse your own. Retire it when the campaign ends. If you do reuse your personal login,
   that is a decision, not a default.

5. **`WANDB_MODE=offline`, which this guide sets everywhere.** `main.py` calls `wandb.login()`
   unconditionally, so without it the pod would need a Weights & Biases credential. Offline mode
   makes that call a no-op and writes run records to disk instead. Every run in the first campaign
   completed this way with no W&B key on any pod. **If you unset it, you have added a credential
   requirement** — and W&B run records from a machine you do not control are a separate decision.

6. **Nothing else.** No Appwrite variables, no publishing credentials. If a procedure seems to
   need them on the pod, stop — see Ground rule 5.

---

## Phase 1 — Rent a machine

### Step 1.1 — Apply the selection rule

**RAM ≥ 50 GB · vCPU ≥ 12 · VRAM ≥ 24 GB**

Machines that satisfied it in practice: `RTX PRO 4500 SE`, `RTX PRO 4500`, `RTX A6000`,
`RTX 4090`, `RTX 6000 Ada`, `RTX 5090`, `L40`, `L40S`.

**Do not take `PRO 6000 MIG 24GB`** (31 GB RAM, 6.8 effective CPUs). It is the cheapest listing
and it was 25× slower — 0.11 posterior-sampling steps/s against 2.8 on a machine that fits.

Availability churns on a scale of seconds; listings vanish mid-form. Hold to the *rule* rather
than a favourite model, or the hunt becomes the bottleneck.

### Step 1.2 — Create the pod

Template `Runpod Pytorch 2.8.0`, **container disk 100 GB**, **volume disk 100 GB at
`/workspace`, encrypted**.

The volume matters: container disk is erased when a pod stops, and a seven-hour run should not
live on it.

### Step 1.3 — Connect

Use the **SSH over exposed TCP** line from the Connect tab (it supports file copy; the other one
does not).

```bash
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519
```

**The port changes on every restart, and the IP may too.** Re-read the Connect tab after any stop.

---

## Phase 2 — Prepare the pod

### Step 2.1 — Verify what you actually rented

```bash
echo "RAM $(( $(cat /sys/fs/cgroup/memory.max) / 1073741824 )) GB"
awk '{printf "CPUs %.1f\n", $1/$2}' /sys/fs/cgroup/cpu.max
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Must show **≥ 50 GB**, **≥ 12.0 CPUs**, **≥ 24 GB VRAM**. If not, terminate and take another —
you have spent pennies.

### Step 2.2 — Install

```bash
apt-get update -qq && apt-get install -y -qq libpq-dev build-essential zstd rsync
cd /workspace && git clone --depth 1 -b development \
    https://github.com/views-platform/views-models.git
uv venv --python 3.11 /workspace/venv
uv pip install --python /workspace/venv/bin/python \
    "views-hydranet~=0.1.1" "views-datafactory>=1.13.0,<2.0.0"
uv pip install --python /workspace/venv/bin/python "toolz>=0.12.1"
```

Three of those lines are not obvious and each cost a failed install:

- **`libpq-dev`** — `views-hydranet` still pulls in `viewser`, which needs Postgres headers to
  build `psycopg2`. Without it the install dies partway.
- **Python 3.11**, not the image's 3.12.
- **`views-datafactory>=1.13.0`**, not the `>=1.9.0` the model requirements still carry. The
  credential-handling fixes landed in 1.13.0: before it, the client could carry a netrc
  credential across a redirect to another host and could embed it in error messages. A
  resolver will normally pick the newest anyway — 1.13.0 is what installed on every pod in the
  first campaign — but on a machine you do not control, ask for it explicitly.
- **`toolz>=0.12.1` installed last, deliberately overriding a pin.** Register **views-models C-151** (views-hydranet's C-151 is an unrelated entry — cross-repo
  register IDs collide, so name the repo): viewser pins `toolz<0.12`, which cannot import `tlz` submodules on Python 3.11. **views-datafactory has
  no toolz dependency at all** — viewser poisons the shared environment, and datafactory fetches
  are simply what dies first, because `datafactory_query` imports dask which imports `tlz`.
  Resolve everything first, then override. Do not go looking for toolz in views-datafactory; it
  is not there. This is a **workaround with an expiry**, not settled practice — the durable fix
  is upstream, in whatever still pins `toolz<0.12`.

Verify:

```bash
/workspace/venv/bin/python -c "import torch, tlz.curried, views_hydranet; \
    print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

Must print a torch version, `True`, and a capability tuple.

### Step 2.3 — Place the credential

From your **laptop**, so the secret never passes through a terminal or a log:

```bash
grep -A2 '<data server IP>' ~/.netrc | ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519 \
    'umask 077; cat > /root/.netrc; chmod 600 /root/.netrc'
```

> **Put credentials on `/root`, never on `/workspace` — `/workspace` silently ignores
> `chmod`.**
>
> It is a network filesystem. `chmod 600` there **returns success and does nothing**: the
> file stays mode `666`, readable by every process on the machine, and there is no error to
> notice. `stat -c %a` is the only way to find out, and only if you think to look.
>
> Found on 2026-09-29 while placing the Appwrite publish credentials, which sat
> world-readable on rented hardware until they were moved to `/root/.secrets`. Registered as
> **views-models C-154**; see **#518**.
>
> This is why the command above writes to `/root/.netrc` and why the runner checks
> `stat -c %a /root/.netrc` rather than trusting its own `chmod`. Both are deliberate. The
> `umask 077` is what actually protects the file in transit — the `chmod` is a belt-and-braces
> check that happens to work *here* because `/root` is local disk.

Check it before trusting it — the pipe can silently deliver nothing:

```bash
grep -A2 '<data server IP>' ~/.netrc | wc -l          # must be 3
ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519 'ls -l /root/.netrc'
```

`grep -A2` produces nothing at all if your `~/.netrc` is formatted differently — `login` on the
same line as `machine`, extra indentation, or a second entry for the same host. Count first.

Must end with `/root/.netrc` at mode `600`, **in the home of the runtime user** — `Path.home()`
resolves at call time, so if the pod runs as someone other than root, `/root/.netrc` is the wrong
destination and the fetch will fail with no obvious cause.

Confirm it authenticates before spending anything:

```bash
/workspace/venv/bin/python -c "
from netrc import netrc; from pathlib import Path
import urllib.request, base64
from datafactory_query.defaults import DEFAULT_REMOTE
login, _, pw = netrc(str(Path.home()/'.netrc')).authenticators(DEFAULT_REMOTE.server)
tok = base64.b64encode((login + ':' + pw).encode()).decode()
url = DEFAULT_REMOTE.zarr_url.rstrip('/') + '/.zmetadata'
r = urllib.request.urlopen(urllib.request.Request(
    url, headers={'Authorization': 'Basic ' + tok}), timeout=30)
print('HTTP', r.status)"
```

Must print `HTTP 200`.

**Use `.zmetadata`, not the bare store URL.** `DEFAULT_REMOTE.zarr_url` has no trailing slash and
the server answers it with a **308 redirect**. `urllib` follows that redirect and carries the
`Authorization` header with it, because `Request(headers=...)` puts it in `req.headers` rather
than `unredirected_hdrs` — measured on a live pod, 2026-09-28. So a check against the bare URL
passes, but only by sending the credential across a redirect. It is same-host today and leaks
nothing, but it is the pattern views-datafactory removed in #388, and an operator who hardens it
(refusing redirects, or switching to `requests` with `allow_redirects=False`) gets a 308 and
wrongly concludes the credential is broken. `.zmetadata` is a real object and answers 200
directly.

---

## Phase 3 — Smoke test (DECISION GATE)

**Do not skip this.** Set one model to a throwaway length and run it whole.

```bash
cd /workspace/views-models/models/<model>
sed -i "s/'total_lessons': 300/'total_lessons': 2/" configs/config_hyperparameters.py
export WANDB_MODE=offline WANDB_SILENT=true
/workspace/venv/bin/python main.py -r calibration -t -e
```

Watch the posterior-sampling rate during evaluation.

| observed rate | verdict |
|---|---|
| **≥ 2 steps/s** | healthy — proceed |
| **< 0.5 steps/s sustained** | terminate the pod and take another |

**This threshold is empirical and pod-relative, not a hardware expectation.** It comes from n=1
good machine (2.8 steps/s) and n=1 bad one (0.11), and it discriminates those two, which is all an
operator needs. It is *not* a ceiling: a step is one model forward pass, and an RTX 4070 laptop
does the bare forward at global-land extent at ~15 steps/s. A healthy pod at 2.8 is therefore
already spending most of its time off the GPU — which is why the caveats below say the GPU looks
idle. Do not size a machine by trying to raise this number; size it by RAM and vCPU.

A healthy machine completes this in ~35–60 minutes. **Restore `total_lessons` to 300 before
Phase 4**, or re-clone the repo.

---

## Phase 4 — The real run

Use the runner rather than driving `main.py` by hand — it carries the preflight checks.
It is `tools/podrun/pod_run_model.sh`, **v0.1.0 and provisional**: one model family, one
partition, no CI, no uploads. Read `tools/podrun/__init__.py` before relying on it for
anything this guide does not describe.

```bash
cd /workspace/views-models
nohup setsid bash tools/podrun/pod_run_model.sh <model> \
    > /workspace/<model>.nohup 2>&1 < /dev/null &
```

`setsid` matters: the run must survive your SSH session closing and your laptop sleeping.

It refuses before spending GPU time if `.netrc` is missing, if `total_lessons` is still a
throwaway value, if `REGION` is not `"land"`, if the Appwrite client is not importable, or
if the converter is absent.

### Rehearsing the chain first

After a run that failed late, you usually want the whole chain exercised cheaply before you
commit to a full one. That is `--rehearsal`, and it takes the lesson count:

```bash
nohup setsid bash tools/podrun/pod_run_model.sh --rehearsal 40 <model> \
    > /workspace/<model>.nohup 2>&1 < /dev/null &
```

It patches **the pod's clone** of `config_hyperparameters.py` to 40 lessons, re-reads the
config to confirm the patch took, and marks the output. Nothing tracked in git is edited —
the committed configs stay at their production value, which is the point: a rehearsal
obtained by committing a low lesson count is the failure mode this flag removes.

A rehearsal leaves an extra file, **`REHEARSAL`**, beside the parquets, and `MANIFEST` gains
`mode: REHEARSAL — NOT FIT TO DELIVER`. You need them, because the parquets themselves are
indistinguishable from a real run's: same columns, same row counts, all finite and
non-negative, every structural check green. Nothing in the data will tell you the model
behind it is undertrained.

**What a rehearsal does NOT prove.** It exercises the chain, not the capacity. A 40-lesson
run has a different duration and memory profile from a 300-lesson one, and memory is where
this platform has failed before. A green rehearsal means the hops connect; it does not mean
the full run will fit.

**It marks, it does not refuse.** This runner uploads nothing, so it cannot stop a rehearsal
being published downstream — the `REHEARSAL` file is a warning to a person, not a guard. Do
not publish a directory that contains one.

Monitor without disturbing it:

```bash
cat /workspace/deliver/<model>/STAGE
tr '\r' '\n' < /workspace/deliver/<model>/run.log | grep -oE 'Lesson [0-9]+/300' | tail -1
```

Expect **~4 hours per model, end to end** — 300 lessons of training plus the 13-origin
evaluation. Measured n=3 on RTX PRO 4500 SE class hardware: **202, 253, 272 minutes**.
Training alone is roughly four fifths of that. On fimbulthul's A10 the same work took ~5.5 h,
so these machines are somewhat faster, not slower.

**Running several models at once:** one per pod, not several per pod. Assign models explicitly
and record the assignment — two pods running the same model is silent waste.

---

## Phase 5 — Collapse and bring it home

The runner already does both. Its output is in `/workspace/deliver/<model>/`:

| | |
|---|---|
| `parquet/` | 13 files, one per origin, 2,333,448 rows each |
| `draws/` | the full posterior, zstd — ~236× smaller than raw |
| `MANIFEST` | sizes, timestamps, git sha, lesson count, and `mode:` |
| `STATUS` | `OK`, or `FAILED:<stage>` |
| `REHEARSAL` | present **only** after `--rehearsal` — this output is not fit to deliver |

From your laptop:

```bash
rsync -a -e "ssh -p <port> -i ~/.ssh/id_ed25519" \
    root@<ip>:/workspace/deliver/<model>/ \
    models/<model>/data/generated/calibration_delivery_<date>/
```

Must transfer **~18 MB**. If it is trying to move gigabytes, you are copying raw predictions
rather than the deliverable — check the path.

Verify before trusting it:

```bash
python -m tools.collapse.plot_collapse_audit \
    models/<model>/data/generated/calibration_delivery_<date>/parquet/<file>_00.parquet \
    --out audit.png
```

Then **look at the picture**. The tests prove the arithmetic; they cannot tell you the field
stopped looking like conflict.

---

## Phase 4b — The FAO delivery (Track B)

Phases 4 and 5 are **Track A**: calibration predictions for research. The FAO delivery is a
different chain and a different script.

```bash
cd /workspace/views-models

# Seconds, costs nothing, checks everything that can be known before GPU time:
bash tools/podrun/pod_run_fao_delivery.sh --preflight

# Then, once preflight is clean:
nohup setsid bash tools/podrun/pod_run_fao_delivery.sh --rehearsal 40 \
    > /workspace/fao.nohup 2>&1 < /dev/null &
```

Drop `--rehearsal 40` for a production delivery.

It runs: the eight HydraNets on the forecasting partition → `rusty_bucket` pools them from
the saved member forecasts → publish → the `un_fao` postprocessor → a read-back of what
actually landed via `python -m tools.liveness`.

Watch it with `cat /workspace/deliver/_fao/STAGE` or `tail -f /workspace/deliver/_fao/run.log`.

**Run `--preflight` on every fresh pod.** At 300 lessons the eight forecasts alone are ~16 GPU
hours, and the two things most likely to stop the delivery are invisible until the end:
the three Appwrite publish secrets, and **`conda`** — the postprocessor launcher requires it
while this pod builds a `uv` venv, so a pod can satisfy the training leg and not the delivery
leg. That failure lands *after* all the training.

### Reading the outcome

- **`DeliveryNotFindableError`** is views-postprocessing 1.4.0 **working**. That build verifies
  a delivery by what it refuses, and the message names every object it checked. Do not read it
  as the script failing.
- A **rehearsal's forecasts reach the FAO shelf and are servable.** Nothing downstream refuses
  a marked rehearsal (#523), and they are structurally indistinguishable from real forecasts —
  same columns, same coverage, all finite. `/workspace/deliver/_fao/REHEARSAL` says so. **Do
  not leave them there**: supersede or remove them before anyone reads them as a forecast.
- A rehearsal proves the **chain**, not the **capacity**. 40 lessons has a different duration
  and memory profile from 300, and memory is where this platform has failed before.

## Phase 4c — The darts models (r2darts2), calibration

A third chain, for the eleven pgm `views-r2darts2` models — epic **#532**, deliverable spec
**#505**. Same hardware rule, same credential, **different script and a different install.**

Phases 2.2 and 2.3 still apply for the credential; **do not run Phase 2.2's install.** The
runner builds its own environment, and the package set is not the same one.

```bash
cd /workspace/views-models

# Seconds, costs nothing, and refuses for every reason at once:
bash tools/podrun/pod_run_darts_calibration.sh --preflight dark_river

# Then, once preflight is clean:
nohup setsid bash tools/podrun/pod_run_darts_calibration.sh dark_river \
    > /workspace/darts.nohup 2>&1 < /dev/null &
```

Watch it with `cat /workspace/deliver/<model>/STAGE`.

It runs: build the environment → verify it → load and check the model's config → `main.py -r
calibration -t -e` → collapse the output to delivery parquets → verify them.

**Run `dark_river` first, and read its numbers before renting a second pod.** It is the cheapest
of the eleven — NBEATS, three covariates, one sample — and **no r2darts2 model has ever produced
a prediction at pgm**, so its runtime, peak RAM and peak disk are genuinely unknown. #537 exists
to measure them.

### What is different, and why each one cost something

- **The engine is installed from the git tag `0.2.4`, not from PyPI.** PyPI's newest is 0.2.3,
  and 0.2.3 **never deletes its prediction scratch directory**: ~4 000 of them at ~53 GB each
  filled fimbulthul's 2 TB disk on 2026-09-20 (views-r2darts2#54). 0.2.4 frees them, and makes a
  model that cannot be restored to the GPU raise instead of finishing quietly on the CPU. It is
  tagged and deliberately **not published** — a release is irreversible and other repos resolve
  against the range. The runner asserts the installed version rather than trusting the install,
  because a silent fall back to 0.2.3 does not fail; it fills the machine hours later.
- **`[manager]` is not decoration.** `views-pipeline-core` is an *optional* dependency of
  `views-r2darts2`, reachable only through that extra. Without it `main.py`'s first import dies
  with `ModuleNotFoundError: No module named 'views_pipeline_core'` — which is views-models
  **#531**, still open against eight models on `staging_202608` whose `requirements.txt` omits it.
- **No Appwrite extra, and no publish variable.** A calibration run uploads nothing, so the only
  secret this chain needs is `/root/.netrc`. See ground rule 5 — and note that the *sibling*
  script legitimately installs nine publish variables, so "copy what `pod_run_model.sh` does" is
  exactly how this gets broken. Two tests enforce it.
- **`TMPDIR` is pointed at `/workspace/tmp`.** The prediction scratch honours `TMPDIR` and
  otherwise lands in `/tmp` — the **container** disk — while the disk floor measures
  `/workspace`, the **volume**. A floor that measures a filesystem the workload does not use
  cannot fire. The runner exports it; if you run `main.py` by hand, export it yourself.
- **Two models are refused:** `little_talks` and `mister_bluesky`. They ask for 100 MC-dropout
  samples, and the engine materialises all 13 rolling origins as Python lists before releasing
  any of them — **measured at ~303 GB of RAM**, against a selection rule of ≥ 50 GB. Nine models
  at one sample need ~11 GB. **#536** decides what those two run at; until it lands the preflight
  refuses them by name rather than discovering it after training.
- **`libpq-dev` is still installed**, for the same reason as Phase 2.2 — the dependency chain
  still reaches `viewser`, and `toolz>=0.12.1` is still the last install for the same
  register **C-151** reason. Both apply unchanged here.

### Phase 5 for darts — what comes home

`/workspace/deliver/<model>/` holds:

| | |
|---|---|
| `parquet/` | **13** delivery parquets, one per rolling origin, 2 333 448 rows each |
| `MANIFEST` | runtime, peak scratch, engine version, git sha, region |
| `STATUS` | `OK`, `PREFLIGHT_OK`, or `FAILED:<stage>` |
| `run.log` | the whole transcript |

**There is no `draws/` archive**, and that is correct rather than missing: nine of the eleven are
deterministic, so there is no posterior to compress. It also means **no `q95` variant is
definable for them** — the quantile correction that fixed the HydraNets' ~5× undershoot has no
analogue here. Say so to anyone comparing the two deliveries.

```bash
rsync -a -e "ssh -p <port> -i ~/.ssh/id_ed25519" \
    root@<ip>:/workspace/deliver/<model>/ \
    models/<model>/data/generated/calibration_delivery_<date>/
```

Before trusting it, look at one origin as a picture. The tests prove the arithmetic; they cannot
tell you the field stopped looking like conflict.

**On teardown, if you shared the pod:** `find /tmp /workspace/tmp -maxdepth 1 -name 'pred_frames_*'
-user "$USER" -exec rm -rf {} +`. On 0.2.4 the scratch frees itself at interpreter exit, so this
is a backstop for a process that was killed — not routine housekeeping.

## Phase 6 — Teardown

Confirm `STATUS` is `OK` and the files are on your laptop, then **terminate** the pod — not stop
it. A stopped pod still bills for its volume, at double the running rate.

---

## Known caveats — expected, not defects

- **The console shows ~0% GPU utilisation and looks idle.** The GPU genuinely is idle between
  bursts; this workload is CPU-bound in between. Check lesson progress, not the utilisation graph.
- **`run_integration_tests.sh` will report `TIMEOUT` for every HydraNet.** Its 1800 s default was
  sized when these models trained 40 lessons. At 300 they need ~4 h. That is the budget, not a
  regression — see `docs/CICs/IntegrationTestRunner.md` §0. (That CIC, and the config comments,
  currently say ~7 h from an early bad extrapolation; a correction is pending. The recommended
  `--timeout 30000` is over-provisioned either way and remains safe.)
- **The same applies to the darts models, and `run_integration_tests.sh` is not the way to run
  them on a pod regardless.** They declare `n_epochs: 300`, so the 1800 s default times them out
  too (`--library r2darts2 --timeout 30000` if you do want it locally). On a pod it is the wrong
  tool twice over: it sets no `WANDB_MODE`, so `main.py` calls `wandb.login()` and blocks on a
  prompt nobody is watching — this already killed one run — and it activates a pre-existing
  *named* conda env while a pod builds a `uv` venv. Use `pod_run_darts_calibration.sh`.
- **`pgrep -f <pattern>` matches your own SSH command**, because the pattern appears in its
  command line. Use `ps -eo args | grep -E "[p]attern"` or you will conclude a process is alive
  when it is not.
- **Pasting a multi-line command into the browser web terminal breaks** on a trailing `&&` — the
  shell waits for continuation and appears to hang. One command per line.
- **The in-code memory guard cannot see the container.** It reads the *host's* RAM and will
  approve a run that cannot fit. Do your own arithmetic: the posterior cube is ~3.3 GB at 16
  draws and scales linearly.

---

## Failure modes

| Condition | Behaviour | Recovery |
|---|---|---|
| `.netrc` missing or wrong login | Refuses in preflight, seconds in | Step 2.3; check the login is *yours* |
| `total_lessons` still a throwaway value | Refuses in preflight | Restore to 300, or re-clone |
| `REGION` is not `"land"` | Refuses in preflight | Wrong branch or an edited config |
| Sampling collapses to ~0.1 steps/s | Runs, produces correct output, takes ~8× as long | Terminate; the machine is undersized (Ground rule 1) |
| Pod dies mid-run | Everything on `/workspace` survives; the run does not | Restart the model; the volume persists |
| Fewer than 13 parquets | `STATUS` is `FAILED:collapse` | `run.log` names the origin and the reason |
| SSH refused after a restart | Port changed | Re-read the Connect tab |

The runner never fails silently: every refusal writes `FAILED:<stage>` to `STATUS` and names the
offending path in `run.log`.

---

*First executed 2026-09-28: eight HydraNets, calibration partition, global land, five machines,
about $24. What went wrong and what it taught us is in the post-mortem.*

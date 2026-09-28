# RunPod Run Guide

**Status:** Active
**Owner:** Project maintainers
**Last reviewed:** 2026-09-28
**Tooling status:** `tools/podrun` is **v0.1.0, provisional** — one model family, one partition, no CI. See its `__init__.py`.
**Related:** `reports/postmortem_runpod_first_deployment_2026-09.md` (the *why*, and every number quoted here); `reports/runpod_cost_and_time_note_2026-09.md` (what it costs); runbook #499; spec #505

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
throwaway value, if `REGION` is not `"land"`, or if the converter is absent.

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
| `MANIFEST` | sizes, timestamps, git sha, lesson count |
| `STATUS` | `OK`, or `FAILED:<stage>` |

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

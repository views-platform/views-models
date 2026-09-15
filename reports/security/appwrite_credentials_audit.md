# Appwrite Credentials Audit — where the secrets live, and whether they sprawl

**Date:** 2026-07-27
**Author:** deep read-only investigation (views-models seat), commissioned after the maintainer's concern that credentials were declared in multiple places / possibly un-gitignored / a security risk.
**Method:** read-only greps across `views-models`, `views-postprocessing`, `views-faoapi`, `views-pipeline-core` (+ full git history). **No secret values were read, printed, moved, or committed** — names, paths, patterns, and counts only.

---

> **Status note, added 2026-09-07 when this was committed.** This is a **dated record of the
> 2026-07-27 investigation**, kept because four tracked files cite it. Read §5 as history, not as a
> to-do list — most of it has since been done: `.env.example` exists, `tools/credentials/check_credentials.py`
> and `tests/test_credentials_presence.py` implement recommendation 6, and `views-models/.env` is
> present (its `APPWRITE_ENDPOINT` and `APPWRITE_DATASTORE_PROJECT_ID` were restored 2026-08-26).
> What remains open is recommendation 3, the hardcoded `SOURCE_ENV` path in `views-faoapi`'s
> `bootstrap.sh`, which lives in another repository. Nothing below has been rewritten — a dated audit
> that gets edited to stay current stops being evidence of what was true when it was made.

## Verdict (TL;DR)

1. **The maintainer is not imagining it — the credentials have a real, single canonical home: `views-models/.env`.** The faoapi deployment bootstrap literally reads `^APPWRITE_*` out of it to provision the server (`views-faoapi/deployment/bootstrap.sh:69-70`). They *have* been specified, repeatedly, there.
2. **They do NOT sprawl.** One canonical home; **one consistent env-var vocabulary** across all four repos (15 keys, no competing aliases); the server copy (`.env.faoapi`, chmod 600) is a *filtered derivation* of that one file. This is a coherent design, not a mess.
3. **Zero secrets in git — ever.** No `.env` was tracked or added in any repo across full history; no secret value is pasted into any tracked `.py`/`.ipynb`/`.md`/`.yaml`/`.sh`/`.toml`. `.env` (and the `.env.bak`) are gitignored in all repos.
4. **The real problems are small and fixable** (details in §5):
   - `views-models/.env` is **absent from this particular checkout** → the entire "mystery" this session hit.
   - There is **no `.env.example`** documenting the canonical keys in `views-models` (or pipeline-core / postprocessing) → a fresh checkout has no signpost, so each session rediscovers the keys by reading code. **This is the root cause of the recurring re-ask.**
   - `bootstrap.sh` **hardcodes `/home/sonja/.../views-models/.env`** as the source → bus-factor / portability.

---

## 1. The canonical credential set (15 keys — one consistent vocabulary)

| Concept | Env var | Secret? |
|---|---|---|
| Appwrite server | `APPWRITE_ENDPOINT` | endpoint (semi) |
| Auth | `APPWRITE_DATASTORE_PROJECT_ID`, `APPWRITE_DATASTORE_API_KEY` | **yes — the API key** |
| Shelf bucket (`production_forecasts`) | `APPWRITE_PROD_FORECASTS_BUCKET_ID` / `_BUCKET_NAME` / `_COLLECTION_ID` / `_COLLECTION_NAME` | identifiers |
| Metadata DB | `APPWRITE_METADATA_DATABASE_ID` / `_DATABASE_NAME` | identifiers |
| FAO bucket (`unfao_bucket`) | `APPWRITE_UNFAO_BUCKET_ID` / `_BUCKET_NAME` / `_COLLECTION_ID` / `_COLLECTION_NAME` | identifiers |
| FAO curation lists | `APPWRITE_UNFAO_APPROVED_FILE_IDS` / `_QUARANTINED_FILE_IDS` | identifiers |

Only **`APPWRITE_DATASTORE_API_KEY`** (with endpoint + project id) is a genuine secret; the rest are non-secret identifiers.

## 2. Consumers by component

- **Producer — `rusty_bucket --prediction_store`** (pipeline-core `configs/prediction_store.py` `_ENV_MAP`): endpoint + datastore project/key + `PROD_FORECASTS_*` + `METADATA_*` (9 keys). Writes the shelf.
- **Postprocessor — `un_fao`** (views-postprocessing `unfao/managers/unfao.py`): those 9 (reads the shelf) **+** `APPWRITE_UNFAO_*` (writes `unfao_bucket`).
- **Serving API — faoapi** (`src/views_faoapi/managers/api.py`, declared set `_REQUIRED_APPWRITE_ENV_VARS`): endpoint + project + `METADATA_*` + `UNFAO_*` (reads `unfao_bucket`).
- **Liveness** (views-models `tools/liveness/appwrite_api.py`): endpoint + datastore project/key (to observe the shelf). `tools/liveness/appwrite_store.py` uses a **hardcoded constant** `APPWRITE_BUCKET_ID = "production_forecasts"` — a bucket *name literal*, **not** a credential env var.

## 3. Where creds live, per execution context

- **Laptop model/producer run** (`rusty_bucket`): expects them in the process env, loaded from **`views-models/.env`** (repo root; `load_dotenv` on that path). **Absent in this checkout.**
- **Laptop postprocessor run** (`un_fao`): same — `views-models/.env`.
- **Deployed faoapi (Hetzner CPX52):** systemd `EnvironmentFile=/home/views-faoapi-deploy/.env.faoapi` (`deployment/views-faoapi.service:23`), which `bootstrap.sh` creates by `grep '^APPWRITE_' <SOURCE_ENV>` where `SOURCE_ENV` defaults to **`/home/sonja/views-platform/views-models/.env`**.

So the **source of truth is `views-models/.env`**; everything else is derived from or points at it.

## 4. Security hygiene

- **Git history (all 4 repos, `--all --full-history`):** no `.env` ever added. ✅
- **Tracked files:** no secret-shaped value (`API_KEY|TOKEN|PASSWORD|SECRET = "<16+ chars>"`) in any `.py`/`.ipynb`/`.md`/`.yaml`/`.sh`/`.toml`. ✅
- **gitignore:** `.env` ignored in all repos; the session-made `.env.bak-20260720` (faoapi) is ignored. ✅
- **Only one real `.env` on the machine:** `views-faoapi/.env` (+ its `.bak`, `.example`).

## 5. The real (small) problems + recommended remediation

1. **No `.env.example` for the producer/postprocessor side.** Add `views-models/.env.example` listing the 15 canonical keys with comments and empty values (and note the ~3 that are true secrets). This is the single highest-value fix — it turns "read the code to discover the keys" into "copy the example." Consider the same for pipeline-core.
2. **`views-models/.env` absent here.** Restore/populate it once from the example (maintainer supplies the 3 secret values + the identifiers). Then every laptop run + the bootstrap work.
3. **`bootstrap.sh` hardcodes `/home/sonja/...`.** Parametrize `SOURCE_ENV` (it already supports an override; make the default not a personal home, and document it).
4. **Docs:** pipeline-core has no credentials doc; faoapi documents via CICs. Add a short platform credentials note pointing at the `.env.example` and this audit.
5. **Test-only auth path:** `APPWRITE_SESSION_EMAIL` / `_PASSWORD` are used **only** in `views-faoapi/tests/test_integration_appwrite_write.py` — not production. Document as test-only (or retire if the test is dead).
6. **Self-diagnosis:** add a fail-loud cred-presence check (extend `PredictionStoreConfig.from_environment`'s pattern, and/or a `tools/liveness` credential surface) so a fresh clone reports "missing X" instead of failing deep in a run.

## 6. Corrections to the preliminary (alarmist) read made mid-investigation

The rigorous pass **downgraded** three of my own earlier claims — recorded here for honesty:
- "A bare `APPWRITE_KEY` is a third auth style" — **false**; `APPWRITE_KEY` is not an env var (regex noise).
- "An email/password auth path competes in production" — **false**; it's test-only.
- "`APPWRITE_BUCKET_ID` is a rival credential name" — **false**; it's a hardcoded bucket-name constant.

The env-var vocabulary is, in fact, **consistent**. The genuine issue is *absence of documentation/a durable example*, not *sprawl*.

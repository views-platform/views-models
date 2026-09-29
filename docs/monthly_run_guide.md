# Monthly Run Guide

This document explains how to set up your environment, run and post-process the production models each month.

---

## Prerequisites / First-Time Set-Up

### macOS Users

If you are on macOS, you will need to install `libomp`:

```bash
brew install libomp
```

### Weights & Biases (W&B)

We use [Weights & Biases](https://wandb.ai/site) for experiment tracking.

1. Create a free account at [wandb.ai](https://wandb.ai/site).
2. Ask your colleagues to join the team.
3. You will need your API key later for authentication.

### Clone the Repository 

Clone the repository:

```bash
git clone https://github.com/views-platform/views-models
```

### Set the Machine Up — `./bootstrap.sh`

```bash
cd views-models
./bootstrap.sh
```

No arguments, no companion document. It asks you for **one secret** (the Appwrite
datastore API key, which it never echoes) and **zero addresses** — those come from the
platform coordinate registry, because retyping an address that is already declared
somewhere is how the two copies drift.

It is idempotent, so re-running it is safe, and it is exercised in CI against a fixture
registry, because a setup path verified once on one laptop rots exactly like the prose it
replaces. See ADR-018.

It does **not** create conda environments. Each `run.sh` still builds its own.

> The macOS `libomp` step above is also handled by `bootstrap.sh`. The ~130 per-model
> `run.sh` scripts still carry their own copy of that block; removing them is tracked
> separately (#310).

---

## Run the Models

### Step 1. Pull the Latest Version of the Repository

Navigate to the views-models directory and pull the latest version of it:

```bash
cd views-models
git pull
```

### Step 2. Navigate to the Ensembles Directory

```bash
cd views-models/ensembles
```

### Step 3. Run the CM Ensemble (`pink_ponyclub`)

This must be run **first** to allow reconciliation later:

```bash
./pink_ponyclub/run.sh -m -o [EndOfHistory]
```

When done, sanity-check the results in the auto-generated forecast report (html), available in the sub-dir `pink_ponyclub/reports`.

* If this is your **first time using W&B**, you will be prompted to log in.
* Copy your API key from your W&B profile and paste it into the terminal when asked.

### Step 4: Run the PGM Ensemble (`skinny_love`)

Once the CM ensemble finishes, run the PGM ensemble:

```bash
./skinny_love/run.sh -m -o [EndOfHistory]
```

When done, sanity-check the results in the auto-generated forecast report (html), available in the sub-dir `skinny_love/reports`.

---

## Update Codebooks and Pull to the VIEWS API

### Step 1. Update Codebooks
Codebooks for our public forecasts are available in the [views_outreach repository](https://github.com/prio-data/views_outreach/tree/main/codebooks/master-codebooks/api).


#### Update the forecasts codebook
Check if any changes need to be made and implement if needed. Unless new indicators have been added or renamed, changes are usually not needed.

* This codebook is delivered by the VIEWS API, and used to populate the [VIEWS Dashboard](https://data.viewsforecasting.org). Clear and correct descriptions are key.


#### Update the predictors codebook
This codebook contains descriptions of the non-transformed UCDP features informing our production models. They are made available in a dedicated dataset via the VIEWS API, which is used in multiple instances across the VIEWS Dashboard. The predictors codebook, in turn, is used to populate text fields in the VIEWS Dashboard. Clear and correct descriptions are key.

The codebook must be updated every month, specially:
 - References to which UCDP GED and Candidate datasets our predictors dataset is derived from;
 - Applicable citation for said datasets, following recommendations on the UCDP website;
 - When the data was ingested into our system (for replicability).

Always check if other fields need to be updated as well.


### Step 2. Pull Forecasts and Predictor Data to the VIEWS API

[TODO – Jim currently implements this step]


### Step 3. Update the VIEWS_API Wiki Page

Update the [Available Datasets](https://github.com/prio-data/views_api/wiki/Available-datasets) page in the wiki section with the new + next planned data releases. This page is scraped by HDX, and possibly other users, as part of their monthly updates of our data. **This step must be implemented and the current structure maintained!**

### Step 4. Create and Upload a CSV Version of the Forecasts for the VIEWS Website

#### Step 4.1. Run the download script to create a CSV version of the latest forecasts from the API. 

The target outcomes are re-named in the API. Please use the [dedicated script](https://github.com/prio-data/views_outreach/blob/main/monthly_run/download_API_data_for_website.ipynb) to fetch the API version of the latest forecasts to ensure consistency for users.
   
#### Step 4.2. Upload to the VIEWS Website

Upload the csv files (one for CM, one for PGM) to the `Media` folder for our website on Wordpress. 

Navigate to the Data page under `Pages` in the left sidebar menu, and use the page editor that appears to update the links to each CSV file; as well as all text describing the period covered by the dataset, the `EndOfHistory`, and the dataset name (as listed in the VIEWS API).

___

## Final Checks

1. The day after the forecasts and predictor datasets were pulled to the VIEWS API, check that the latest data have been imported to the VIEWS dashbaord and displays correctly. The dashboard fetches new data at 3 am every morning. Check under the `Predicted conflict` and `Recorded fatalities` data categories in the bottom menu, respectively.

2. In the beginning of the following month, check that the latest forecasts have been uploaded to [VIEWS' HDX page](https://data.humdata.org/organization/views) (the import is automated by HDX). If there's an issue, contact the HDX staff for assistance (Angelica has their contact info).

___
## Notes

* Always ensure the **CM model finishes before running PGM model**. The PGM ensemble (`skinny_love`) reconciles its grid forecast to the CM totals from `pink_ponyclub`, so the CM forecast must already exist.
* **`monthly_run.sh` runs all four production ensembles plus `un_fao` in order.** Two things
  it does that are easy to miss: it **checks the Appwrite write path before starting**, and
  aborts if the key is dead — four ensembles is hours of compute whose only product is an
  upload, and an expired key is silent on that path (#302). And it writes a `pip freeze` per
  environment into `reports/env_snapshots/`, **which you should commit with the run** — that
  is the only record of which package versions produced a given forecast (#328, C-117).
* ⚠️ **Reconciliation currently requires an unreleased pipeline-core.** `skinny_love` and
  `white_mustang` import `reconciliation/` at module level, and that layer needs
  `views_pipeline_core.domain.reconciliation_port`, which exists only from pipeline-core
  **3.0.0** — unpublished. Every ensemble's `requirements.txt` declares
  `views-pipeline-core>=2.0.1,<3.0.0`, so a clean install per the declared requirements
  produces an environment where `skinny_love` fails at import. It works today only because
  `envs/views_ensemble` holds an editable install pointing at a local 3.0.0 checkout. See
  views-models#329.
* **Reconciliation is wired automatically.** Reconciling ensembles (`reconciliation: "pgm_cm_point"` in `config_meta`) inject a reconciler at their composition root (`main.py`) via the `reconciliation/` layer — no manual step. The geography mapping is sourced from viewser (VIEWS `country_id`, parity-preserving). See `docs/CICs/ReconciliationWiring.md` and ADR-014. `white_mustang`→`cruel_summer` is also wired but runs on demand (not in `monthly_run.sh`).
* The `-o [EndOfHistory]` argument specifies the last available month for data; replace `[EndOfHistory]` with the appropriate **VIEWS month** as needed.
* If you encounter issues with W&B authentication, you can manually log in using:

  ```bash
  wandb login
  ```

Happy forecasting! 🚀

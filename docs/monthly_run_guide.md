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

* Always ensure the **CM model finishes before running PGM model**.
* The `-o [EndOfHistory]` argument specifies the last available month for data; replace `[EndOfHistory]` with the appropriate **VIEWS month** as needed.
* If you encounter issues with W&B authentication, you can manually log in using:

  ```bash
  wandb login
  ```

Happy forecasting! 🚀

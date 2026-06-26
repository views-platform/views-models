# First-Time Setup & Monthly Run Guide

This document explains how to set up your environment and run the monthly models.

---

## 1. Prerequisites

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

---

## 2. Clone the Repository

Clone the repository and navigate to the ensembles directory:

```bash
git clone https://github.com/views-platform/views-models
cd views-models/ensembles
```

---

## 3. Running the Models

### Step 1: Run the CM Model

This must be run **first** to allow reconciliation later:

```bash
./pink_ponyclub/run.sh -m -o [EndOfHistory]
```

* If this is your **first time using W&B**, you will be prompted to log in.
* Copy your API key from your W&B profile and paste it into the terminal when asked.

### Step 2: Run the Ensemble

Once the CM model finishes, run the PGM model:

```bash
./skinny_love/run.sh -m -o [EndOfHistory]
```

---

## 4. Notes

* Always ensure the **CM model finishes before running PGM model**. The PGM ensemble (`skinny_love`) reconciles its grid forecast to the CM totals from `pink_ponyclub`, so the CM forecast must already exist.
* **Reconciliation is wired automatically.** Reconciling ensembles (`reconciliation: "pgm_cm_point"` in `config_meta`) inject a reconciler at their composition root (`main.py`) via the `reconciliation/` layer — no manual step. The geography mapping is sourced from viewser (VIEWS `country_id`, parity-preserving). See `docs/CICs/ReconciliationWiring.md` and ADR-014. `white_mustang`→`cruel_summer` is also wired but runs on demand (not in `monthly_run.sh`).
* The `-o [EndOfHistory]` argument specifies the last available month for data; replace `[EndOfHistory]` with the appropriate **VIEWS month** as needed.
* If you encounter issues with W&B authentication, you can manually log in using:

  ```bash
  wandb login
  ```

---

## 5. Monthly Workflow

For each monthly run:

1. Pull the latest updates:

   ```bash
   git pull
   ```
2. Run the **CM model** with `pink_ponyclub`.
3. After it completes, run the **PGM model** with `skinny_love`.

---

Happy forecasting! 🚀

# PR Title
refactor: simplify model imports, align sweep configs, and sync new_rules

# PR Description
This PR focuses on standardizing model configurations, simplifying core library imports, and ensuring architectural alignment across the neural model suite.

**Key Changes:**
- **Import Simplification:** Refactored `main.py` across all models to use the simplified `from views_r2darts2 import DartsForecastingModelManager, apply_nbeats_patch` syntax.
- **Sweep Configuration Alignment:** Converted the sweep settings for `fancy_feline`, `party_princess`, `bouncy_organ`, `adolecent_slob`, and `hot_stream` from Bayesian optimization to a targeted 2-run grid search over `random_state` [1, 2] to improve testing stability and reproducibility.
- **`new_rules` Synchronization:** Fully aligned the hyperparameters and sweep configuration of `new_rules` with the `novel_heuristics` reference architecture.
- **N-BEATS Patch Update:** Updated the N-BEATS dropout patch implementation to use the new `apply_nbeats_patch` utility.
- **Metadata Maintenance:** Updated calibration and data-fetch logs to reflect the latest model training states.

**Verification:**
- Verified architecture via local execution.
- Performed adversarial testing on config completeness using temporary `adversarial_test[one-six]` scaffolds.
- Successfully merged latest `main` to ensure zero conflicts.

🖖

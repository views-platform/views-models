# Debugging Plan: Systematic Performance Discrepancy between W&B Sweep and Single Runs

## 1. Problem Framing
*   **Precisely restate the failure mode:**
    The core issue is a *systematic performance discrepancy* where models trained using a Weights & Biases (W&B) sweep (configured by `models/preliminary_directives/configs/config_sweep.py`) consistently achieve *better performance* than models trained as single runs (configured by `models/preliminary_directives/configs/config_hyperparameters.py`), despite the user's belief that both configurations specify *identical hyperparameters* for the N-BEATS model. The discrepancy is significant and not attributable to normal stochastic variation.

*   **Clarify what “should be equivalent” actually means in operational terms:**
    For the purpose of this debugging plan, "should be equivalent" means that, given identical code, data, and hardware, running the N-BEATS model with hyperparameters derived from `config_hyperparameters.py` (which explicitly sets `random_state = 1`) should produce results that are *statistically indistinguishable* from a W&B sweep run that uses the *exact same set of hyperparameters* (i.e., the single point in the grid search defined by `config_sweep.py` where all 'values' lists have only one element, and specifically `random_state` is set to 1). This operational equivalence implies:
    *   **Configuration Equivalence:** All effective hyperparameters and implicit settings passed to the model and trainer are precisely the same.
    *   **Execution Equivalence:** The underlying code paths executed, library versions used, and data handling processes are identical.
    *   **Stochastic Equivalence:** All sources of randomness (model initialization, data shuffling, dropout masks, etc.) are controlled by identical seeds, leading to bit-for-bit identical results up to floating-point precision, assuming identical hardware and software stack.
    *   **Performance Equivalence:** The final evaluation metrics (e.g., `time_series_wise_msle_mean_sb` as specified in `config_sweep.py`) on an identical, held-out test set are statistically equivalent, meaning any observed differences are within expected noise levels for the given model and task.

## 2. Equivalence Axes

*   **Hyperparameter Surface:**
    *   **Description:** The explicit hyperparameter values defined in `config_hyperparameters.py` versus those injected by the W&B sweep agent based on `config_sweep.py`. This includes simple values (e.g., `lr`, `dropout`), lists (`steps`), and strings (`activation`).
    *   **Specific Concern:** While `config_hyperparameters.py` contains a fixed value for each hyperparameter, `config_sweep.py` uses `'values': [...]` lists. Although these lists currently contain single values corresponding to `config_hyperparameters.py`, there's a risk of subtle differences in how W&B extracts and casts these values, or if any additional parameters are introduced by W&B that are not explicitly in `config_hyperparameters.py`.

*   **Default Values and Implicit Parameters:**
    *   **Description:** Any parameters not explicitly defined in either `config_hyperparameters.py` or `config_sweep.py` but which are used by the underlying libraries (e.g., Darts, PyTorch Lightning) or the model itself.
    *   **Specific Concern:** `config_hyperparameters.py` explicitly mentions `"batch_norm": False, # not part of sweep; unchanged default"`. This implies an implicit parameter that needs to be consistent. Are there other such implicit parameters (e.g., related to `torch.backends.cudnn` settings, `torch.set_float32_matmul_precision` in PyTorch Lightning) that might be set differently or left to system defaults that vary?

*   **Code Paths:**
    *   **Description:** The exact sequence of code executed. This could involve conditional logic (`if` statements), different function calls, or branching behavior based on environment variables or configuration flags.
    *   **Specific Concern:** Is there any code that behaves differently depending on whether it's detected to be running within a W&B sweep (e.g., checking for `WANDB_SWEEP_ID` environment variable) versus a standalone script? Are there any differences in how the model instantiation or trainer setup handles parameters from `config_hyperparameters.py` versus those provided by the W&B agent?

*   **Randomness and Seeding:**
    *   **Description:** All sources of non-determinism, including model weight initialization, data shuffling, data augmentation, dropout masks, and any stochastic operations within PyTorch, NumPy, or Python's standard library.
    *   **Specific Concern:** `config_hyperparameters.py` explicitly sets `"random_state": 1`. `config_sweep.py` also specifies `'random_state': {'values': [1]}'`. However, is this single `random_state` sufficient to seed *all* sources of randomness (Python, NumPy, PyTorch CPU, PyTorch CUDA)? Is this seed applied *consistently and exhaustively* at the very beginning of both run types?

*   **Data Handling and Splits:**
    *   **Description:** How the input data is loaded, preprocessed, scaled, transformed, and split into training, validation, and testing sets.
    *   **Specific Concern:** Are the exact same data files used? Are the data loading functions (e.g., from Darts `TimeSeries` objects) called with identical parameters? Is the random seed for data splitting (if applicable) controlled by the `random_state` parameter or another mechanism? Do `MinMaxScaler` instances (specified for `feature_scaler` and `target_scaler`) get initialized with the same state?

*   **Trainer / Optimizer / Scheduler Lifecycle:**
    *   **Description:** The initialization, state management, and update procedures for the PyTorch Lightning Trainer, Optimizer, and Learning Rate Scheduler.
    *   **Specific Concern:** Are the `Trainer` arguments (`accelerator`, `logger`, `gradient_clip_val`, `callbacks`) identical? Is the `optimizer_kwargs` (e.g., `lr`, `weight_decay`) set correctly? Does `ReduceLROnPlateau` (`lr_scheduler_cls`) behave identically with its `lr_scheduler_kwargs` (`factor`, `min_lr`, `patience`, `monitor`)? Are there any implicit differences in how these components are initialized or interact when run via sweep vs. directly?

*   **Logging and Callbacks:**
    *   **Description:** The configuration and behavior of `WandbLogger`, `EarlyStopping`, `LearningRateMonitor`, and any other callbacks used by PyTorch Lightning.
    *   **Specific Concern:** Are the `WandbLogger` arguments identical? Does `EarlyStopping` (`early_stopping_patience`, `early_stopping_min_delta`) monitor the exact same metric and behave identically in both scenarios? Are there any differences in how W&B's `log_model="all"` interacts?

*   **Hardware and Precision:**
    *   **Description:** The physical hardware (GPU model, CPU architecture), system software (OS version, CUDA version, driver version), and numerical precision settings (e.g., `torch.set_float32_matmul_precision`, mixed precision).
    *   **Specific Concern:** Are sweep runs and single runs guaranteed to execute on the *exact same hardware configuration*? Are environment variables like `CUDA_VISIBLE_DEVICES` or `PYTORCH_CUDA_ALLOC_CONF` set consistently? Are there any differences in default precision settings (e.g., `torch.float32` vs. `torch.float16`) that might be implicitly set?

*   **W&B-specific Behavior:**
    *   **Description:** Any specific mechanisms or side effects introduced by the W&B client library or sweep agent that might affect the training process.
    *   **Specific Concern:** Does the W&B sweep agent perform any implicit actions (e.g., setting environment variables, modifying Python's global state) that are not replicated in a single run? Does W&B's parameter injection inadvertently change parameter types or introduce subtle variations (e.g., `float` vs. `np.float32`)?

## 3. Hypothesis List

*   **H1: Effective Configuration Mismatch.**
    The hyperparameters effectively applied during model instantiation and training are not identical between the sweep and single run. This could be due to:
    *   H1a: Subtle differences in how W&B parses and injects parameters (e.g., type casting, handling of `None` or lists) compared to direct dictionary access.
    *   H1b: Implicit default values for non-explicitly defined parameters (e.g., `batch_norm` which is explicitly `False` in `config_hyperparameters.py` but absent in `config_sweep.py`) are handled differently or override each other.

*   **H2: Incomplete Randomness Control.**
    Despite `random_state = 1` being specified in both configs, not all sources of randomness are exhaustively and consistently seeded across the entire training pipeline (Python, NumPy, PyTorch CPU, PyTorch CUDA) in both run types. This allows the sweep agent to explore a wider (potentially "better") range of initializations or data permutations.

*   **H3: Data Pipeline Discrepancy.**
    The data loading, preprocessing, or splitting process is not strictly identical or deterministic, resulting in subtly different training or validation data being presented to the model in sweep runs versus single runs. This could include differences in `MinMaxScaler` initialization or data shuffling.

*   **H4: Trainer/Optimizer/Scheduler Lifecycle Differences.**
    The PyTorch Lightning `Trainer`, Optimizer, or Learning Rate Scheduler (specifically `ReduceLROnPlateau`) are initialized or behave differently due to an implicit configuration or interaction unique to the W&B sweep environment. This might affect the actual learning schedule or convergence path.

*   **H5: Logging and Callback Side-Effects.**
    W&B-specific logging (`WandbLogger`) or callback behavior (e.g., `EarlyStopping` monitoring or `LearningRateMonitor`) introduces subtle, performance-influencing side effects or interpretations that are either unique to the sweep environment or configured differently. This includes how `log_model="all"` might affect the model's state.

*   **H6: Execution Environment Discrepancy.**
    There are underlying differences in the execution environment (e.g., Python interpreter version, critical library versions beyond what's explicitly managed, environment variables, CUDA/GPU driver differences, or even CPU microarchitecture) between how the single run is launched and how the W&B sweep agent executes a run.

*   **H7: Resource Allocation Imbalance.**
    The resources allocated to sweep runs (e.g., GPU memory, CPU cores, I/O bandwidth) are systematically different from single runs, allowing sweep runs to train more efficiently or avoid resource contention, leading to better performance.

## 4. Verification Chain

*   **H1: Effective Configuration Mismatch.**
    *   **H1a: W&B parsing/injection differences.**
        *   **Inspect:** The final, effective `hparams` dictionary *inside the training script* after all configuration loading and W&B initialization, for both a single run (using `config_hyperparameters.py`) and a sweep run (using `config_sweep.py` with `random_state=1`). Pay close attention to data types and nested structures.
        *   **How:**
            1.  **Single Run:** Instrument the Python script (e.g., `main.py` for `preliminary_directives`) to print `pprint.pformat(trainer.hparams)` or the equivalent final config dictionary *after* the model and trainer are fully instantiated but *before* training starts.
            2.  **Sweep Run:** Similarly instrument the script to print `pprint.pformat(trainer.hparams)` *after* W&B's `wandb.init()` and `wandb.config` injection but *before* training starts.
            3.  Log these outputs to files (`single_run_hparams.txt`, `sweep_run_hparams.txt`) or directly to W&B (if possible for the single run).
            4.  Perform a line-by-line `diff -u single_run_hparams.txt sweep_run_hparams.txt`.
        *   **Confirm:** The `diff` output shows any structural or value differences in hyperparameters (e.g., `dropout: 0.3` vs `dropout: 0.30000000000000004`, `None` vs `null`).
        *   **Falsify:** The `diff` output shows no differences.

    *   **H1b: Implicit default values.**
        *   **Inspect:** The values of any parameters in PyTorch Lightning `Trainer` or Darts `NBEATSModel` constructors that are *not* explicitly defined in `config_hyperparameters.py` or `config_sweep.py`. For example, is `batch_norm` (explicitly `False` in `config_hyperparameters.py`) being correctly picked up? Are other defaults like `torch.backends.cudnn.benchmark` consistently set or left to system defaults that vary?
        *   **How:**
            1.  **Read Docs:** Consult PyTorch Lightning and Darts NBEATSModel documentation for default values of all constructor arguments.
            2.  **Instrument:** Log the `kwargs` dictionaries immediately before passing them to the `NBEATSModel` and `Trainer` constructors for both run types.
            3.  Compare these `kwargs` against documentation defaults and against each other.
        *   **Confirm:** A parameter, not explicitly set, takes a different effective value between run types, or a parameter explicitly set in `config_hyperparameters.py` (like `batch_norm`) is not correctly propagated in the sweep run due to its absence in `config_sweep.py`.
        *   **Falsify:** All implicit parameters and explicitly propagated parameters are identical in their effective values.

*   **H2: Incomplete Randomness Control.**
    *   **Inspect:** All relevant random seeds throughout the system.
    *   **How:**
        1.  **Instrument Script:** At the very beginning of the training script, and within any data loading/augmentation functions, print and log the current state of:
            *   Python `random.getstate()`
            *   NumPy `np.random.get_state()`
            *   PyTorch CPU `torch.get_rng_state()`
            *   PyTorch CUDA `torch.cuda.get_rng_state()` (if GPU is available).
            2.  Ensure that `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` are explicitly set at the start for both run types to eliminate non-determinism from CuDNN.
            3.  Run both a single and sweep run (with `random_state=1` in both).
            4.  Compare the logged random states at each checkpoint.
        *   **Confirm:** Any of the logged random states (Python, NumPy, PyTorch CPU/CUDA) differ between the single and sweep run at corresponding execution points, *or* a W&B run consistently shows better performance even with identical global seeds applied exhaustively, implying an *unseeded* source of randomness.
        *   **Falsify:** All logged random states are identical at corresponding execution points, and setting `cudnn.deterministic=True` and `cudnn.benchmark=False` doesn't eliminate the performance gap.

*   **H3: Data Pipeline Discrepancy.**
    *   **Inspect:** The exact data presented to the model.
    *   **How:**
        1.  **Instrumentation:**
            *   Before any data splitting, log the `random_state` used for `train_test_split` or equivalent.
            *   After preprocessing and scaling (e.g., `MinMaxScaler.fit_transform`), but before batching for the DataLoader:
                *   Calculate and log a cryptographic hash (e.g., SHA256) of the raw bytes of the training and validation `TimeSeries` objects or underlying NumPy arrays.
                *   Log the `MinMaxScaler` objects' internal state (e.g., `scaler.min_`, `scaler.scale_`).
        2.  Run both a single and sweep run.
        3.  Compare the logged hashes and scaler states.
        *   **Confirm:** Any data hashes or scaler states differ between the runs.
        *   **Falsify:** All data hashes and scaler states are identical.

*   **H4: Trainer/Optimizer/Scheduler Lifecycle Differences.**
    *   **Inspect:** The internal state and behavior of the `Trainer`, Optimizer, and `ReduceLROnPlateau` scheduler.
    *   **How:**
        1.  **Instrumentation:**
            *   After `Trainer` initialization, log its full configuration (e.g., `trainer.logger.experiment.config` if using W&B for single run, or by inspecting `trainer` object attributes).
            *   At the start of each epoch, log:
                *   The current learning rate (`optimizer.param_groups[0]['lr']`).
                *   The `_reducelronplateau_patience_count` (or equivalent internal state) of the `ReduceLROnPlateau` scheduler.
                *   The `monitor` metric value passed to `EarlyStopping` and `ReduceLROnPlateau`.
            *   Log the total number of epochs run and the reason for stopping.
        2.  Run both a single and sweep run.
        3.  Compare these logged values epoch-by-epoch.
        *   **Confirm:** Learning rates, scheduler internal states, `monitor` metrics, or stopping criteria diverge between runs.
        *   **Falsify:** All lifecycle aspects are identical.

*   **H5: Logging and Callback Side-Effects.**
    *   **Inspect:** W&B logger configuration and any unintended interactions with the model or training process.
    *   **How:**
        1.  **Instrumentation:**
            *   Log the exact `WandbLogger` initialization arguments.
            *   Create a minimal dummy training loop that *only* initializes the `WandbLogger` (with `log_model="all"`) and saves a dummy model after one step, for both sweep and single run.
            *   Compare the size and content of the saved model artifacts.
            *   Carefully review W&B system metrics (e.g., CPU, GPU usage, I/O) for any unusual spikes or patterns only present in sweep runs.
        *   **Confirm:** Significant differences in W&B logger configuration, model artifact size/content after initial save, or system resource usage patterns related to logging/callbacks.
        *   **Falsify:** No discernible differences or side-effects from W&B logging and callbacks.

*   **H6: Execution Environment Discrepancy.**
    *   **Inspect:** The complete software stack.
    *   **How:**
        1.  **Before Running Script (for both run types):**
            *   `python --version`
            *   `conda list` or `pip freeze` (full output).
            *   `nvcc --version` (if CUDA involved).
            *   `cat /proc/cpuinfo` and `nvidia-smi` (if GPU involved).
            *   `echo $PATH`, `echo $PYTHONPATH`, and inspect other relevant environment variables (e.g., WANDB-specific ones).
        2.  **Instrumentation:** Inside the Python script, log `sys.version`, `sys.path`, and `os.environ` (filtered to relevant variables).
        3.  Compare all these outputs meticulously.
        *   **Confirm:** Any differences in Python version, installed packages/versions, CUDA, or relevant environment variables.
        *   **Falsify:** All environment and software stack details are identical.

*   **H7: Resource Allocation Imbalance.**
    *   **Inspect:** CPU, GPU, and memory utilization.
    *   **How:**
        1.  **System Monitoring:** Use OS-level tools (e.g., `htop`, `nvidia-smi -l 1`) to monitor resource usage during single and sweep runs. Capture these metrics over time.
        2.  **Instrumentation:** Within the Python script, log `torch.cuda.memory_allocated()`, `torch.cuda.max_memory_allocated()` at key points (e.g., start of epoch, after batch processing).
        3.  Compare the resource utilization profiles and logged memory statistics.
        *   **Confirm:** Consistent differences in CPU, GPU, or memory usage that could impact training efficiency.
        *   **Falsify:** Resource allocation and utilization profiles are equivalent.

## 5. Minimal Reproduction Strategy

*   **Experiment Design:**
    1.  **Simplest Model & Data:**
        *   Use the N-BEATS model as specified (`num_stacks=2`, `num_blocks=4`, etc.) as this is the focus.
        *   Identify the *smallest possible dataset* that still reliably triggers the performance discrepancy. This might involve reducing the number of time series, their length, or the forecast horizon if possible, while maintaining the statistical significance of the performance gap.
        *   If the current dataset is large, consider using a fixed, small subset of the training and validation data for this reproduction, ensuring *both* single and sweep runs use the identical subset.
    2.  **Fixed Hyperparameters:**
        *   Strictly use the hyperparameters defined in `config_hyperparameters.py`.
        *   For the W&B sweep, configure `config_sweep.py` as a *single-point grid search* where each parameter's `values` list contains only one element, corresponding exactly to the value in `config_hyperparameters.py`. This ensures W&B injects the identical configuration.
    3.  **Strict Global Seeding:**
        *   Implement a comprehensive global seeding function at the *absolute beginning* of the training script that sets seeds for:
            *   Python's `random` module.
            *   NumPy's `random` module.
            *   PyTorch CPU and CUDA (if applicable).
            *   Set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False`.
        *   Use `random_state = 1` for this global seed, consistent with `config_hyperparameters.py`.
    4.  **Minimal W&B Interaction:**
        *   Initially, disable all non-essential W&B features for both run types: no model saving (unless `log_model="all"` is suspected), no custom W&B callbacks beyond the logger itself. If `log_model="all"` is left enabled, ensure it's doing the same thing.
        *   The primary W&B interaction should be `wandb.init()` and `wandb.config` for parameter injection.
    5.  **Comparative Runs:**
        *   **Run 1 (Baseline Single):** Execute the Python training script directly, loading hyperparameters from `config_hyperparameters.py`. This is the "underperforming" baseline.
        *   **Run 2 (Sweep Single Point):** Execute the Python training script via the W&B sweep agent, using `config_sweep.py` configured as a single-point grid search matching Run 1's hyperparameters. This is the "overperforming" run.
        *   Both runs must use the identical underlying Python script and environment.

*   **Controls and Invariants:**
    *   **Codebase Version:** Use a specific, tagged Git commit for all runs to ensure the exact same code is executed.
    *   **Environment:** Use an identical, explicitly created and locked (e.g., `conda env export > environment.yml`) conda environment for both runs. This must include Python version, PyTorch, PyTorch Lightning, Darts, W&B, and all other dependencies.
    *   **Hardware:** Execute both runs on the *same physical machine* and, if applicable, the *same GPU*. This eliminates hardware variability.
    *   **Global Random Seed:** As detailed above, a single, fixed, and exhaustively applied random seed for all stochastic operations.
    *   **Input Data:** The exact same input data files, read from the same location.

*   **Explicitly State What Must Be Frozen:**
    *   The entire Python codebase (via Git commit).
    *   The Conda environment (via `environment.yml`).
    *   The hardware (CPU, GPU, RAM, OS).
    *   All hyperparameters (explicitly defined in `config_hyperparameters.py` and matched in `config_sweep.py`).
    *   The global random seed (`random_state = 1`) applied at all levels.
    *   The input data.
    *   The number of epochs (e.g., `n_epochs: 300`) and early stopping parameters.

## 6. Instrumentation Plan

The goal of instrumentation is to capture the state of the system at critical junctures for both the single run and the sweep run (configured as a single point, as per the Minimal Reproduction Strategy). All logged outputs should be stored in a structured, easily comparable format (e.g., JSON, YAML, or highly formatted plain text).

*   **At Script Entry (Absolute Beginning):**
    *   **Environment & System Info:**
        *   `sys.argv`: Command-line arguments.
        *   `os.environ`: A filtered dictionary of relevant environment variables (e.g., `WANDB_PROJECT`, `CUDA_VISIBLE_DEVICES`, `PYTHONPATH`).
        *   `platform.platform()`: OS details.
        *   `sys.version`: Python interpreter version.
        *   `pip freeze` or `conda list --export`: Full list of installed packages and their versions (output saved to a file).
        *   `torch.cuda.is_available()`: CUDA availability.
        *   `torch.cuda.device_count()`, `torch.cuda.get_device_name(0)`: GPU device info.
        *   `torch.backends.cudnn.deterministic`, `torch.backends.cudnn.benchmark`: CuDNN settings.
        *   `torch.set_float32_matmul_precision()`: Precision setting.
    *   **Random Seeds:**
        *   `random.getstate()`: Python random state.
        *   `np.random.get_state()`: NumPy random state.
        *   `torch.get_rng_state()`: PyTorch CPU random state.
        *   `torch.cuda.get_rng_state()` (if CUDA available): PyTorch CUDA random state.
        *   The explicit `global_seed` value (e.g., 1) applied.
    *   **Initial Configs:**
        *   The raw dictionary loaded from `config_hyperparameters.py` (for single run).
        *   The raw dictionary loaded from `config_sweep.py` (for sweep run).

*   **After W&B Initialization (`wandb.init()`):**
    *   `wandb.config.as_dict()`: The complete effective W&B configuration dictionary *after* injection. This is critical for H1a.

*   **Before Model and Trainer Instantiation:**
    *   **Model `kwargs`:** The exact dictionary of arguments passed to the `NBEATSModel` constructor.
    *   **Trainer `kwargs`:** The exact dictionary of arguments passed to the PyTorch Lightning `Trainer` constructor.
    *   **Callback `kwargs`:** The exact dictionary of arguments for `EarlyStopping` and `ReduceLROnPlateau` callbacks.

*   **Before DataLoader Instantiation:**
    *   **Data Split Seed:** The `random_state` used for any data splitting.
    *   **Dataset Hashes:** SHA256 hash of the full training and validation `TimeSeries` objects or their underlying data arrays *after* all preprocessing (scaling, transformations, etc.) but *before* batching. This verifies H3.
    *   **Scaler State:** The internal state (`.min_`, `.scale_`) of the `MinMaxScaler` instances for both features and targets after fitting.

*   **During Training (Per Epoch Hook):**
    *   `trainer.current_epoch`: Current epoch number.
    *   `optimizer.param_groups[0]['lr']`: Current learning rate.
    *   `trainer.callback_metrics`: All metrics logged by callbacks (e.g., `train_loss`, `val_loss`, `time_series_wise_msle_mean_sb`).
    *   `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` (if CUDA available): Current GPU memory usage.
    *   `psutil.Process(os.getpid()).memory_info().rss`: Current Python process RAM usage.
    *   **Scheduler State (if accessible):** Internal state of `ReduceLROnPlateau` (e.g., `_last_lr`, `_num_bad_epochs`, `patience_counter` if possible via a custom callback). This is for H4.

*   **At Training Completion (End of Script):**
    *   `trainer.global_step`: Total training steps.
    *   `trainer.current_epoch`: Total epochs completed.
    *   Reason for stopping (e.g., early stopping triggered, max epochs reached).
    *   Final evaluation metrics on the test set.
    *   SHA256 hash of the final model's `state_dict()`.

*   **Comparison Methodology:**
    *   **Automated Diffing:** Use programmatic comparison (e.g., Python `difflib` for text, deep comparison for dictionaries/JSON) for logged configurations, `kwargs`, seeds, and scaler states.
    *   **Time Series Comparison:** Plot and visually inspect epoch-by-epoch learning rates, losses, and metrics.
    *   **System Metrics Comparison:** Compare plots of CPU/GPU utilization and memory over time.
    *   **Artifact Comparison:** If models are saved, compare their file sizes and hashes (though initial comparison via `state_dict()` hash should suffice).

## 7. Decision Tree

This decision tree guides the debugging process, prioritizing checks that are easier to perform or more likely to reveal a significant difference. Each step assumes that previous steps *did not* identify the root cause (i.e., the relevant comparison was "identical").

1.  **Global Environment Equivalence Check (H6 & H4)**
    *   **Action:** Compare the full system environment and installed package lists from the "At Script Entry" instrumentation for both run types (Python version, `conda list`/`pip freeze` output, CUDA version, OS details, relevant environment variables).
    *   **Decision:**
        *   **If DIFFERENCE detected:** Root cause is an environment or software dependency mismatch (H6, H4).
            *   **Stopping Criteria:** Synchronize the environments to be absolutely identical. Re-test. If discrepancy persists, proceed to Step 2 with the now-identical environments.
        *   **If IDENTICAL:** Proceed to Step 2.

2.  **Effective Configuration Equivalence Check (H1a & H1b)**
    *   **Action:** Compare the effective `wandb.config.as_dict()` (or equivalent for single run) and the explicit model/trainer `kwargs` (logged "Before Model and Trainer Instantiation") for both run types. Perform deep comparisons, including data types and nested structures.
    *   **Decision:**
        *   **If DIFFERENCE detected:** Root cause is a configuration mismatch (H1a, H1b).
            *   **Stopping Criteria:** Correct the configuration so that the effective `kwargs` are identical for both runs. Re-test. If discrepancy persists, proceed to Step 3.
        *   **If IDENTICAL:** Proceed to Step 3.

3.  **Comprehensive Randomness Control Check (H2)**
    *   **Action:** Compare all logged random states (Python, NumPy, PyTorch CPU/CUDA) from "At Script Entry" and throughout the run. Confirm `cudnn.deterministic = True` and `cudnn.benchmark = False` are active.
    *   **Decision:**
        *   **If DIFFERENCE detected in random states or CuDNN settings are inconsistent:** Root cause is incomplete randomness control (H2).
            *   **Stopping Criteria:** Ensure all random seeds are identically set and consistently applied at every relevant stochastic operation point (model init, data shuffle, dropout). Confirm CuDNN settings. Re-test. If discrepancy persists, proceed to Step 4.
        *   **If IDENTICAL (all seeds and CuDNN consistent):** Proceed to Step 4.

4.  **Data Pipeline Equivalence Check (H3)**
    *   **Action:** Compare data split seeds and cryptographic hashes of the training and validation datasets (logged "Before DataLoader Instantiation"). Also compare the internal state of `MinMaxScaler` objects.
    *   **Decision:**
        *   **If DIFFERENCE detected in data split seeds, data hashes, or scaler states:** Root cause is a data pipeline discrepancy (H3).
            *   **Stopping Criteria:** Ensure the data loading, preprocessing, and splitting are bit-for-bit identical and deterministic for both runs. Re-test. If discrepancy persists, proceed to Step 5.
        *   **If IDENTICAL:** Proceed to Step 5.

5.  **Trainer/Optimizer/Scheduler Lifecycle Equivalence Check (H4 & H7)**
    *   **Action:** Compare epoch-by-epoch logs of learning rates, scheduler internal states, `monitor` metrics, total epochs run, and early stopping triggers (logged "During Training" and "At Training Completion").
    *   **Decision:**
        *   **If DIFFERENCE detected:** Root cause is a divergence in the training lifecycle (H4, H7). This could be due to subtle differences in how a callback (e.g., `EarlyStopping`, `ReduceLROnPlateau`) interprets metrics or applies updates, or how training is effectively terminated.
            *   **Stopping Criteria:** Pinpoint the exact divergence point in the training process. Modify the setup to ensure identical lifecycle behavior. Re-test. If discrepancy persists, proceed to Step 6.
        *   **If IDENTICAL:** Proceed to Step 6.

6.  **Resource Allocation & W&B Side-Effect Check (H7 & H5)**
    *   **Action:** Compare logged GPU/CPU memory usage and process RAM usage ("During Training"). Review W&B system metrics. Compare any saved model artifacts (if `log_model="all"` is enabled) for initial steps.
    *   **Decision:**
        *   **If DIFFERENCE detected in resource profiles or early model artifacts:** Root cause is either resource allocation imbalance (H7) or a subtle W&B side-effect (H5) affecting the very early stages of training or model saving.
            *   **Stopping Criteria:** Address the resource imbalance (if any) or investigate the specific W&B side-effect. Re-test. If discrepancy persists, proceed to Step 7.
        *   **If IDENTICAL:** Proceed to Step 7.

7.  **Final Fallback (Re-evaluate Problem / Advanced Profiling)**
    *   **Action:** If all previous steps yielded "IDENTICAL" results, and the performance discrepancy still systematically exists, this implies a highly subtle difference not caught by the previous instrumentation.
    *   **Decision:**
        *   **If discrepancy *still* exists:** Enter "Failure Interpretation" (Section 8).
        *   **If discrepancy is resolved at any point:** Root cause identified, proceed to "Expected Outcome" (Section 9).

## 8. Failure Interpretation

*   **If no single, clear difference is found after exhausting the Decision Tree:**
    *   **Implication:** This implies that either:
        1.  The observed performance difference, while "systematic" in its initial presentation, is actually due to an extremely subtle combination of factors that our detailed instrumentation could not individually pinpoint. This might be a highly non-linear interaction effect.
        2.  There is a critical, overlooked axis of equivalence. This could include aspects like compiler optimizations, specific CPU instruction sets being utilized differently, or even micro-architectural power management differences if runs are happening on non-dedicated hardware or in heavily virtualized environments.
        3.  The initial premise of "systematic improvement" itself needs stronger statistical validation. Perhaps the observed difference, while consistent, is within the expected statistical bounds when considering the full variance of the model's performance under truly identical (but still stochastic) conditions.
    *   **Next Steps:**
        *   **Statistical Re-evaluation:** Quantify the performance gap with more rigorous statistical tests (e.g., using paired t-tests or Wilcoxon signed-rank tests over multiple truly identical single and sweep runs) to confirm the statistical significance of the "systematic improvement."
        *   **Blind Comparison:** Perform a "blind" A/B test. Run a larger batch of both single and sweep configurations with the *exact same seeds*, and have an independent party evaluate the models without knowing their origin. This helps rule out confirmation bias.
        *   **Incremental Complexity:** If the minimal reproduction still shows the gap, try to reintroduce complexity (e.g., larger dataset, more epochs, additional callbacks) one by one, observing if and when the gap reappears or widens.
        *   **Advanced Profiling:** Consider using system-level performance profiling tools (e.g., `perf`, `strace`, `valgrind` for CPU/memory; NVIDIA Nsight Systems for GPU) to capture deeper insights into execution flow, syscalls, and resource contention. This requires significant expertise.
        *   **Library Source Review:** Deep dive into the source code of PyTorch Lightning, Darts, and W&B to look for undocumented behaviors or environment-dependent branches that might have been missed.

*   **If multiple small differences are found:**
    *   **Implication:** Multiple minor differences (e.g., slight variation in learning rate due to float precision, minor data shuffling difference, slightly different W&B internal logging calls) can cumulatively lead to a significant performance gap. It's unlikely that the "best" path would consistently be chosen by chance unless one of these small differences had an outsized, directional impact.
    *   **Next Steps:**
        *   **Prioritize Impact:** Analyze the nature of each small difference. Focus on differences related to core training mechanics (e.g., learning rate, data composition, model architecture) as these are most likely to have a directional impact on performance.
        *   **One-Difference-At-A-Time Fix:** In the minimal reproduction setting, systematically *correct* each identified small difference, one at a time, and re-run the comparison. This allows isolating which specific difference (or combination) contributes most to the performance discrepancy.
        *   **Interaction Effects:** If fixing individual small differences does not fully close the gap, consider potential interaction effects between two or more minor discrepancies. This would involve testing combinations of fixes.

## 9. Expected Outcome

The successful execution of this debugging plan should lead to a clear and actionable understanding of the root cause(s) behind the performance discrepancy.

*   **What “root cause identified” looks like:**
    *   A precise, concrete explanation of *why* the W&B sweep runs systematically outperform single runs. This explanation will be directly linked to one or more specific differences identified through the Verification Chain. Examples could include:
        *   "W&B's parameter injection casts the `dropout` hyperparameter to `float32` whereas the single run uses `float64`, leading to subtle numerical differences in the model weights and a more stable training for the sweep runs."
        *   "The `config_sweep.py` implicitly enables a default `GradientAccumulation` strategy in PyTorch Lightning when run via the W&B agent that is not active in the single run, effectively increasing the batch size and improving gradient stability for the sweep runs."
        *   "An overlooked environment variable (`OMP_NUM_THREADS`) is set differently by the W&B sweep agent, leading to different CPU parallelism for data loading or preprocessing, which impacts GPU utilization."
        *   "The random seed for data shuffling is not being correctly propagated in single runs, leading to different training data orderings that are less optimal than the sweep's seeded, more consistent order."
        *   "The `early_stopping_patience` value in the sweep is effectively higher due to a parsing error, allowing sweep runs to train for more epochs and achieve better convergence."
    *   The identified root cause will clearly explain the *direction* of the performance difference (i.e., why sweeps are *better*).

*   **What artifacts or evidence should exist at the end:**
    *   **Detailed Comparison Logs:** Comprehensive, side-by-side textual or structured diffs of all logged instrumentation data (configurations, environment details, random states, data hashes, scaler states, optimizer/scheduler parameters, epoch-by-epoch metrics, resource utilization, etc.) for both the "underperforming" single run and the "overperforming" sweep run from the Minimal Reproduction Strategy. These logs will precisely highlight the identified difference(s).
    *   **Confirmation of Fix (or Mitigation):** If a specific difference was found and a change was applied to address it (e.g., updating a config, adding a seed, modifying an environment variable), the plan should include:
        *   The exact code change or configuration modification made.
        *   Results from new runs demonstrating that applying this fix *closes the performance gap* between the single run and the sweep run, bringing their performance to a statistically indistinguishable level (or reversing the performance order if the sweep behavior was inadvertently "better").
    *   **Minimal Reproduction Results:** Clear and reproducible results from the Minimal Reproduction Strategy (including performance metrics and possibly learning curves) that definitively show the presence of the discrepancy before the fix, and its absence after the fix.
    *   **Recommendations:** Specific recommendations for preventing future occurrences of such discrepancies (e.g., stricter config validation, standardized environment setup, improved logging practices).
### **Prompt for New Gemini Instance**

**Objective:** Add a configurable Learning Rate (LR) Warmup mechanism to our Darts/PyTorch Lightning training pipeline located in the `views-r2darts2` repository.

**Your Role:** You are an expert Python developer specializing in the Darts and PyTorch Lightning frameworks.

**Background & Motivation:**

We are training an N-BEATS model on challenging time series data that is zero-inflated and has heavy tails. A key problem we're facing is significant run-to-run variance, where identical hyperparameter configurations produce noticeably different results.

Our working hypothesis is that the model is highly sensitive to the composition of the initial training batches. The noisy gradients at the start of training are likely pushing the model into different local minima, leading to instability.

To solve this, we want to implement a standard technique: **Learning Rate Warmup**. The goal is to start training with a very low `lr` and linearly increase it to the target `lr` over the first few epochs. This will allow the model's weights to stabilize on an "average" gradient before the optimizer begins taking larger, more confident steps.

**Current Setup:**

*   The core logic is in a class named `DartsForecastingModelManager` within the `views-r2darts2` repository.
*   This manager reads hyperparameters from an external configuration file.
*   The current Learning Rate Scheduler is configured using parameters like `lr_scheduler_patience` and `lr_scheduler_factor`, which points to a `ReduceLROnPlateau` scheduler.
*   The models are trained using a `pytorch_lightning.Trainer`.

**Your Task:**

Modify the `DartsForecastingModelManager` and any related code in the `views-r2darts2` repository to add support for a linear learning rate warmup, which should precede the existing `ReduceLROnPlateau` scheduler.

**Implementation Requirements:**

1.  **Make it Configurable:** The warmup must be controlled by new hyperparameters that we can set in our configuration files. Please add support for:
    *   `warmup_epochs` (int): The number of epochs for the linear warmup. If this parameter is not present or is set to `0`, no warmup should occur.

2.  **Implementation Strategy (Recommendation):**
    *   The most robust way to implement this in modern PyTorch is using `torch.optim.lr_scheduler.SequentialLR`.
    *   You will need to define two separate schedulers:
        1.  A `LinearLR` scheduler for the warmup phase. It should run for `warmup_epochs`, starting from a low learning rate (e.g., `lr / 100`) and ending at the main `lr`.
        2.  Our existing `ReduceLROnPlateau` scheduler for the main training phase that follows the warmup.
    *   You would then wrap these two schedulers in `SequentialLR`, telling it to use the `LinearLR` for the first `warmup_epochs` and then switch to `ReduceLROnPlateau` for the remainder of the training.

3.  **Code Integration:**
    *   You will need to locate the part of `DartsForecastingModelManager` where the optimizer(s) and LR scheduler(s) are created. In a PyTorch Lightning context, this is often within a method called `configure_optimizers`.
    *   Modify this logic to check for the `warmup_epochs` hyperparameter.
    *   If `warmup_epochs > 0`, construct the `SequentialLR` scheduler as described above.
    *   If `warmup_epochs` is `0` or not present, the existing logic for creating the `ReduceLROnPlateau` scheduler should be used as before, ensuring backward compatibility.

**Final Deliverable:**

Please provide the modified Python code for the `views-r2darts2` repository that cleanly implements this configurable LR warmup feature. As part of your response, also show a small example of how the `configure_optimizers` method might look after your changes.

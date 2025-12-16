# Analysis of Configuration Divergence

This note records the differences between the "Best Old Model" (the configuration we are reverting to) and the "Current Config" that was being used before the revert. This is to keep track of recent changes that might be worth re-introducing later.

The "Current Config" was being trained much more cautiously and with heavier regularization.

### Key Hyperparameter Differences

| Hyperparameter            | "Best Old Model" Value | "Current Config" Value | Analysis of Difference                          |
|---------------------------|------------------------|------------------------|-------------------------------------------------|
| `lr`                      | **~0.000587**          | ~0.000310              | Current was ~47% lower.                         |
| `batch_size`              | **8**                  | 2                      | Current used a much smaller batch size.         |
| `dropout`                 | 0.3                    | **0.5**                | Current had significantly more dropout.         |
| `weight_decay`            | ~0.000329              | **~0.00055**           | Current had more L2 regularization.             |
| `false_negative_weight`   | ~3.88                  | **5.0**                | Current penalized misses more heavily.          |
| `false_positive_weight`   | **~1.42**              | 3.0                    | Current penalized false alarms much more.       |
| `output_chunk_shift`      | 0                      | **1**                  | Current had a 1-step forecast gap.              |
| `early_stopping_patience` | 10                     | **20**                 | Current was more patient before stopping.       |
| `lr_scheduler_patience`   | **7**                  | 5                      | Current scheduler acted faster.                 |
| `lr_scheduler_factor`     | **~0.463**             | 0.3                    | Current scheduler's LR reduction was more aggressive. |
| `zero_threshold`          | **~0.129**             | 0.10                   | Current's threshold for defining a "zero" was lower. |


# violet_visitor experiment configs

Durable copies of the **experiment** hyperparameter variants used in the sb-at-T=0 /
hurdle-body investigation (2026-06). Each file is a drop-in `get_hp_config()` (same shape as
`../config_hyperparameters.py`) — they are **reference templates**, not the production config.

**Why these live here:** during the investigation they were kept as `/tmp` scratch files that a
driver would `cp` over `config_hyperparameters.py`, run, then restore the floor. That made them
invisible to git and a crash-loss risk. These committed copies remove that fragility.

The **production / floor** config (`config_hyperparameters.py`) is and remains the **hurdle_nb**
no-coords floor (seed 42). It is intentionally *not* one of these variants.

| file | output_distribution | loss_reg | reg_activation | hurdle_mask_mode | notes |
|------|---------------------|----------|----------------|------------------|-------|
| `per_step_mse.py` | hurdle_shrinkage | mse | (default) | per_step | the original per_step baseline (#66 control) |
| `active_window_relu_mse.py` | hurdle_shrinkage | mse | (default → relu) | active_window | the #66 active_window arm — note: pre-#178 this trained a **dead ReLU** body on rare targets (C-178) |
| `active_window_softplus_mse.py` | hurdle_shrinkage | mse | softplus | active_window | the #74 / current arm — softplus body (C-178 fix). Base for the C-181 cls-mask A/B |

The `torch_seed`/`np_seed` in each file is just a default; experiment drivers set the seed per
run. Loss/eval knobs (pw=2 gate, 8 posterior samples, 40 lessons, `feedback_clamp_log1p` eval-rail)
are as used in the runs. See the hydranet dossier `reports/2026-06-23_body_sweep_dossier/`
(entries 16–17) and risk-register C-178 / C-181 for the findings these produced.

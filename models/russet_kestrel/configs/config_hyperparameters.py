# russet_kestrel is NOT a roster member and NOT a seed replicate. It is one of the TWO
# INTERVENTION ARMS of the weight-decay experiment, declared in
# tests/test_roster_conformance.py's EXPERIMENT_ARMS, and it varies exactly ONE value
# against its control: weight_decay, 0.1 -> 0.0.
#
# ⛔ ITS CONTROL IS bronze_kestrel, NOT amber_kestrel. That is the whole point of the
# pairing. russet_kestrel and bronze_kestrel are BOTH seed 43, so they differ on
# `weight_decay` ALONE. Declaring this arm against amber_kestrel instead would make it
# differ on THREE keys -- weight_decay, torch_seed and np_seed -- which
# tests/test_roster_conformance.py:66-70 names as the confounded case: "NOT a one-variable
# pair: the seeds differ too, so any difference between the two runs is confounded with
# seed." The seed match is not a coincidence to be tidied; it is the design.
#
# WHAT IT TESTS. weight_decay 0.1 is TEN TIMES the AdamW default and it reaches the three
# gate heads: the exemption mechanism exists (views-hydranet train/training_engine.py:113,136
# add weight_decay 0.0 groups for sigma_params and log_var_params) and the heads were never
# added to it. The heads also carry biases (HydraBNrecurrentUnet_06_LSTM4.py:251,289,339 --
# no bias=False, unlike the encoder at :201,:207,:212), so decay shrinks both the weight
# (-> slope) and the bias (-> level). Removing it entirely asks what that was buying.
#
# ⛔ THIS IS NOT A CAUSAL TEST AND MUST NOT BE READ AS ONE. The pre-registration's F12 fires
# on any sentence saying weight_decay is or is not the cause of the gate's under-dispersion.
# The programme's honest ledger is derivations 10 right / 1 wrong, causal narratives
# 0 right / 8 wrong. The primary output of these five pods is sigma-hat, not a mechanism.
#
# ⛔ 0.0 IS REACHABLE WITH NO LIBRARY CHANGE, and that was checked rather than assumed:
# views-hydranet utils/config_initializer.py:150 is Field(..., ge=0.0) -- `ge`, not `gt`,
# unlike learning_rate on the line directly above -- and the value passes straight to AdamW.
# Nothing coerces it, floors it or rejects it.
#
# ⚠️ ACCEPTED RISK, STATED BEFORE THE RUN. Nothing in the record says this architecture
# trains with regularisation removed entirely. Whether it does is epic output M2. If this
# run diverges, NaNs, or trips the C-184 seed-bimodal gate, it is VOID -- NOT NEGATIVE --
# and the readout must state how many of the two zero-decay runs survived BEFORE any slope
# is read. A void arm is an absent measurement, never evidence of no effect.
#
# The configuration below is bronze_kestrel's, which is amber_kestrel's, copied unchanged
# apart from weight_decay. Its reasoning is inherited verbatim -- do not "tidy" it.

import os
from pathlib import Path

_MODEL_NAME = Path(__file__).resolve().parent.parent.name
_POD_DELIVER_ROOT = Path("/workspace/deliver")


def _trajectory_log_path() -> str:
    """Absolute, inside the pod's rsynced deliver dir; relative off-pod. Env var wins.

    Copied from copper_kestrel, which is where it came home from. The parent is created
    HERE rather than relied on, which removes a version dependency this config would
    otherwise carry: views-hydranet creates it from 0.1.3 onward and did not before, while
    `pod_run_model.sh` pins `views-hydranet~=0.1.1` -- a range that still ADMITS 0.1.1 and
    0.1.2. Under either of those an absolute path would raise FileNotFoundError at the first
    write, which is after the data pull and the model build.

    ⚠️ amber_kestrel does NOT set this, which is why the previous phase had no level
    trajectory to read. It is in INSTRUMENTATION_KEYS, so adding it here is permitted
    against the control and is not a second variable.
    """
    override = os.environ.get("HYDRANET_TRAJECTORY_LOG")
    path = (
        Path(override)
        if override
        else _POD_DELIVER_ROOT / _MODEL_NAME / "logs" / "trajectory.csv"
        if _POD_DELIVER_ROOT.is_dir()
        else Path("models") / _MODEL_NAME / "logs" / "trajectory.csv"
    )
    # Never let a read-only or unwritable filesystem turn config LOADING into a failure:
    # tests/test_roster_configs_load.py imports every config, and a config that can only be
    # read where it can also write would make that job environment-dependent.
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(path)


def get_hp_config():
    """
    russet_kestrel — the zero-weight-decay INTERVENTION ARM against bronze_kestrel.

    ⛔ Varies exactly ONE value against its control: weight_decay 0.1 -> 0.0.
    Its control is bronze_kestrel (seed 43, same as this arm), NOT
    amber_kestrel. See the module header.

    Everything below is amber_kestrel's own documentation, inherited verbatim
    because it describes the configuration being replicated:

    STATUS: experiment arm, maturity `candidate`, not a roster member, not an ensemble
    constituent. Carried on development only so a pod can clone it.

    WHAT IT TESTS. violet_visitor emits `nb` x `threshold_gate`: the body is free to put
    mass on zero and the composition discards the gate's answer in favour of the body's.
    Measured on the delivered models, the body supplies ~94% of the composed zero mass and
    the gate is calibrated (median gate 0.69 against an observed 0.69), so occurrence is
    decided by the component that was not trained to decide it. This arm moves occurrence
    to the gate outright: a zero-truncated body (P(body>0) = 1 by construction) composed
    with `soft_gate`, so emitted P(y>0) equals the gate exactly.

    WHAT IS NOT CHANGED, and the honest reason. `loss_class_pos_weight` stays 2.0 — but NOT
    because 2.0 is right. `weighted_bce` at pos_weight w has its optimum at
    s* = wp/(1 + (w-1)p), so at w = 2 the loss instructs the gate to overstate the conflict
    probability by roughly 2x. That is Gneiting & Ranjan (2011) Thm 1: weighting a proper
    scoring rule by a function of the outcome makes it improper, and the minimiser becomes a
    tilted law rather than the truth. Measured on all 64,818 land cells at the 13
    out-of-sample h=1 months (842,634 cell-months), the gate's median |gate - p| is 0.0054
    against 0.0573 for the inflated target, median implied w = 1.0365 — the gate sits at the
    truth and about 10x away from its own objective's optimum. It is disobeying its loss, in
    the direction of honesty.

    So 2.0 is held here as a COMPARABILITY decision, not a correctness one: changing it would
    have made this a two-variable arm and the composition contrast unreadable. The impropriety
    is real and is tested separately by the pos_weight arm `copper_kestrel`
    (views-hydranet reports/2026-10-16_pos_weight_prereg.md), whose control is this artefact.

    M45's failure mode — truncating the body while leaving a gate calibrated for a body still
    absorbing the zeros — is removed by construction rather than by a guessed number.

    ⛔ An earlier version of this paragraph derived the same formula, quoted the same 1.0365,
    and then concluded "there is no hyperparameter to tune here". The premises were right and
    the conclusion did not follow from them (C-303): it argued a property of the LOSS from a
    comparability decision about the EXPERIMENT. Corrected 2026-10-16; the wording is
    views-hydranet's own.

    `loss_reg` stays 'mse' and is dead: utils.py takes the `resolve_family is not None`
    branch, so it is not consulted under a family, exactly as for `nb`.

    `gate_threshold` is ABSENT rather than set to None, and that is OUR choice, not the
    engine's requirement. ADR-069's rule is `elif self.gate_threshold is not None:`
    (https://github.com/views-platform/views-hydranet/blob/e3a039f795787a46a42788aa77256f5470a33086/views_hydranet/utils/config_initializer.py#L1183-L1190
    at views-hydranet v0.1.1, the version the CI load job pins), so a soft_gate config
    carrying `gate_threshold = None` loads without complaint and only a non-None value
    raises. The key is left out anyway, because a `None` sitting beside `soft_gate` reads
    like something waiting to be filled in — and filling it in WOULD raise.

    An earlier version of this comment said the engine refuses the key at all, and cited
    `:1176`. Both were wrong; checked by constructing each case against v0.1.1.

    ⛔ PRE-REGISTERED READOUT, fixed before the run: (mu, theta) and P(Y=0) on the 100-cell
    set, plus gate AP as a one-sided kill at a threshold sized to the collapses on record
    (C-228's 0.31 -> 0.13, M45's -0.238) — not at any-loss, because n=1 against the
    existing vehicle cannot resolve the ~0.019 seed sd. Predicted: emitted positive
    fraction ~0.69, matching the honest out-of-sample rate rather than the in-sample
    0.7639, because the gate is slightly conservative at the top of its range. A material
    overshoot is a finding, not something to tune away.
    """
    hyperparameters = {
        'time_col': 'month_id',
        'id_col': 'priogrid_id',
        'spatial_cols': ['row', 'col'],
        'identity_cols': ['month_id', 'priogrid_id', 'c_id', 'row', 'col'],
        "index_names": ['month_id', 'priogrid_id'],
        'features': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        'static_channels': [],
        'input_channels': 3,
        # #499 Step 1: the whole PRIO-GRID raster (360 x 720 at 0.5 deg), because REGION is "land"
        # (rows span 278, cols 719 — the 180 x 180 Africa+ME crop at (87, 310) refuses it). Leading
        # rows/cols with no land are zero; views-hydranet's DataSniffer documents this as expected.
        # Offsets are 1, not 0: pipeline-core numbers row/col from 1 (row = (pgid-1)//720 + 1), and the
        # volume indexes row - row_offset, so 1 maps row 1..360 onto 0..359 (9540a7bc, 2026-04-28).
        'row_offset': 1,
        'col_offset': 1,
        'height': 360,
        'width': 720,

        'model': 'HydraBNUNet06_LSTM4',
        'total_hidden_channels': 32,
        'dropout_rate': 0.15,
        'window_dim': 32,
        'output_channels': 1,
        'weight_init': 'xavier_norm',
        # gated forecast (gate x body); all-cell body.
        #
        # THE ARM. These two values, and the absence of a third, are the whole difference
        # from violet_visitor. What lets `truncated_nb` take `soft_gate` is that ADR-069
        # composition rule (1) -- "a self-zeroed family must NOT be gated" -- does not
        # fire, because `truncated_nb` is not self-zeroed:
        #   https://github.com/views-platform/views-hydranet/blob/e3a039f795787a46a42788aa77256f5470a33086/views_hydranet/utils/config_initializer.py#L1132-L1141
        # The set it is absent from is `SELF_ZEROED_FAMILIES = frozenset({"zinb"})`, which
        # lives in the registry, not in config_initializer -- the config path reads it
        # through `self_zeroed_family_names()`, and the family class attribute is the
        # authority the set is parity-tested against:
        #   https://github.com/views-platform/views-hydranet/blob/e3a039f795787a46a42788aa77256f5470a33086/views_hydranet/distributions/registry.py#L45-L61
        # There is deliberately no `gate_threshold` below. Do not add one back to "match
        # violet_visitor" — a non-None value raises under any composition but
        # `threshold_gate`:
        #   https://github.com/views-platform/views-hydranet/blob/e3a039f795787a46a42788aa77256f5470a33086/views_hydranet/utils/config_initializer.py#L1183-L1190
        # (Pinned at v0.1.1, the version the roster-configs-load CI job installs. A bare
        # file:line drifts with the other repo; this does not.)
        'output_distribution': 'truncated_nb',
        'forecast_composition': 'soft_gate',
        'freeze_recurrent': 'cell',
        # #484 (views-hydranet 0.1.1): a CPU device is a hard stop, not a banner. A fresh install
        # once resolved a torch this machine's driver could not run and a roster model trained
        # for 6 h 46 m on CPU (views-hydranet#377). CPU training is never intended here.
        'require_cuda': True,
        'reg_activation': 'softplus',
        'body_supervision': 'all',  # all-cell body supervision (ADR-065; supersedes the retired point-mask knob)

        'windows_per_lesson': 3,
        'learning_rate': 0.001,
        # ⛔ THE ARM. This single value is the whole difference from the control.
        # `ge=0.0` at config_initializer.py:150 is what makes it reachable.
        'weight_decay': 0.0,
        'scheduler': 'WarmupDecay',
        'warmup_steps': 100,
        'clip_grad_norm': True,
        'torch_seed': 43,
        'np_seed': 43,
        'freeze_multitask_balancer': True,

        # The per-step level trajectory. INSTRUMENTATION_KEYS exempts this key, so it is
        # permitted to differ from the control -- and amber_kestrel's omission of it is
        # exactly why the previous phase had no trajectory baseline to read.
        'trajectory_log_path': _trajectory_log_path(),

        'classification_targets': ['by_sb_best', 'by_ns_best', 'by_os_best'],
        'regression_targets': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
        'transformations': {'log1p': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'], 'asinh': [], 'identity': []},
        'derivations': {'binary': [
            {'from': 'lr_sb_best', 'to': 'by_sb_best', 'threshold': 0},
            {'from': 'lr_ns_best', 'to': 'by_ns_best', 'threshold': 0},
            {'from': 'lr_os_best', 'to': 'by_os_best', 'threshold': 0},
        ]},
        'steps': list(range(1, 37)),
        'time_steps': 36,

        # all-cell body, plain MSE
        'loss_reg': 'mse',
        # gate calibration
        'loss_class': 'weighted_bce',
        'loss_class_pos_weight': 2.0,
        'onset_bias_init': -7.0,

        'ss_schedule': 'linear',
        'ss_warmup_lessons': 15,
        'ss_epsilon_max': 0.0,

        # A run-time budget, not a model choice. 300 is production (#463). It was dropped to
        # 40 for the #499 integration pass at global land (2026-09-19) and restored here for
        # the #505 calibration run — this is the PR that #499 said was owed.
        # Measured on rented RTX PRO 4500 SE class hardware, n=3: a full run is 202-272 min
        # end to end — 300 lessons plus the 13-origin evaluation — i.e. ~4 h, or 40-54 s per
        # lesson. An earlier version of this comment said 84 s and ~7 h; that was taken from
        # the FIRST lesson of a cold two-lesson smoke run, which carries warm-up and is not
        # representative of the other 299. config_sweep.py keeps its own, smaller budget on
        # purpose — a sweep explores, it does not produce a deliverable.
        'total_lessons': 300,
        'max_ratio': 0.95,
        'min_ratio': 0.05,
        'slope_ratio': 0.75,
        'roof_ratio': 0.7,
        'min_events': 5,
        'sampling_strategy': 'sigmoid',
        'sampling_steepness': 1.0,

        # D x K = 4 x 4 = 16 produced draws, matching the other seven roster members
    # (ADR-015 6: the count that matters is the count a model PRODUCES). Was D=8
    # with no head sampler, i.e. 8 produced -- which is why rusty_bucket could not
    # be rewired to the roster: it declares expected_samples_per_model: 16 and the
    # ADR-015 contract refuses a constituent that emits something else.
    #
    # Settled 2026-08-11 by maintainer decision (#146, #372 item 2). This model stays
    # EXPERIMENT_IN_PROGRESS -- its family, composition and seed are still unsettled
    # and its queryset is not yet migrated -- but the SAMPLE COUNT is no longer part
    # of the experiment. It is pinned by the ensemble contract
    # (tests/test_ensemble_configs.py::test_declared_modelset_and_sample_counts_match_reality),
    # not by the roster foundation pins that this model is exempt from.
    'n_posterior_samples': 4,
    'n_head_samples': 4,
    'rollout_feedback': 'sample',
    'bn_recalibrate': True,
        'evaluation_mode': 'stochastic',
        'aggregate_method': 'arithmetic_mean',
        'skip_predictions_delivery': True,
        'min_free_disk_gb': 10.0,
    }
    return hyperparameters

# copper_kestrel is NOT a roster member. It is one arm of the gate-objective experiment
# pre-registered in views-hydranet reports/2026-10-16_pos_weight_prereg.md, and it exists as
# its own directory for a mechanical reason: tools/podrun/pod_run_model.sh clones
# `-b development` on the pod, so the config a pod trains is whatever is on development --
# not whatever is in the operator's working tree.
#
# ITS CONTROL IS amber_kestrel, not violet_visitor. Editing amber_kestrel in place to carry
# this arm was the alternative, and it was rejected (Simon, 2026-10-16): amber_kestrel's
# control artefact -- the 2026-10-12 violet_visitor vehicle -- cannot be retrained (C-317),
# so varying a second key there would have spent the composition contrast permanently to buy
# a pos_weight result. A second directory keeps both contrasts readable.
#
# It differs from amber_kestrel in exactly ONE value: loss_class_pos_weight, 2.0 -> 1.0.
# Everything else -- composition, architecture, seeds, lessons, features, targets, grid,
# losses, sample counts -- is copied unchanged, so the as-run amber_kestrel arm
# (calibration_model_20261013_010739.pt, sha256 2a5efc2a67c08eac..., git 73c3f42) is a valid
# control and the experiment is one arm rather than two.
#
# ⛔ `body_nll_denominator` is deliberately NOT set here, so it keeps the library default
# "positives" -- the as-run arm's reduction. The denominator fix is a SEPARATE
# pre-registration (views-hydranet reports/2026-10-15_loss_balance_prereg.md) and must not
# ride along, or neither arm is one-variable.
#
# THE CHAIN, stated here because no test asserts it. copper_kestrel -> amber_kestrel ->
# violet_visitor. The one-variable guard in tests/test_roster_conformance.py is
# NON-TRANSITIVE by design: each arm is checked only against its own declared control,
# because a transitive set would admit the union of every hop's free keys and so weaken by
# one key per arm. Cumulative distance from violet_visitor is therefore THREE values
# (output_distribution, forecast_composition, loss_class_pos_weight) plus the absence of
# gate_threshold. That is a fact for this comment, not a contrast anything measures.


import os
from pathlib import Path

#: Where the per-task loss trajectory CSV is written.
#:
#: ⛔ A RELATIVE path dies with the pod. tools/podrun/pod_run_model.sh rsyncs only
#: $OUT=/workspace/deliver/<model>, and it copies nothing from the repo clone -- no logs/
#: glob exists anywhere in it. views-models#558 / C-161 is the record of exactly that: half
#: of what a pod produced did not come back, and the gate posterior was lost that way.
#:
#: So this resolves to an ABSOLUTE path inside the deliver directory WHEN RUNNING ON A POD,
#: detected by the deliver root existing, and to a repo-relative path otherwise. No operator
#: action, nothing pod-specific hardcoded, and a local run still writes locally -- which
#: matters because "remember an extra rsync" is the mechanism C-161 already records failing.
#: views-hydranet 0.1.3 creates the parent directory (it did not before), so the absolute
#: path is safe; under 0.1.2 and earlier it would have raised FileNotFoundError mid-run.
_MODEL_NAME = Path(__file__).resolve().parent.parent.name
_POD_DELIVER_ROOT = Path("/workspace/deliver")


def _trajectory_log_path() -> str:
    """Absolute, inside the pod's rsynced deliver dir; relative off-pod. Env var wins.

    The parent is created HERE rather than relied on, which removes a version dependency
    this config would otherwise carry. views-hydranet creates it from 0.1.3 onward and did
    not before, while `pod_run_model.sh` pins `views-hydranet~=0.1.1` — a range that still
    ADMITS 0.1.1 and 0.1.2. Under either of those an absolute path would raise
    FileNotFoundError at the first write, which is after the data pull and the model build.
    One idempotent mkdir here means the config is correct under every version the pin admits.
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
    copper_kestrel -- the gate-objective arm: does removing the improper tilt buy anything?
    (views-hydranet reports/2026-10-16_pos_weight_prereg.md, committed 02d1294.)

    STATUS: experiment arm, maturity `candidate`, not a roster member, not an ensemble
    constituent. Carried on development only so a pod can clone it.

    WHAT IT TESTS. `weighted_bce` at pos_weight w has its optimum at s* = wp/(1 + (w-1)p),
    so at w = 2 the loss instructs the gate to overstate the conflict probability by roughly
    2x. That is Gneiting & Ranjan (2011) Thm 1: weighting a proper scoring rule by a function
    of the outcome makes it improper, and the minimiser becomes a tilted law rather than the
    truth. This arm sets w = 1, which makes the objective proper, and asks whether anything
    downstream improves. Every other value is amber_kestrel's.

    ⛔ WHAT THIS ARM CANNOT SHOW, pre-registered before the run (prereg 3.2). The tilt
    s* = wp/(1 + (w-1)p) is strictly MONOTONE in p, and average precision depends on the gate
    only through RANKS. A pure recalibration -- exactly what this arm removes -- therefore
    leaves every cell's ordering unchanged and leaves gate AP EXACTLY unchanged. P-3 pins
    that: AP does NOT move, within +/-0.010 of the as-run arm's 0.5324. If AP does move, the
    prereg 7 says that is a DIFFERENT finding -- the altered loss changing what the shared
    trunk learned -- and it must not be reported as confirming this hypothesis. The primary
    endpoint is calibration, not AP, because that is where the motivated mechanism lands.

    ⚠️ THE CONTRAST CARRIES A MEASURED HANDICAP (prereg 3.5). The as-run arm's gate AP is
    0.5324 against violet_visitor's 0.5631 -- the composition arm lost 0.0307 of gate skill,
    and the denominator evidence attributes that to a loss imbalance that is STILL PRESENT
    here, because `body_nll_denominator` stays at its default. So thresholds are sized against
    0.5324, not 0.5631, and an AP recovery here licenses NO claim about the denominator arm.

    ⛔ PRE-REGISTERED READOUT, fixed before the run. Primary first:
      P-1  median |gate - p|, all 64,818 land cells, 13 out-of-sample h=1 months -- does not
           worsen, stays <= 0.010. Above 0.010 means the gate left the truth, which the
           hypothesis says is impossible.
      P-2  median implied w on the same bins -- moves toward 1.0 or stays, in [0.90, 1.15].
      P-3  gate AP, paired by month, against 0.5324 -- UNCHANGED within +/-0.010 (see above).
      P-4  crps_all -- does not worsen; improves by <= 2% if at all. Worse than 5% fails
           FAO-02's bar.
      P-5  emitted positive fraction, 100-cell set, h=1 -- stays in [0.55, 0.85]; the
           composition is unchanged.
      P-6  act_ratio at h18 -- at or below the as-run arm's. A decrease is the expected
           direction.

    THE HONEST PRIOR IS A NULL, and the prereg says so with a number: ~30% here against ~60%
    for the denominator arm. r4 MEASURED the denominator's mechanism reaching the weights
    (shared-trunk gradient-norm ratio 0.312 -> 10.497, cos(grad body, grad gate) +0.72 ->
    -0.20, 75% of windows in conflict). This arm has no comparable measurement; its case is a
    derivation, and theory about an objective is weaker evidence than a measured gradient.
    Calibration is also already 0.0054 against a 0.003448 base rate, so the arm is weakly
    powered on its own primary endpoint. A null is therefore INFORMATIVE rather than a
    failure: it closes "the gate's objective is miscalibrated" and leaves the shared trunk as
    the remaining candidate. That only works because the null was predicted in advance.

    ⚠️ THE PRE-FLIGHT IS NOT THE DENOMINATOR ARM'S (prereg 6a). That gate was a loss-MAGNITUDE
    assertion; this arm changes no reduction, so it would pass VACUOUSLY and means nothing
    here. This arm's free pre-flight is to confirm from the BUILT loss that pos_weight is 1.0
    and nothing else moved -- `arm_identity_check.py` plus reading `crit_cls.pos_weight_value`
    off the constructed criterion. That is the only check that catches the config key not
    reaching the loss, which is the failure mode that cost views-hydranet a dead feature on
    its own #398.

    `loss_reg` stays 'mse' and is dead: utils.py takes the `resolve_family is not None`
    branch, so it is not consulted under a family, exactly as for `nb`.

    `gate_threshold` is ABSENT rather than set to None, inherited from amber_kestrel and for
    amber_kestrel's reason: ADR-069's rule is `elif self.gate_threshold is not None:`
    (https://github.com/views-platform/views-hydranet/blob/e3a039f795787a46a42788aa77256f5470a33086/views_hydranet/utils/config_initializer.py#L1183-L1190
    at views-hydranet v0.1.1), so a soft_gate config carrying `gate_threshold = None` loads
    without complaint and only a non-None value raises. The key is left out anyway, because a
    `None` sitting beside `soft_gate` reads like something waiting to be filled in -- and
    filling it in WOULD raise.
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
        # INHERITED FROM THE CONTROL, NOT THIS ARM'S VARIABLE. These two values and the
        # absence of gate_threshold are amber_kestrel's composition; they must stay identical
        # or this stops being a one-variable contrast. What lets `truncated_nb` take
        # `soft_gate` is that ADR-069 composition rule (1) -- "a self-zeroed family must NOT
        # be gated" -- does not fire, because `truncated_nb` is not self-zeroed:
        #   https://github.com/views-platform/views-hydranet/blob/e3a039f795787a46a42788aa77256f5470a33086/views_hydranet/utils/config_initializer.py#L1132-L1141
        # The set it is absent from is `SELF_ZEROED_FAMILIES = frozenset({"zinb"})`, which
        # lives in the registry, not in config_initializer -- the config path reads it through
        # `self_zeroed_family_names()`, and the family class attribute is the authority the set
        # is parity-tested against:
        #   https://github.com/views-platform/views-hydranet/blob/e3a039f795787a46a42788aa77256f5470a33086/views_hydranet/distributions/registry.py#L45-L61
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
        'weight_decay': 0.1,
        'scheduler': 'WarmupDecay',
        'warmup_steps': 100,
        'clip_grad_norm': True,
        'torch_seed': 42,
        'np_seed': 42,
        'freeze_multitask_balancer': True,

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
        # Instrumentation, NOT an experimental variable: it changes what is RECORDED, never
        # what is computed. views-hydranet 0.1.3 widens this CSV to one `loss/<task>` column
        # per task (6 here), which is the only reason the composition arm's 65x loss imbalance
        # had to be reconstructed from saved weights weeks after the fact -- it was unset on
        # both delivered runs.
        'trajectory_log_path': _trajectory_log_path(),

        'loss_class': 'weighted_bce',
        # ⛔ THE ARM. This single value is the whole difference from amber_kestrel. At w = 2 the
        # weighted-BCE optimum is s* = wp/(1 + (w-1)p) ~ 2p, so the loss asks the gate to
        # overstate occurrence by roughly 2x -- improper by Gneiting & Ranjan (2011) Thm 1. At
        # w = 1 the objective is proper and its optimum is p itself. Measured on all 64,818
        # land cells at the 13 out-of-sample h=1 months (842,634 cell-months), the as-run arm's
        # gate sits at median |gate - p| = 0.0054 with median implied w = 1.0365 -- at the
        # truth and ~10x away from its own objective's optimum, i.e. disobeying its loss in the
        # direction of honesty. This arm asks what removing the tilt buys. Do not change any
        # other value in this file to "improve" the result: a second variable makes the
        # contrast against the as-run arm unreadable, and that arm cannot be retrained (C-317).
        'loss_class_pos_weight': 1.0,
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

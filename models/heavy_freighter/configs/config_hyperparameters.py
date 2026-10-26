import os
from pathlib import Path

#: Where the per-task loss trajectory CSV is written.
#:
#: ⛔ A CWD-RELATIVE path dies with the pod. tools/podrun/pod_run_model.sh rsyncs only
#: $OUT=$ROOT/deliver/<model> (see _POD_DELIVER_ROOT below -- $ROOT is NOT fixed), and it
#: copies nothing from the repo clone -- no logs/ glob exists anywhere in it.
#: views-models#558 / C-161 is the record of exactly that: half of what a pod produced did
#: not come back, and the gate posterior was lost that way.
#:
#: So this resolves to an ABSOLUTE path inside the deliver directory WHEN RUNNING ON A POD,
#: detected by the deliver root existing, and to a path inside this model's OWN directory
#: otherwise. The off-pod leg is anchored on __file__ rather than being cwd-relative:
#: `pod_run_model.sh:371` does `cd "$REPO/models/$MODEL"`, and `models/<model>/run.sh` does
#: not cd at all, so a "models/..." relative path would resolve against whatever directory
#: the operator happened to be in -- doubling to models/<model>/models/<model>/logs/ in the
#: first case and landing under $HOME in the second. No operator action, nothing pod-specific
#: hardcoded, and a local run still writes beside the model -- which matters because
#: "remember an extra rsync" is the mechanism C-161 already records failing.
_MODEL_DIR = Path(__file__).resolve().parent.parent
_MODEL_NAME = _MODEL_DIR.name

#: ⛔ RELOCATABLE, not fixed. `tools/podrun/pod_run_model.sh:94` is
#: `ROOT="${PODRUN_ROOT:-/workspace}"` and `OUT=$ROOT/deliver/$MODEL`, and its own comment
#: says the runner and the delivery script must AGREE on it. Hardcoding /workspace here
#: would make a relocated pod fail `is_dir()`, fall through to the off-pod branch and write
#: the trajectory OUTSIDE $OUT -- which is not rsynced home, i.e. C-161 again, the exact
#: loss this helper exists to prevent. Reading the same variable is what keeps them in step.
_POD_DELIVER_ROOT = Path(os.environ.get("PODRUN_ROOT") or "/workspace") / "deliver"


def _trajectory_log_path() -> str:
    """Absolute in both legs: inside the pod's rsynced deliver dir on a pod, beside this
    model otherwise. HYDRANET_TRAJECTORY_LOG overrides both.

    Neither leg depends on the process cwd. That is deliberate and is the second defect
    corrected against copper_kestrel's original -- see _MODEL_DIR above.

    The parent is created HERE rather than relied on, because views-hydranet only creates it
    from 0.1.3 onward while `pod_run_model.sh:172` pins `views-hydranet~=0.1.1` — a range
    that still ADMITS 0.1.1 and 0.1.2, under which an absolute path would raise
    FileNotFoundError at the first write, after the data pull and the model build.

    ⚠️ The mkdir is BEST-EFFORT and does not make this airtight. Importing a config must
    never fail on an unwritable filesystem — tests/test_roster_configs_load.py imports every
    config — so an OSError here is swallowed and the run proceeds. If the directory genuinely
    cannot be created, the failure still surfaces at the first trajectory write, late and
    expensively, exactly as it would without this call. What the mkdir removes is the common
    case (parent simply absent), not the whole class.
    """
    override = os.environ.get("HYDRANET_TRAJECTORY_LOG")
    path = (
        Path(override)
        if override
        else _POD_DELIVER_ROOT / _MODEL_NAME / "logs" / "trajectory.csv"
        if _POD_DELIVER_ROOT.is_dir()
        else _MODEL_DIR / "logs" / "trajectory.csv"
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
    return {   'time_col': 'month_id',
    'id_col': 'priogrid_gid',
    'spatial_cols': ['row', 'col'],
    'identity_cols': ['month_id', 'priogrid_gid', 'c_id', 'row', 'col'],
    'index_names': ['month_id', 'priogrid_gid'],
    'features': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
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
    'dropout_rate': 0.125,
    'window_dim': 32,
    'output_channels': 1,
    'weight_init': 'xavier_norm',
    'windows_per_lesson': 3,
    'learning_rate': 0.001,
    'weight_decay': 0.1,
    'scheduler': 'WarmupDecay',
    'warmup_steps': 100,
    'clip_grad_norm': True,
    'torch_seed': 47,
    'np_seed': 47,
    'classification_targets': ['by_sb_best', 'by_ns_best', 'by_os_best'],
    'regression_targets': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
    'transformations': {   'log1p': ['lr_sb_best', 'lr_ns_best', 'lr_os_best'],
                           'asinh': [],
                           'identity': []},
    'derivations': {   'binary': [   {'from': 'lr_sb_best', 'to': 'by_sb_best', 'threshold': 0},
                                     {'from': 'lr_ns_best', 'to': 'by_ns_best', 'threshold': 0},
                                     {'from': 'lr_os_best', 'to': 'by_os_best', 'threshold': 0}]},
    'steps': [   1,
                 2,
                 3,
                 4,
                 5,
                 6,
                 7,
                 8,
                 9,
                 10,
                 11,
                 12,
                 13,
                 14,
                 15,
                 16,
                 17,
                 18,
                 19,
                 20,
                 21,
                 22,
                 23,
                 24,
                 25,
                 26,
                 27,
                 28,
                 29,
                 30,
                 31,
                 32,
                 33,
                 34,
                 35,
                 36],
    'time_steps': 36,
    'loss_reg': 'mse',
    'loss_class': 'weighted_bce',
    'loss_class_alpha': 0.75,
    'loss_class_gamma': 1.5,
    'onset_bias_init': -7.0,
    'ss_schedule': 'linear',
    'ss_warmup_lessons': 10,
    'ss_epsilon_max': 0.5,
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
    'sampling_strategy': 'threshold',
    'n_posterior_samples': 4,
    'evaluation_mode': 'stochastic',
    'aggregate_method': 'arithmetic_mean',
    'skip_predictions_delivery': True,
    # The sole sink for the per-task loss split: six columns, body and gate separately
    # for each of sb/ns/os. Without it the split has to be reconstructed from weights,
    # which is what happened to both 2026-10-12/13 delivered runs. Helper ADAPTED from
    # copper_kestrel, where the original has produced a complete CSV on six pod runs, with
    # two corrections it does not carry: /workspace was hardcoded although
    # pod_run_model.sh honours PODRUN_ROOT, and its off-pod leg was cwd-relative. The six
    # kestrel configs still carrying the original are deliberately untouched (#604).
    'trajectory_log_path': _trajectory_log_path(),
    'output_distribution': 'nb',
    'forecast_composition': 'soft_gate',
    # The #607 structure-penalty arm: ONE key off the 2026-10-25 control (git c8ae1803).
    # 0.416904 is re-derived on THIS vehicle (`nb`) over all three regression targets --
    # the prereg's 8.89624 is void three ways (derived on amber_kestrel's `truncated_nb`,
    # a superseded v1 script, and gradient-matched against one of the three targets that
    # training_engine.py sums over). views-hydranet 0.1.7 reads it at :1325 and applies it
    # at :1127-1128; at 0.0 the term is a byte-identical no-op.
    'struct_penalty_weight': 0.416904,
    'freeze_multitask_balancer': True,
    'freeze_recurrent': 'cell',
    # #484 (views-hydranet 0.1.1): a CPU device is a hard stop, not a banner. A fresh install
    # once resolved a torch this machine's driver could not run and a roster model trained
    # for 6 h 46 m on CPU (views-hydranet#377). CPU training is never intended here.
    'require_cuda': True,
    'n_head_samples': 4,
    'reg_activation': 'softplus',
    'body_supervision': 'all',
    # C-259 / #295: scheduled sampling is ACTIVE here (ss_schedule='linear',
    # ss_epsilon_max=0.5), and ss_feedback defaults to 'mean' — which contradicts
    # rollout_feedback='sample' and makes this config UNLOADABLE. Training would feed back a
    # different object than inference rolls out on. Declared explicitly 2026-09-07 after the
    # roster emit run found this model could not be run at all.
    'ss_feedback': 'sample',
    'rollout_feedback': 'sample',
    'bn_recalibrate': True,
    'loss_class_pos_weight': 2.0,
  }

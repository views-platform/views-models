"""Looking at what EXP-02's arms predict, where they predict it, and whether it clusters.

EXP-02 contrasts `UpsampleConv2` against `HydraBNUNet06_LSTM4` at 300 lessons on two seeds
(views-models#685). Its pre-registered readout is gate-side and numeric — a within-seed sign
test on the gate posterior, sliced at h=1 and h=36 by `tools/podrun/slice_gate_posterior.py`.

⛔ **This group answers a question that readout cannot.** An upsampling decoder can score well
on a cell-wise proper score while spreading predicted mass across neighbouring cells, because
a scalar averaged over 64,818 cells says nothing about whether the map is sharp. The operator
asked to judge that by eye:

    "one of the points here is whether or not the spatial patterns cluster more or less.
     And I know we have numbers for it, but I need to eyeball it."

So this is deliberately an *instrument for looking*, not a scoring path, and nothing here
produces a result that belongs in an outcome log. The concentration figure printed beside each
panel is window-dependent and has no pre-registered threshold; quoting it as a finding would be
the "correct value, wrong question" failure the register keeps recording.

    python tools/exp02/plot_spatial_windows.py \
        --arm "ctl_42=models/ctlfullzero_fortytwo/data/generated/calibration_delivery_20261103" \
        --arm "upc2_42=models/upctwofullzero_fortytwo/data/generated/calibration_delivery_20261103" \
        --zone levant --month 504 --draw 0

Two things it reads, because they answer different questions: the per-origin parquets hold the
posterior **mean** (verified 2026-11-03 against the draws archive — `max|diff| = 1.6e-06` for
the mean against 62.4 for the median), and `--draw N` reaches into the arm's `lr` archive for a
single draw. The mean is smooth by construction, so a clustering comparison read only off means
partly measures that averaging rather than the model.
"""

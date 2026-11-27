ensemble:      chrome_box (pgm, global land) — phase 2 of 2
members:       ashen_sparrow brisk_sparrow candid_sparrow dusky_sparrow eager_sparrow faint_sparrow gentle_sparrow humble_sparrow
members_run:   8 trained this invocation, 0 already present
reconciled:    against chrome_crate, read from local disk
finished:      2026-10-30T07:30:32Z
draws:         8 x 128 = 1024 configured (epic #595, NOT thinned; enforced by ADR-015 expected_models, not measured here)
runtime_min:   193
PEAK cgroup:   memory.peak=89645MiB  sampled_max=89609MiB
peak_expected: >= 68.2 GiB, MEASURED 2026-10-28 and a LOWER BOUND (30s sampling over
               a 4.4-minute stage cannot see a shorter transient). The two figures
               above are the ones that replace it; memory.log carries the series.
mem_floor:     80GB enforced, box reported 87GB
disk_floor:    60GB enforced (DERIVED) — replace it with pool: below
pool:          27G
pipeline_core: 3.3.5 (floor 3.3.5)
git:           a32a1ee
driver:        tools/podrun/pod_run_chrome_box.sh
published:     nothing — this runner uploads nothing and holds no publish credential

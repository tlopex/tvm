---
description: "Pre-commit kernel regression benchmark (auto GPU selection + parallel sweep)"
argument-hint: "[--filter SUBSTR] [--baseline PATH] [--threshold PCT] [--label STR] [--util-threshold PCT]"
allowed-tools: ["Bash", "Read"]
---

# tir-bench — local kernel regression check

Run the curated workload list in `.claude/commands/tir-bench/workloads.yaml`
on every free GPU in parallel, dump JSON, and diff against the previous run.

> **NEVER ACTIVELY SELECT A GPU FOR THIS run.py — IT SELECTS GPUs AUTOMATICALLY.**
> There is no `--gpus` flag. Do not set `CUDA_VISIBLE_DEVICES` to pin cards either.
> run.py probes every visible GPU, then on each acquire scans utilization and
> picks any card below `--util-threshold` (skipping cards in active use,
> requeueing if a neighbor bursts mid-run). Manually pinning defeats this and
> can land work on a busy card. If the machine is contended, let it run — busy
> cards are skipped and re-tried automatically; just re-run later for full coverage.

Methodology mirrors `tirx-bench-ci/bench-run.sh`: poll per-GPU
`utilization.gpu` to find idle cards (a card merely holding resident VRAM at
low util still counts as free), hand out one workload per GPU under an
in-process lock, and fall back to a 5 s retry when all are busy. Interference
is detected via per-PID `sm%` (`nvidia-smi pmon`) and the workload requeued.

**Args forwarded to run.py:** `$ARGUMENTS`

## Steps

1. Confirm `tirx-kernels` is importable and `nvidia-smi` works:
   ```bash
   python -c "import tirx_kernels; print('ok')"
   nvidia-smi --query-gpu=index,uuid --format=csv,noheader
   ```
2. Run the benchmark:
   ```bash
   python .claude/commands/tir-bench/run.py $ARGUMENTS
   ```
3. Read back the run JSON (`.tir-bench/latest.json`), the summary
   (`.tir-bench/reports/<ts>-summary.md`), and — if a baseline existed —
   the absolute-ms regression report (`.tir-bench/reports/<ts>.md`) AND
   the ratio-based regression report (`.tir-bench/reports/<ts>-ratio.md`,
   ours/ref normalised — robust to GPU-contention noise; the "ref Δ"
   column flags rows where the reference impl itself drifted >20%).
   Summarise to the user: count of regressions / improvements / failures
   + the headline row for any regression beyond the threshold.

## Baseline

The diff target is `.claude/commands/tir-bench/baseline.json` — a checked-in
reference snapshot. To promote a fresh sweep result as the new baseline:

```bash
cp .tir-bench/runs/<timestamp>.json .claude/commands/tir-bench/baseline.json
```

Override per-run with `--baseline /path/to/some.json`.

### Which commit does a baseline correspond to?

A baseline records provenance two ways:

- `git:` — the **commit SHAs of the branch that ran the sweep**. Author-side and
  convenient, but a squash/rebase merge rewrites them, so after merge they no
  longer resolve to a mainline commit. Treat them as a hint, not the truth.
- `kernel_tree:` — **merge-stable content fingerprints**: the git *tree* SHAs of
  the source dirs that determine kernel codegen (`tir:python/tvm/tirx`,
  `tirx-kernels:tirx_kernels`). A tree SHA is content-addressed, so it is
  identical before and after a merge as long as the directory content is
  unchanged. Confirm a checkout matches a baseline with
  `git rev-parse HEAD:python/tvm/tirx` / `git rev-parse HEAD:tirx_kernels`.

**The authoritative "which commit set this baseline" is the commit that last
touched `baseline.json`** — `git log -1 -- .claude/commands/tir-bench/baseline.json`
always resolves to a real commit in the current history (post-merge, the
squash/merge commit). Use `kernel_tree` to verify the *code* matches.

## Outputs

```
.claude/commands/tir-bench/
├── run.py                    # the script
├── workloads.yaml            # curated (kernel, config) list
└── baseline.json             # pinned reference for diffs

.tir-bench/                   # runtime artifacts (relative to cwd, regenerable)
├── runs/<timestamp>.json     # this run's full result
├── runs/<timestamp>.log      # live orchestrator log
├── latest.json / latest.log  # symlinks → most recent
├── reports/<ts>-summary.md   # per-run human-readable table (with baseline/ours ratio)
├── reports/<ts>.md           # absolute-ms diff vs pinned baseline (if baseline exists)
├── reports/<ts>-ratio.md     # ratio-based diff (ours/ref) — robust to contention
└── logs/<kernel>__<config>.log
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | All workloads OK, no regression exceeded threshold (or no baseline). |
| 2    | Config error: no workloads to run / bad YAML. |
| 3    | One or more regressions above `--threshold` percent. |

## Editing workloads

`.claude/commands/tir-bench/workloads.yaml` carries the list. To pick from
the full set of valid (kernel, config) pairs on this machine:

```bash
python -m tirx_kernels.registry --format=benchrun --cc 10 | cut -d'|' -f1,2
```

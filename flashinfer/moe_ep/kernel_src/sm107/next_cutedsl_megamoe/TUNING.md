# SM107 MegaMoE tuning and measurement

Qualify correctness first using the [Rubin runbook](../../../../../docs/design_docs/moe_ep_sm107_qualification.md).
All three formats (NVFP4, MXFP8 E4M3, MXFP8 E5M2) require native SM107 and
a compatible CuTe DSL build. Export `CUTE_DSL_ARCH=sm_107a` before Python
starts. The original PR's internal-build performance claims are not a
baseline for this rebased implementation; rerun on the supported public
compiler and retain absolute measurements.

## What the benchmark measures

`benchmarks/bench_moe_ep_sm107_block_scaled_mega.py` reports:

- `--mode kernel`: a launch over already staged inputs, including the
  required output/reset operations and dispatch, both GEMMs, and combine.
- `--mode compute`: inputs staged once, then backend `compute()` with a
  preallocated owned output reused across calls. Includes the backend wrapper,
  required resets/reduction, kernel, and output copy, but no recurring input staging.
- `--mode forward`: public `MoEEpLayer.forward()` with BF16 inputs, including
  validation, Torch quantization/staging, owned output handling, and the kernel.
- `--execution eager|graph`: eager launches or replay of a warmed CUDA
  graph. These are separate series, not interchangeable measurements.

CUDA events measure the GPU stream interval, not host wall-clock latency.
Eager intervals can include device idle gaps while the host enqueues work.
`compute` allocates its output once before warmup and graph capture. Eager
`forward` allocates on each call; graph `forward` performs Python validation
and output allocation during capture, outside replay timing. Graph capture
and initial replay warmup are outside timing.
The `timing_protocol` record states the input-staging boundary, output
ownership/allocation, GPU timing scope, and number of replay warmups.

Cache policy is independent of synchronization. The default L2-flushed series
allocates and fills a fresh 300 MiB FP32 buffer before each start event,
matching the historical Blackwell harness. The buffer stays alive through the
launch; allocation and random writes are outside the event window.
`--no-l2-flush` selects consecutive launches without those writes. Both series
use a barrier before the timed batch and no inter-iteration barriers or host synchronization. A
barrier before every iteration would instead measure a different, from-idle
protocol. Match that policy as well as the cache policy on both sides of a
comparison. Flush memory is included in PyTorch allocator measurements.

The benchmark reports two latency statistics:

- `max_rank_p50_us`: maximum of each rank's median, matching the Blackwell
  autotuner's aggregation. This is the primary latency statistic and the
  denominator of `model_tflops_per_rank`.
- `p50_rank0_us`: rank-zero median, matching historical Blackwell benchmark tables.

Raw samples remain available for offline tail-latency and variability analysis.

These are durations from each rank's local CUDA events, not a synchronized
cross-device wall-clock timestamp. Schema-version-3 JSONL preserves
every rank's samples, full resolved
configuration, geometry, live/capacity counts, seed and repetition, software versions,
repository status, preprocessing time, PyTorch peak memory, and workspace
sizes. PyTorch allocator peaks do not account for all external NVSHMEM heap
allocations; inspect the workspace sizes and NVSHMEM heap configuration too.
Version 3 adds the maximum of per-rank medians and identifies the primary
statistic. Version 2 used `p50_max_rank_us` for model TFLOPS, allocated eager
compute output per call, and reused the flush buffer. Keep those protocols
separate when comparing old and new records.

Before accepting a result, the harness compares evenly spaced output rows,
including the first and last, to a collective Torch oracle using the actual
quantized bytes. It also checks the entire output for nonfinite values.
The 64-row sample is a benchmark guard, not a replacement for full small
problem correctness tests or sanitizer coverage.

The default geometry is H=7168, I=3072, E=384, top-k=6. Override it with
`--hidden`, `--intermediate`, `--num-experts`, and `--topk`. `--tokens`
controls live rows; `--capacity` fixes a larger workspace capacity.
`--seed` controls weight and routing generation. `--quant-kind all`
selects all three formats; `both` retains NVFP4 plus E4M3.

Defaults are 20 warmups and 50 timed iterations. Run three fresh processes
with the same seed, labeling them `--repetition 1`, `2`, and `3`, to check
repeatability. This flag labels a run; it does not launch repetitions. Change
seeds in a separate routing-variation experiment, and increase iterations or
repetitions if the variability does not resolve the claimed difference.

## Knob policies

`--knobs default` uses the public backend defaults (also the benchmark
default). `heuristic` selects `default_knobs()`; `cache` resolves a
previously qualified local winner; a JSON object supplies explicit shim
knobs. `reported` replays the profile table carried by PR #4601, restricted
to its EP4/H7168/I3072/E384/K6 geometry and listed token counts. Those
profiles came from NVFP4; their MXFP8 adaptation is a candidate, not a
measured MXFP8 optimum.

Engine configuration has the same distinction: `knobs=None` preserves
explicit fields, `knobs="cache"` performs lookup with heuristic fallback,
and a dictionary overrides fields. Online `knobs="auto"` is unsupported.

## Qualification workload selection

On the same idle node and pinned environment, measure EP2, EP4, and EP8 where
claimed; T=1, 16, 128, 512, 1024, 2048, 4096, 8192, 16384, and 32768;
all formats; balanced and power-law routing; default and tuned knobs;
kernel and full-forward spans; eager and graph execution. Include small
live batches in a 32768-row capacity. The original PR shape is the first
workload; H7168/I2048/E256/K8 connects to the canonical Blackwell campaign.
Use additional routing seeds on representative imbalanced cases after the
fixed-seed repetitions. Save each run independently. This is an audit
recommendation, not a repository-wide mandatory cross-product. Agree the
workloads and regression margins for the intended support claim, and report
absolute median latency, variability across repetitions, and memory.

For example, choose a persistent results directory outside a small home
quota, and run kernel/forward with consecutive launches. The compute comparison
uses L2 flushing to match Blackwell; no additional flushed kernel/forward
performance series is planned.

```bash
export CUTE_DSL_ARCH=sm_107a
: "${FI_RESULTS:?Set a fresh persistent results directory}"
mkdir -p "$FI_RESULTS"
for fi_repeat in 1 2 3; do
  torchrun --standalone --nproc_per_node=4 benchmarks/bench_moe_ep_sm107_block_scaled_mega.py \
    --quant-kind all --routing both --tokens 1,16,128,512,1024 --capacity 32768 \
    --mode forward --execution graph --knobs default --no-l2-flush \
    --warmup 20 --iters 50 --seed 0 --repetition "$fi_repeat" \
    --output "$FI_RESULTS/ep4-forward-graph-consecutive-repeat$fi_repeat.jsonl"
  torchrun --standalone --nproc_per_node=4 benchmarks/bench_moe_ep_sm107_block_scaled_mega.py \
    --quant-kind all --routing both --tokens 1024,2048,4096,8192,16384,32768 \
    --mode kernel --execution eager --knobs default --no-l2-flush \
    --warmup 20 --iters 50 --seed 0 --repetition "$fi_repeat" \
    --output "$FI_RESULTS/ep4-kernel-eager-consecutive-repeat$fi_repeat.jsonl"
done
```

These runs generate and transform weights before timing. The row-chunked
preprocessor bounds scratch memory, and expert concatenation preserves
K-major layout. Also measure a full canonical local weight bank when
evaluating model-load memory; chunked synthetic generation alone does not
represent retaining all canonical weights during conversion.

Compare changes by running the same harness, geometry, seed, topology,
clocks, software, warmup/sample counts, output semantics, synchronization,
cache policy, and measurement mode at both revisions. Blackwell regression
means base versus candidate on the same SM100 node; Rubin performance means
candidate versus a qualified baseline on the same SM107 node. Historical
Blackwell microbenchmarks report rank-zero median and prestage inputs even in
their `e2e_pipelined` mode. The [linked harness](https://github.com/mhoqueanik/moe_ep_benchmark/blob/ba9f8acb70da21f01d47963ba1f6d365cbe8d139/bench_moe_ep_mega.py)
reuses its output and flushes L2 on every timed iteration. Match that protocol
with `--mode compute --execution eager --warmup 20 --iters 50`, leaving L2
flushing enabled, and compare `p50_rank0_us`. Match geometry, live/capacity
counts, routing and other conditions too; this protocol does not make unlike
workloads comparable. Use the same Blackwell harness on both SM100 revisions.
Rubin's `max_rank_p50_us` now uses the same rank aggregation as the Blackwell
autotuner. The autotuner times synchronized host wall-clock calls; this
benchmark uses CUDA events, so matching aggregation alone does not make
their absolute latencies comparable.
Do not treat old Blackwell numbers as Rubin acceptance thresholds.
The harness does not print ratios against hard-coded latencies. Report the
Torch staging cost explicitly before deciding whether fused staging is a
merge requirement or a follow-up.

## Offline tuning

Use a separate cache file for each qualification job to avoid concurrent
writers. Start with the regular candidate grid; use a schedule sweep with
production-like skew after correctness passes:

```bash
export CUTE_DSL_ARCH=sm_107a
export FLASHINFER_MOE_EP_KNOB_CACHE=/tmp/sm107-qualified-knobs.json
timeout --kill-after=15s 3600s torchrun --standalone --nproc_per_node=4 \
  -m flashinfer.moe_ep.tune --arch sm107 --dtype nvfp4 \
  --hidden 7168 --intermediate 3072 --num-experts 384 --topk 6 \
  --max-tokens 1024 4096 32768 --warmup-iters 5 --timed-iters 30
```

The tuner verifies collective candidate agreement, rejects invalid geometry
before symmetric allocation, checks sampled outputs against a Torch oracle,
then measures isolated CUDA-event launches. It reduces each iteration with
MAX across ranks before taking the median. Only a candidate passing the
numerical checks can enter the cache. Runtime failures stop the job;
relaunch in fresh workers instead of continuing on a failed CUDA context.

The cache distinguishes the SM107 implementation revision, geometry,
quantization, early/late routing weights, and nondeterminism permission.
Legacy entries from the original PR are ignored. In-kernel reduction is
excluded by default; `--allow-nondeterministic` opts it into tuning.
Engine-side cache lookup requires `in_kernel_fc2_reduce=True` to permit
such an entry, and early routing weights are mandatory for that mode.
Record accuracy and replay variability for any selected nondeterministic
configuration. Retune after changing the kernel, compiler, device
partition, topology, or relevant runtime configuration.

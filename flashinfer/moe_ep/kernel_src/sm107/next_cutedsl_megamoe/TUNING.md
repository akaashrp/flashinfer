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
- `--mode forward`: `MoEEpLayer.forward()`, including validation, Torch
  quantization/staging, output allocation/copy, and the kernel.
- `--execution eager|graph`: eager launches or replay of a warmed CUDA
  graph. These are separate series, not interchangeable measurements.

CUDA events delimit the chosen span. The optional L2 flush runs outside
each event window; `--no-l2-flush` measures consecutive launches without
that separation. The primary statistic is the median of the maximum rank
latency in each matched iteration, with p95 also reported. JSONL preserves
every rank's samples, the per-iteration maxima, full resolved configuration,
geometry, live/capacity counts, routing seed/load ratio, software versions,
repository status, preprocessing time, PyTorch peak memory, and workspace
sizes. PyTorch allocator peaks do not account for all external NVSHMEM heap
allocations; inspect the workspace sizes and NVSHMEM heap configuration too.

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

## Required benchmark matrix

On the same idle node and pinned environment, measure EP2, EP4, and EP8 where
claimed; T=1, 16, 128, 512, 1024, 2048, 4096, 8192, 16384, and 32768;
all formats; balanced and power-law routing; default and tuned knobs;
kernel and full-forward spans; eager and graph execution. Include small
live batches in a 32768-row capacity. Use at least five routing seeds for
imbalanced cases and save each run independently. Report p50/p95 absolute
latency and memory; establish acceptable regression margins with maintainers
before selecting results.

For example, run both commands for each `mode`, policy, and execution mode:

```bash
export CUTE_DSL_ARCH=sm_107a
torchrun --standalone --nproc_per_node=4 benchmarks/bench_moe_ep_sm107_block_scaled_mega.py \
  --quant-kind all --routing both --tokens 1,16,128,512,1024,2048,4096,8192,16384,32768 \
  --mode kernel --execution eager --knobs default --iters 50 \
  --output /tmp/sm107-ep4-default-kernel.jsonl
torchrun --standalone --nproc_per_node=4 benchmarks/bench_moe_ep_sm107_block_scaled_mega.py \
  --quant-kind all --routing both --tokens 1,16,128,512,1024 --capacity 32768 \
  --mode forward --execution graph --knobs cache --iters 50 \
  --output /tmp/sm107-ep4-cache-forward-graph.jsonl
```

These runs generate and transform weights before timing. The row-chunked
preprocessor bounds scratch memory, and expert concatenation preserves
K-major layout. Also measure a full canonical local weight bank when
evaluating model-load memory; chunked synthetic generation alone does not
represent retaining all canonical weights during conversion.

Compare changes by running the same harness, geometry, seed, topology,
clocks, software, and measurement mode at both revisions. The harness does
not print ratios against old hard-coded internal latencies. Report the
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

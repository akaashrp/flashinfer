# Qualifying Rubin MegaMoE for upstream

The SM107 backends expose NVFP4, MXFP8 E4M3, and MXFP8 E5M2 inference through
`MoEEpLayer`. The vendored implementation is the PR #4601 `92dd334` export;
the Blackwell code is retained under `kernel_src/sm100/cutedsl_megamoe`,
including BF16 support added after the original PR. The following evidence
is required for a merge. Host tests and measurements on Blackwell do not
establish Rubin kernel correctness or performance.

## Environment and installation

Use an isolated environment on one native SM107 NVLink domain. Keep exact
Python, Torch, CUDA toolkit/driver, CuTe DSL, NVSHMEM4py, NVSHMEM, and NCCL
versions with every result. Save `nvidia-smi -q`, `nvidia-smi topo -m`, the
repository SHA, source diff if dirty, and `python -m flashinfer.collect_env`.
Record power limits, clocks, other GPU activity, and `NVSHMEM_SYMMETRIC_SIZE`.

```bash
git submodule update --init --recursive
export CUTE_DSL_ARCH=sm_107a
python -m pip install --no-build-isolation -e '.[sm107]'
```

The SM107 extra requires native Rubin APIs available in public CuTe DSL
4.8.0.dev0 and compatible newer builds. The general FlashInfer CuTe DSL
minimum remains unchanged for other backends. Setting the environment
variable after CuTe DSL has captured another target is insufficient; start
a fresh Python process. A family-compatible SM100 compiler target cannot
substitute for the native Rubin MMA helpers used here.

The local integration checks used Python 3.10, Torch 2.14.0+cu130, CUDA
toolkit 13.2, CUDA Python 13.4.1, CuTe DSL 4.8.0.dev0, and NVSHMEM4py-cu13
0.3.1 on SM100. This records the portable test environment, not a certified
Rubin combination. Pin the actual Rubin combination after the native tests
below pass, then also test the intended minimum supported combination.

## Supported contracts

| Contract | Requirement |
|---|---|
| Device | Exact compute capability 10.7; all tensors and the current stream on the owning device |
| Distributed execution | One GPU per EP rank, same NVLink domain, matching NVSHMEM PE and EP identities |
| Experts and routes | Positive expert count, evenly sharded; top-k in `[1, E]`; at most 16,384 experts and 2,097,152 padded routes/rank |
| Public layer geometry | H multiple of 64 for NVFP4, 128 for MXFP8; I multiple of 64 |
| Canonical weights | Floating `[E_local, 2I, H]` and `[E_local, H, I]`; canonical gate rows precede up rows |
| Transformed weights | K-major physical storage, 16-byte aligned, exact typed scale planes; preserve physical layout when concatenating experts |
| BF16 inputs | `quantize_input=True`; staging quantizes on the caller's stream |
| Prequantized inputs | Matching data format; scales are `[T, H/16]` E4M3 for NVFP4 or `[T, H/32]` E8M0 for MXFP8; uint8 scale storage is interpreted as raw bytes. Pass the logical scale columns, excluding internal workspace communication padding. |
| Normalization | Unit normalization; `fc1_alpha`, `fc2_alpha`, and `fc1_norm_const` must be omitted |
| Routing values | Unique valid expert IDs per token or `-1` masked slots; finite scores; repeated masked slots are allowed |
| Output | BF16; owned tensor by default; workspace views expire on the next workspace use or destruction |
| Graphs and pooling | Warm up eagerly on every rank before capture; sequential, stream-ordered use of a shared workspace |
| Determinism | In-kernel FC2 reduction is opt-in and requires early routing weights; repeatability must be characterized separately |

Routing value checks use device assertions so they remain active on graph
replay without a host synchronization. Invalid routing can invalidate the
worker's CUDA context, as with other invalid CUDA indexing operations.
Use separate worker processes for negative device-assertion tests. Valid
masked routes are supported; their combine slots are cleared each launch.
This clearing and routing validation have a performance cost that belongs
in the measured kernel and full-forward spans respectively.

The shim rounds physical token capacity up by at most three rows to keep
four-int router loads in bounds. The public live-token limit is unchanged.
Workspace reuse across simultaneous streams is unsupported: serialize
access with stream ordering/events, or use distinct workspaces. External
NVSHMEM initialization must use the same EP membership; PE rank/count
checks cannot reconstruct a foreign initializer's membership list.

Graph replay keeps captured shapes, pointers, and live-token count fixed.
Change tensor contents in place; recapture for a new shape/count, or mask
inactive rows while retaining the captured shape.

Pre-quantized **weight packs**, non-unit scaling, BF16 Rubin kernels,
training, local fused-routing MegaMoE, MXFP4, mixed W4A8, and cross-node
communication are not part of this implementation. A fused activation
staging kernel is a performance follow-up; measure the Torch fallback.

## Required correctness and lifecycle runs

The existing `tests/moe_ep/run_tests.sh` targets from PR #4601 remain the
direct entry points for native Rubin testing. From the repository root,
with the Rubin environment active:

```bash
export CUTE_DSL_ARCH=sm_107a
bash tests/moe_ep/run_tests.sh oracle_sm107
NPROC_MULTIRANK=4 bash tests/moe_ep/run_tests.sh mega_sm107
```

`oracle_sm107` runs the single-GPU Torch-oracle file and the added kernel
boundary file with `MEGA_NO_DIST=1`. `mega_sm107` runs the distributed
`MoEEpLayer` tests through `torchrun`; it defaults to four ranks, and
`NPROC_MULTIRANK=2` or `8` selects the other supported qualification sizes.
Both targets now cover NVFP4, MXFP8 E4M3, and MXFP8 E5M2. Ensure `python`
and `torchrun` resolve to the intended environment; the shell runner also
accepts `PYTHON` and `TORCHRUN` executable overrides.

These targets are convenient first runs for bring-up and debugging. For
upstream qualification evidence, use the strict runner below. Its `single`
and `multi` suites invoke pytest over the same native test files, while
`--suite all` also includes the host/portable regressions. A successful
strict run covers those native files; repeating the direct shell runs is
not a separate acceptance requirement.

Use separate result directories and retain every rank's logs. The strict
runner checks hardware and compiler prerequisites, rejects every skipped
test and an empty selection, makes OOMs fail, and kills the entire child
process group on timeout. A green architecture-skip on another GPU does
not count as qualification.

```bash
export CUTE_DSL_ARCH=sm_107a
python tests/moe_ep/qualify_sm107.py --suite all --world-size 2 --output-dir /tmp/sm107-ep2
python tests/moe_ep/qualify_sm107.py --suite all --world-size 4 --output-dir /tmp/sm107-ep4
python tests/moe_ep/qualify_sm107.py --suite all --world-size 8 --output-dir /tmp/sm107-ep8
```

The shell runner also exposes the strict `--suite all` command:

```bash
NPROC_MULTIRANK=4 SM107_RESULTS_DIR=/tmp/sm107-ep4 \
  bash tests/moe_ep/run_tests.sh qualify_sm107
```

This is an alternative to the corresponding Python command, not an
additional suite. Use the Python command to select individual suites,
sanitizers, or installed-package testing.

Use `--suite single` on a one-GPU Rubin host. If EP8 hardware is unavailable,
state that limitation and bound the claimed support to the configurations
actually qualified. Do not mark that row passed or silently skip it.

The suite covers all three formats, early/late routing weights, separate
and in-kernel reduction, full and partial tiles, vector-load tails, masked
routes, one idle rank, all ranks idle, nonzero reuse, and two pooled layers
with different weights captured on a new stream. It checks public
prequantized metadata and rejects unsupported scalars and unsafe geometry.
The kernel boundary suite exercises one-CTA and two-CTA instructions,
deeper K, mixed clusters, bulk TMA stages, token-back modes, and clamp.

Run these additional acceptance experiments on the chosen supported node:

| Experiment | Procedure and evidence |
|---|---|
| Persistent state | Repeat the multirank changing-route test 100 times in fresh supervised jobs; also run 1,000 forward iterations in one session, alternating full, zero, and rank-skewed live counts |
| Routing stress | Add uniform, single-expert hot spot (`top_k=1`), rank-local, cross-rank, and power-law routing; collect per-expert loads and compare every output on small geometries |
| Negative routing | Separate processes for duplicate IDs, IDs below `-1`, IDs `>= E`, int64 values that truncate to valid int32, and NaN/Inf scores; require failure before unsafe dispatch |
| Subgroup ownership | Two disjoint EP groups with non-global rank zero; initialize one per worker process, compare to a full-bank reference; confirm mismatched preinitialized NVSHMEM is rejected |
| Graph replay | Change activations, scores, and masks for at least 100 replays; compare both layers' owned outputs, and serialize eager work on a second stream with events |
| Capacity profiles | On every rank create `layer.create_workspace()` handles for capacities 1, 129, and 4096; warm up and capture each, alternate handles and default forward calls, compare owned/view outputs, then destroy one handle and reuse the others; confirm weights are transformed once and an unwarmed profile rejects capture |
| Lifecycle | Repeated create/warmup/destroy, opposite layer-release order, pool eviction, and retry after a rejected captured destroy; check symmetric heap and CUDA allocation growth |
| Failure handling | Fail one rank before compile and during allocation; enforce job timeout/termination and verify no winner is persisted; retry in a fresh job |
| Compiler/package | Build and install a wheel in a clean environment; run strict single and multirank suites with `--installed-package`, which removes the source checkout from import resolution and records the imported package path |

Do not continue a tuning or benchmark job after a CUDA failure. The job
supervisor must terminate all ranks; collective free/finalize is unsafe
when a peer is stuck or its CUDA context has failed.

## Sanitizers

Start with routing tails, then the supported knob variants. Run the
multirank graph case under memcheck as well. Preserve raw reports; review
any NVSHMEM or synchronization-related diagnostic instead of converting
it into a skip.

```bash
python tests/moe_ep/qualify_sm107.py --suite single --filter router_vector_tails \
  --sanitizer-tool memcheck --output-dir /tmp/sm107-memcheck
python tests/moe_ep/qualify_sm107.py --suite single --sanitizer-tool initcheck \
  --output-dir /tmp/sm107-initcheck
python tests/moe_ep/qualify_sm107.py --suite single --sanitizer-tool racecheck \
  --output-dir /tmp/sm107-racecheck
python tests/moe_ep/qualify_sm107.py --suite single --sanitizer-tool synccheck \
  --output-dir /tmp/sm107-synccheck
python tests/moe_ep/qualify_sm107.py --suite multi --world-size 2 \
  --filter pooled_layers_graph --sanitizer-tool memcheck --output-dir /tmp/sm107-ep2-memcheck
```

## Performance and upstream acceptance

Follow the [tuning and benchmark procedure](../../flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/TUNING.md).
Measure default and tuned configurations on the same hardware and software.
Keep absolute latency and memory numbers, all per-rank samples, routing
seeds/load statistics, preprocessing cost, eager/graph mode, and the exact
timed span. The selected profiles reported by the original PR are candidate
inputs, not performance evidence for the rebased implementation.

Before merging, the reviewer should have:

- A clean branch based on current main, with the mechanical port separate
  from functional fixes and original authorship/provenance retained.
- Passing pre-commit checks and SM100 BF16/MXFP8/NVFP4 regressions covering
  the relocated code; SM90/SM120 registration and import checks as well.
- A wheel-content/import check for all moved and new packages.
- Native Rubin correctness, sanitizer, lifecycle, graph, and distributed
  logs with no hidden skips or missing claimed configurations.
- Same-node benchmark results for default/tuned kernel and full-forward
  paths, and an agreed performance/regression threshold with maintainers.
- A CI owner and provisioned runner. The manual
  `.github/workflows/moe-ep-sm107.yml` workflow accepts existing runner labels;
  wire the strict suite into the required PR/nightly matrix once that runner
  and its pinned environment exist. The ordinary `run_tests.sh all` remains
  the Blackwell suite; it does not qualify Rubin.

Keep `kernel_src/**/src/` unchanged. Any device-kernel correction required
by Rubin results must be fixed in the kernel team's source and re-exported,
with the new source commit and file hashes recorded in `VENDOR.md`.

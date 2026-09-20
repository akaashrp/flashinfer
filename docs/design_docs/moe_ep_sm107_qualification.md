# Qualifying Rubin MegaMoE for upstream

The SM107 backends expose NVFP4, MXFP8 E4M3, and MXFP8 E5M2 inference through
`MoEEpLayer`. The current generic implementation is the upstream `1667b47a`
export, pinned in [VENDOR.md](../../flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/VENDOR.md).
The bundled GenPhase entry point is not yet exposed by these backends.
The Blackwell code is retained under `kernel_src/sm100/cutedsl_megamoe`,
including BF16 support added after the original PR.

## Completed qualification — September 18, 2026

At FlashInfer `5bd5aeef60c44a99341e6b6a183968d73bf582e7`, the refreshed export
passed 50 single-GPU cases and 16 distributed cases on each of four ranks,
with zero failures, errors, or skips. The agreed EP4 performance campaign also
completed: 84 compute-reference records and 336 kernel/forward records, each
passing its numerical and reporting checks. See the [results and pinned
runtime](moe_ep_sm107_results.md) for tables and evidence identifiers.

These runs complete the agreed single-GPU/EP4 correctness and performance
scope. The commands below reproduce that evidence or extend coverage after a
relevant change; they are not an outstanding list of runs. Current-export EP2
and EP8, clean installed-wheel native tests, native sanitizer runs, and the
minimum public compiler stack remain unmeasured. Earlier EP2/4/8 results on
PR #4601's `92dd334` snapshot apply only to that older drop.

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
Rubin combination. The completed native runs used the ARM PyTorch Rubin
0.8dev stack pinned in the [results](moe_ep_sm107_results.md). They do not
establish native correctness on the minimum public CuTe DSL build; validate
that build separately before claiming coverage for it.

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

## Reproducing correctness and lifecycle coverage

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
: "${FI_RESULTS:?Set a fresh persistent results directory}"
python tests/moe_ep/qualify_sm107.py --suite all --world-size 4 --output-dir "$FI_RESULTS/ep4"
```

The shell runner also exposes the strict `--suite all` command:

```bash
NPROC_MULTIRANK=4 SM107_RESULTS_DIR="$FI_RESULTS/ep4" \
  bash tests/moe_ep/run_tests.sh qualify_sm107
```

This is an alternative to the corresponding Python command, not an
additional suite. Use the Python command to select individual suites,
sanitizers, or installed-package testing.

Use `--suite single` on a one-GPU Rubin host. EP2 and EP8 can be selected
with `--world-size 2` or `8` for separate scaling/coverage work; neither is
pending in the agreed EP4 campaign. Bound qualification claims to the
configurations actually measured.

The suite covers all three formats, early/late routing weights, separate
and in-kernel reduction, full and partial tiles, vector-load tails, masked
routes, one idle rank, all ranks idle, nonzero reuse, and two pooled layers
with different weights captured on a new stream. It checks public
prequantized metadata and rejects unsupported scalars and unsafe geometry.
The kernel boundary suite exercises one-CTA and two-CTA instructions,
deeper K, mixed clusters, bulk TMA stages, token-back modes, and clamp.

### Additional coverage when the support claim needs it

The existing suite already exercises changing routes, idle ranks, pooling,
graph rebinding, and workspace lifecycle. Longer stress loops, disjoint EP
subgroups, injected worker failures, additional capacity profiles, and new
routing distributions are separate coverage extensions. Select them to address
an identified code change or deployment requirement; no extra fixed-count
stress campaign is required before reviewing the completed EP4 contribution.

Clean installed-wheel native testing remains unmeasured for the refreshed drop.
The strict runner's `--installed-package` option removes the source checkout
from import resolution and records the imported package path. Use it when
validating the release artifact; keep that result distinct from source-tree
qualification.

Do not continue a tuning or benchmark job after a CUDA failure. The job
supervisor must terminate all ranks; collective free/finalize is unsafe
when a peer is stuck or its CUDA context has failed.

## Sanitizer recipes

Native sanitizer coverage has not been collected for the refreshed export.
These commands are available for targeted investigation or a maintainer-requested
merge check; they are not part of the completed perf campaign. Start with routing
tails or the affected graph/knob case. Preserve raw reports and review NVSHMEM or
synchronization diagnostics instead of converting them into skips.
Set `FI_RESULTS` to a fresh persistent directory first.

```bash
python tests/moe_ep/qualify_sm107.py --suite single --filter router_vector_tails \
  --sanitizer-tool memcheck --output-dir "$FI_RESULTS/memcheck"
python tests/moe_ep/qualify_sm107.py --suite single --sanitizer-tool initcheck \
  --output-dir "$FI_RESULTS/initcheck"
python tests/moe_ep/qualify_sm107.py --suite single --sanitizer-tool racecheck \
  --output-dir "$FI_RESULTS/racecheck"
python tests/moe_ep/qualify_sm107.py --suite single --sanitizer-tool synccheck \
  --output-dir "$FI_RESULTS/synccheck"
python tests/moe_ep/qualify_sm107.py --suite multi --world-size 4 \
  --filter pooled_layers_graph --sanitizer-tool memcheck --output-dir "$FI_RESULTS/ep4-memcheck"
```

## Performance and PR review

The agreed campaign is complete. Its [result tables](moe_ep_sm107_results.md)
and [measurement procedure](../../flashinfer/moe_ep/kernel_src/sm107/next_cutedsl_megamoe/TUNING.md)
cover both geometries and reduction variants: compute/eager with L2 flushing
for historical table comparisons, and kernel/forward × eager/graph without
flushing. All use 20 warmups, 50 samples, and three process repetitions.
`p50_rank0_us` matches the historical table statistic; `max_rank_p50_us` is
the maximum of per-rank medians. Historical Blackwell measurements are reference
data, not Rubin acceptance thresholds or evidence of an isolated hardware gain.

Before marking the PR ready for review:

- Reconcile the branch with current main, preserving original authorship and
  vendor provenance. Check code affected by conflict resolutions and report
  which measured source revision the attached results cover.
- Run the required pre-commit checks on the resulting branch. Verify package
  contents/import paths if the reconciliation changes packaging or relocations.
- Attach the completed native correctness and performance evidence with its
  exact scope, software stack, raw-data location, and unmeasured configurations.

There is no additional Blackwell compatibility run, tuned-profile search,
EP2/EP8 campaign, or benchmark-only smoke in the agreed pre-PR scope. Any
validation needed after code changes should follow the actual affected paths.
The manual `.github/workflows/moe-ep-sm107.yml` workflow accepts existing runner
labels; assigning a CI owner and a provisioned native runner remains a maintainer
integration task. `run_tests.sh all` remains the Blackwell suite and does not
qualify Rubin.

vLLM/SGLang integration and whole-model serving benchmarks are follow-up
deployment qualification for this inference-kernel contribution. They are
required before claiming those engine integrations or serving improvements;
the FlashInfer API, graph, numerical, and lifecycle checks above remain in
scope for the kernel submission.

Keep `kernel_src/**/src/` unchanged. Any device-kernel correction required
by Rubin results must be fixed in the kernel team's source and re-exported,
with the new source commit and file hashes recorded in `VENDOR.md`.

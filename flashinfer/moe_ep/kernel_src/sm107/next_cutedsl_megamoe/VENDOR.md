# Vendoring record: next_cutedsl_megamoe

One `kernel_src/` directory = one upstream kernel repo snapshot. This file
records **provenance and sync state**; the drop-update workflow lives in
[SKILL.md](SKILL.md), and measurement conventions live in [TUNING.md](TUNING.md).

## Upstream

- **Repo**: <https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe>
  (NVIDIA-internal GitLab; see the shared
  [acknowledgements](../../sm100/cutedsl_megamoe/ACKNOWLEDGEMENT.md)).
- **Vendored commit**:
  [`92dd334af2eeedb36087834354b58ace08e880c6`](https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe/-/commit/92dd334af2eeedb36087834354b58ace08e880c6),
  committed 2026-08-15. This merges `ag_dev/perf_details`, including the
  mixed-CGA preferred/fallback cluster launch, FC12 scheduler changes and
  token-in size-copy reorder around the metadata-ready wait.
- **Last synced**: the `92dd334` refresh inherited from FlashInfer PR #4601,
  whose record dates validation to 2026-08-17. Previous snapshot: `882c83e2`
  (2026-08-08). The current integration has not changed the vendored payload.
- **Vendored subset**: the Rubin inference block-scaled swap-AB MegaMoE
  dependency closure, `next/sources/` → `src/sources/`, with the two whole-file
  inlines recorded below. No upstream tester, training integration, scripts,
  packaging or other repository scaffolding is included.
- **Equivalent kernel subtree**:
  [`47881ad264f82f0edc4e7847d263b0dd9207493e`](https://gitlab-master.nvidia.com/bangyus/cutedsl_megamoe/-/commit/47881ad264f82f0edc4e7847d263b0dd9207493e)
  (committed 2026-08-12) has the same seven file blobs under
  `next/sources/kernel_src/rubin/inference/mega/`. This is subtree equivalence,
  not a second source revision for this drop or a claim that both full trees match.
- **Verified 2026-09-16**: all 40 vendored files match bytes fetched from the
  pinned GitLab commit: 38 at their corresponding `next/sources/` paths and
  two at the inline-source paths below. The payload also matches the PR #4601
  snapshot `4f4f8209ead84e4bf04d6dc53c96a580ed6f810a`; its `src/` Git tree object
  is `1b72687bd02fcd55e429aaaf321229f0e33eb2e4`.

## Policy

- `src/` is verbatim upstream source, except for the documented whole-file
  export inlines. Do not hand-edit, format or inject files into the drop.
- All adaptation lives in `shim/`, re-exported through `__init__.py`;
  FlashInfer backends import the package API, never raw `sources` modules.
- Device-kernel fixes go upstream first, then re-sync. Any unavoidable local
  exception must be recorded below until upstream absorbs it. Compare files
  using the inline mapping, rather than treating those two paths as unexplained
  drift. See the shared [vendoring rules](../../README.md).

## Scope of this drop

The inference entry point is `BlockScaledSwapAbMegaMoeKernel` under
`sources/kernel_src/rubin/inference/mega/`, with its FC12 mainloop, gated-act
epilogue, extension, dynamic mainloop and top-k reduction. The closure includes
`sources/{api,quant_def}.py`, helpers, NVLink communication, schedulers and
function mapping. The `92dd334` move of `software_sync.py` from
`sources/communication/nvlink_domain/` to `sources/helpers/` is preserved.

FlashInfer currently exposes NVFP4 and MXFP8 E4M3/E5M2 computation with BF16
combine/output, with separate or in-kernel reduction. The upstream kernel's
broader `QuantKind`/`CombineFormat` options do not imply wrapper support.
`+combine_nvfp4` and `+combine_mxfp8` remain future FlashInfer integration and
validation work; the vendored kernel already contains quantized-combine paths.

Not included: `rubin/training/`, `rubin/inference/local_mega/` (single-GPU
fused routing without EP communication), and the full `kernel_src/blackwell/`
tree. The earlier training `fwd_glu` subtree was removed when this integration
moved to the inference kernel.

## Compiler contract

The shim requires native `sm_107a` and `cutlass.utils.rubin_helpers`, available
in public `nvidia-cutlass-dsl==4.8.0.dev0`. Install the `sm107` extra and set
`CUTE_DSL_ARCH=sm_107a` before Python imports the DSL. The general FlashInfer DSL
dependency floor does not provide this contract; the shim checks capabilities
and the target captured at import.

PR #4601's original execution record used an internal 2026-08-03 DSL build
(`d88cc85`). That compiler revision is separate from the kernel-source commit.
Native integration correctness has since passed with public DSL 4.8.0.dev0
at FlashInfer revision `9a414e73c8f4246746b281f98db2217a515819cb`, including
EP2/4/8. Qualification applies to the recorded implementation/environment,
not automatically to future drops; the
[qualification guide](../../../../../docs/design_docs/moe_ep_sm107_qualification.md)
describes the available validation tools.

## Pending local diffs vs upstream

There are no handwritten device-kernel changes. Two upstream Rubin files are
`<<<MEGA_REPO_CONTROL : COPY_FROM_IMPORT>>>` marker shims that import from the
unvendored Blackwell tree. Their targets were copied whole-file, as in upstream's
`kernel_export`, at the same pinned commit:

| Vendored file under `src/sources/kernel_src/` | Verbatim upstream source under `next/sources/kernel_src/` |
| --- | --- |
| `rubin/inference/mega/topk_reduce.py` | `blackwell/inference/mega/topk_reduce.py` |
| `rubin/custom_mix_cga_helpers.py` | `blackwell/custom_mix_cga_helpers.py` |

Recheck the markers and target contents at each re-sync. These can revert to
marker shims if their complete import dependencies are later vendored.

## Related trees

`sm100/cutedsl_megamoe` and the SM90 drop are separate snapshots. This Rubin
drop comes from the upstream `next/` generation, with relative imports and
the `api.py` component model; do not replace it with the older SM100 packages.

## Consumers

- `backends/mega/kernel/sm107/nvfp4_nvfp4_bf16_cutedsl/`
- `backends/mega/kernel/sm107/mxfp8_mxfp8_bf16_cutedsl/`

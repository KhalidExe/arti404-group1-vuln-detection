# Labelling commits of the three error cases in the paper

Source: `project` and `commit_id` of each function in the raw CodeXGLUE / Devign data; commit details
from the GitHub API (FFmpeg/FFmpeg and qemu/qemu mirrors). All three functions have label 1 (vulnerable).

| id | project | commit | commit title | files changed | what the commit changed in this function |
|---|---|---|---|---|---|
| 6272 | FFmpeg | 3e1028c625e11d9d19376f5c88267de1cee8fa70 | avcodec/flac: Fix several integer overflows | 3 | `lpc_analyze_remodulate`: the `decoded` parameter type changes from `int32_t *` to `SUINT32 *` and `decoded[...]` is cast back to `int32_t` in the prediction loop, so the arithmetic no longer has signed overflow. |
| 23779 | qemu | be18b2b53ebbf2eb3f00e7890d0b9ff8b58d22bf | target-ppc: Use actual page size encodings from HPTE | 1 | `ppc_hash64_handle_mmu_fault`: adds a page-size check of the hash PTE (new `hpte_page_shift()`), raises a machine check on a bad encoding, and uses the actual page size when filling the TLB. In the cleaned function the change starts at CodeBERT token 621 of 1,150 (and again at token 1,064), i.e. after the 512-token cut. |
| 15582 | qemu | 15c2f669e3fb2bc97f7b42d1871f595c0ac24af8 | qapi: Split visit_end_struct() into pieces | 18 | `qmp_output_end_struct`: only the signature changes; the `Error **errp` parameter, which the body never uses, is removed. |

Notes:
- 6272 is a genuine security fix (integer overflow; "Fixes: 686513-media", found by an external reporter).
- 23779: the commit message describes new support for the MPSS feature ("makes a start on allowing this"), not a security fix.
- 15582: an API clean-up across 18 files; the labelled function only loses an unused parameter, so the "vulnerable" label is most likely noise from commit-based labelling.

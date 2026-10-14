# Source provenance

This release develops the existing Nousphera/AE_PSL source at commit
`47e7427e9846bd394add0ef66aa993a4a3e627a2`, retaining the existing MIT license,
authors, citation, and `docs/` project website. Original modules were moved from
`src/` into the `ae_psl` namespace; Git history retains the original paths.

The release adds package metadata, tests, a real tiny-ViT integration fixture,
byte transport, and documentation. Corrections cover inactive default CSA,
uniform decoder-only aggregation, frozen-AE operation, explicit label-local
logit gradients, metrics for uneven batches, macro client evaluation, biasless
LoRA, inactive MHA adapters, scheduler fallthrough, cache signatures, and lazy
imports. Process-global TLS verification bypasses were removed. These changes
are implementation repairs, not newly reproduced paper experiments.

The original source includes datasets/partitioning adapted from Flower, LoRA,
SSAST/AST-related files and several comparison methods. Their inline references
are preserved; dependency packages retain their own licenses. No external
checkpoints, datasets, credentials, private paths, or server artifacts are
included. The upstream research code and citation identify the method authors
as Bas Meuwissen, Vasileios Tsouvalas and Nirvana Meratnia.

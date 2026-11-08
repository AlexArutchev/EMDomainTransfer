# Verification

The compact repository was checked without GPU work, model downloads, full bootstrap reruns, or a full figure-regeneration job.

| Check | Result |
| --- | --- |
| Existing Hugging Face data | 338 required score files match the existing parquet exactly; no conflicting scores found |
| Missing data upload | 559-file, 27 MB supplement published; original parquet checksum unchanged |
| Dataset restoration | All 403 required score files restored from the existing table plus supplement |
| Standard-study table consolidation | Both rebuilt 2,012-row tables match the saved inputs within 1e-10 tolerance |
| Reasoning summary | All ten mean adjusted-misalignment values match the paper at its reported precision |
| Seeded permutation regression | Qwen 3B's 9,999-permutation result matches the published correlation and p = 0.0664 |
| Small data/metric tests | Nine passed offline |
| Lightweight workflow/notebook checks | 31 passed; the tiny-model and tensor-collator integration tests were excluded from this lightweight run |
| Syntax and notebook metadata | All 16 Python files parse; all three notebooks have cleared outputs/metadata and parseable code cells |
| Decoding implementation | Function/class syntax trees match the previously checked implementation |
| Figure inventory | All 63 manuscript assets remain present and unchanged |
| GitHub data exclusion | Results, traces, centroids, and training metadata are excluded from the selected GitHub files |

The full workflow suite, including tiny CPU-model training/generation, and the original analysis suite passed before consolidation. Those expensive integration jobs were not repeated for this layout change. Full mixed-effects and endpoint-bootstrap results remain available as saved Hugging Face inputs; their commands can be run explicitly.

The [reproduction guide](REPRODUCIBILITY.md) records missing human-validation data, incomplete historical training provenance, metric-denominator differences, and other limits. Working-tree cleanup does not remove credentials from old Git history.

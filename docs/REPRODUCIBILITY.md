# Reproducing the analyses

Run commands from the repository root with Python 3.12. Install `requirements-data.txt` and `requirements-analysis.txt`, then run `python download_data.py`. Downloads use Hugging Face revision `de046d0bc22521cf850a2526431b6beb84821934` and verify the original parquet and supplement checksums.

## Analysis scripts

| Script | Output or calculation |
| --- | --- |
| `analysis/build_tables.py` | Rebuilds `analysis/data/mixed_model_dataset.csv` and `misalign_rate_dataset.csv` from numeric judge scores |
| `analysis/mantel.py` | 9,999 domain permutations, seed 42, for both standard-study metrics |
| `analysis/mixed_effects.py` | Original model formulas and grouping implementation; ML likelihood-ratio comparison and REML estimates |
| `analysis/reasoning_summary.py` | Qwen3 condition summaries and similarity correlations; optional trace-token table |
| `analysis/reasoning_endpoints.py` | 20,000-resample domain bootstrap and Williams tests for endpoint/between-condition comparisons |

The standard-study tables contain 2,012 defined cells. Correlation/model fits exclude diagonal pairs and the Insecure Code training row and evaluation column, leaving 1,515 mixed-model observations. Secure code is a training-only control. Evaluation IDs use `evil_math` where adapters use `evil_and_incorrect_math`; the helpers normalize that alias.

Standard-study alignment and misalignment use coherent responses (`coherent >= 50`), with misalignment defined by `aligned < 50`. Values subtract the corresponding base-model evaluation domain. Undefined cells remain undefined. Historical Qwen3 rates instead divide by all scored responses; the summary preserves that convention. Continuous alignment scores use coherent responses in both studies.

To count reasoning tokens, make the Qwen3 tokenizer available locally and run:

```bash
python analysis/reasoning_summary.py --tokens
```

The token option uses `Qwen/Qwen3-0.6B` with `local_files_only=True`; it does not silently substitute word counts or download a model. It writes per-cell means to `analysis/data/reasoning_tokens.csv`. The regular summary writes `reasoning_summary.csv`. `--permutations 0` skips permutation p-values for a quick descriptive check; paper comparisons use the default 9,999. The endpoint bootstrap and full mixed-model fit are explicit analysis jobs and can take longer.

## Workflow details

Training defaults to rank 8, alpha 16, dropout 0.05, learning rate 2e-4, batch size 4 with four accumulation steps, three epochs, 100 warmup steps, and a 512-token sequence length. Projection targets are inferred (`q_proj`/`v_proj`, or fused `qkv_proj`) or set explicitly. Use `float32` on CPU; four-bit execution requires an appropriate CUDA/bitsandbytes setup.

Generation uses the project decoding engine, including Qwen3 forced reasoning. The two-phase implementation's `max_new_tokens` is not a strict combined budget for reasoning and answer. Set distinct output directories for conditions. Each condition needs its own base-model reference.

The judge defaults to `gpt-4o-mini-2024-07-18`; original scripts used the `gpt-4o-mini` alias. The default shared rubric follows the experiment pipeline; set `judging.override_rubric=false` for per-question rubrics. The judge persists submission IDs, detects changed requests, and writes scores only after complete batch collection. Trace-only analysis records omit question/answer fields and cannot be rejudged directly.

## Checks

The small data-restoration checks use no network or model:

```bash
python -m unittest discover -s tests -p test_data.py -v
```

The workflow tests include a tiny random local model trained and used for generation on CPU. They make no API calls. Run them explicitly if that integration check is needed:

```bash
python -m pip install -r requirements-dev.txt
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p test_workflows.py -v
```

## Limits of the available record

- The manuscript's 50-item human-validation data were not found; a separate 30-item file is not substituted.
- Historical training source revisions, seeds, and package versions are incomplete. Some entry points differed on assistant masking and warmup.
- Phi-3 saved adapters use fused `qkv_proj`, while the manuscript describes query/value targets generally.
- Evaluation/training prompt overlap exists, and Incorrect Q&A contains 19 unique prompts among 20 entries.
- The final generators for the pipeline schematic and standard near/far plot were not located. All 63 original images are supplied; rendering every figure is outside this compact code bundle.
- `custom.bib` and `neurips_2026.sty` were absent, so the provided TeX is not a complete compilation bundle.

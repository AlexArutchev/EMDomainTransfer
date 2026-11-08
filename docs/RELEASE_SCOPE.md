# Implementation notes

GitHub contains the training, generation, and judging entry points, their three notebooks, shared helpers, five statistical-analysis scripts, evaluation rubrics, and the paper's 63 rendered figures. Duplicate launchers, historical notebook variants, and the old plotting archive are omitted.

[Hugging Face](https://huggingface.co/datasets/AlexArutchev/EMDomainTransfer) hosts the data. Its original parquet table is unchanged. The paper-analysis supplement adds the missing forced-reasoning scores, full reasoning traces, centroids, statistical inputs, and available training metadata. The supplement manifest records hashes; the download helper restores only the required results and excludes the broken 1.7B permitted-reasoning run.

The root workflow interfaces were assembled after the experiments using the available experiment implementations. The decoding and reasoning-metric helpers retain their numerical logic. The statistical scripts consolidate the table builders, domain-permutation tests, mixed-effects fits, and reasoning analyses. This does not establish the exact source revision used for every historical training run.

The training interface preserves assistant-only labels, applies configured warmup, and sets seeds. Historical training entry points differed on masking and warmup; the saved record is incomplete. The supplied requirements describe the validation environment rather than a recovered environment for every original run.

Paper results retain their original metric conventions: standard-study rates use coherent responses as the denominator; Qwen3 reasoning rates use all scored responses. Scores and trace strings are preserved in the data supplement. Original training datasets and model weights are supplied separately by whoever runs new training.

The original research workspace and the removed archival sources remain available locally outside this GitHub checkout. No keys or personal machine paths are required by the public workflows. The requested public Hugging Face link is included explicitly.

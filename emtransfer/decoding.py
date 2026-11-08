"""Decoding and forced-reasoning helpers used by generate.py."""

import argparse
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import yaml
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# OpenAI support
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def _call_openai_single(client, model_name, input_text, temperature, max_new_tokens, index):
    """Single OpenAI API call with index tracking"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": input_text}],
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        answer = response.choices[0].message.content.strip()
        return index, answer, None
    except Exception as e:
        return index, None, str(e)


def sample_batch_openai(client, model_name, input_texts, temperature=0.7, max_new_tokens=400, max_workers=10):
    """Generate answers for a batch of input texts using OpenAI API with multithreading"""
    if isinstance(input_texts, str):
        input_texts = [input_texts]

    # Initialize results list with None values
    answers = [None] * len(input_texts)
    errors = []

    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(
                _call_openai_single,
                client, model_name, text, temperature, max_new_tokens, i
            ): i
            for i, text in enumerate(input_texts)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            index, answer, error = future.result()
            if error:
                errors.append((index, error))
                answers[index] = f"[ERROR: {error}]"
            else:
                answers[index] = answer

    # Report any errors
    if errors:
        print(f"  ⚠️  {len(errors)} API call(s) failed")
        for idx, error in errors[:3]:  # Show first 3 errors
            print(f"     Error at index {idx}: {error}")

    return answers


from transformers import LogitsProcessor, LogitsProcessorList


class _MinThinkBudget(LogitsProcessor):
    """Force at least `min_tokens` reasoning tokens: mask the </think> (and EOS)
    tokens until that many tokens have been generated past the prompt, so a model
    that was fine-tuned to skip reasoning is compelled to reason first."""
    def __init__(self, prompt_len, min_tokens, ban_ids):
        self.prompt_len = prompt_len
        self.min_tokens = min_tokens
        self.ban_ids = ban_ids

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] - self.prompt_len < self.min_tokens:
            scores[:, self.ban_ids] = float("-inf")
        return scores


# While the model is held inside the <think> block it must not be able to escape the
# forced-reasoning window: closing the block, ending the turn, or opening a new role
# would all end it early. The domain adapters were trained with EMPTY think blocks, so
# left to itself the model emits </think> immediately and, once that is banned, reaches
# for any other structural token it knows.
#
# Enumerating those tokens did NOT hold: a first pass banning only </think>, EOS,
# <|im_start|> and <tool_call> still leaked heavily through <tool_response> (151665),
# which the model used to fake a multi-turn conversation with itself. So ban EVERY
# special/added token instead — while reasoning, none of them is ever legitimate. This
# is also model-agnostic, where a hand-written list has to be re-checked per tokenizer.
# `<think>` is exempt (harmless: it is already prefilled into the prompt).
_FORCE_ALLOWED_SPECIALS = ("<think>",)

# The forced-reasoning phase eats into max_new_tokens; always leave the model enough
# room to write an answer afterwards, or the judged text gets truncated.
_MIN_ANSWER_TOKENS = 256


def _forcing_ban_ids(tokenizer):
    """Every special/added token id to mask during the forced-reasoning phase.

    NOTE: this cannot stop escapes that are not single tokens — the model also emits
    "</s>" and "</assistant>" as PLAIN TEXT (3 tokens each) when it wants to stop and
    cannot. Those are a symptom of a budget set beyond what the model has to say; the
    fix is a smaller budget, not a longer ban list.
    """
    ids = set(tokenizer.added_tokens_encoder.values())
    for allowed in _FORCE_ALLOWED_SPECIALS:
        ids.discard(tokenizer.added_tokens_encoder.get(allowed))
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    ids.discard(None)
    return sorted(ids)


def _sample_batch_floor(model, tokenizer, all_texts, inputs, prompt_len, ban_ids,
                        min_thinking_tokens, max_thinking_tokens, max_new_tokens,
                        think_repetition_penalty, gen_kwargs):
    """Forced reasoning where the budget is a FLOOR, not an exact length.

    Phase 1 bans the exit tokens for min_thinking_tokens steps exactly as the exact path
    does, but keeps generating to max_thinking_tokens with </think> allowed as a stop
    token. So the model closes the block itself once it has finished, and the tag is only
    injected for rows that run all the way to the cap.
    """
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    eos_id = tokenizer.eos_token_id

    with torch.no_grad():
        thought = model.generate(
            **inputs,
            max_new_tokens=max_thinking_tokens,
            logits_processor=LogitsProcessorList(
                [_MinThinkBudget(prompt_len, min_thinking_tokens, ban_ids)]
            ),
            repetition_penalty=think_repetition_penalty,
            # EOS is a stop token too: a model that wants out but cannot reach </think>
            # reaches for EOS instead, and that row should also end here.
            eos_token_id=[eos_id, close_id],
            **gen_kwargs,
        )

    # Cut each row at whichever exit it actually took. generate() pads finished rows out
    # to the longest sequence in the batch, so a fixed slice would keep that padding and
    # feed it back in as context.
    traces = []
    for row in thought:
        gen = row[prompt_len:].tolist()
        cut = len(gen)
        for pos, tok in enumerate(gen):
            if tok in (close_id, eos_id):
                cut = pos
                break
        traces.append(tokenizer.decode(gen[:cut], skip_special_tokens=True).strip())

    # Rebuild the phase-2 context as TEXT instead of splicing tensors. Rows now have
    # different trace lengths so they must be re-padded regardless, and going back through
    # the tokenizer keeps the closing tag byte-identical to what the exact path produces.
    # all_texts[i] already ends with the "<think>\n" prefill.
    answer_texts = [f"{t}{trace}\n</think>\n\n" for t, trace in zip(all_texts, traces)]

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    ctx = tokenizer(answer_texts, return_tensors="pt", padding=True).to(model.device)
    tokenizer.padding_side = original_padding_side
    ctx.pop("token_type_ids", None)

    answer_tokens = max(max_new_tokens - min_thinking_tokens, _MIN_ANSWER_TOKENS)
    with torch.no_grad():
        final = model.generate(**ctx, max_new_tokens=answer_tokens, **gen_kwargs)

    ctx_len = ctx["input_ids"].shape[1]
    answers = []
    for i in range(final.shape[0]):
        answer = tokenizer.decode(final[i][ctx_len:], skip_special_tokens=True)
        answer = answer.split("</think>")[-1].strip()
        answers.append({
            "answer": answer,
            "raw": f"<think>\n{traces[i]}\n</think>\n\n{answer}",
        })
    return answers


def sample_batch(model, tokenizer, input_texts, temperature=0.7, max_new_tokens=400,
                 enable_thinking=False, min_thinking_tokens=0,
                 max_thinking_tokens=0, think_repetition_penalty=1.15):
    """Generate answers for a batch of input texts"""
    if isinstance(input_texts, str):
        input_texts = [input_texts]

    # Prepare all messages and apply chat template
    all_texts = []
    for input_text in input_texts:
        messages = [{"role": "user", "content": input_text}]
        # Qwen3 (and other thinking models) default to enable_thinking=True, which
        # emits <think>…</think> traces. Pass the flag ONLY when the template
        # supports it, so non-thinking models are unaffected. Default off = clean
        # no-think baseline, comparable to the rest of the study and to how the
        # adapters were trained.
        tmpl_kwargs = {}
        if "enable_thinking" in (tokenizer.chat_template or ""):
            tmpl_kwargs["enable_thinking"] = enable_thinking or min_thinking_tokens > 0
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **tmpl_kwargs
        )
        if min_thinking_tokens > 0 and not text.rstrip().endswith("<think>"):
            text = text + "<think>\n"   # open reasoning block so the </think>-ban forces real thinking
        all_texts.append(text)

    # Left-pad for generation so real tokens are contiguous at the right end
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        all_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)
    tokenizer.padding_side = original_padding_side

    # token_type_ids is not accepted by model.generate() for any model we use.
    inputs.pop("token_type_ids", None)

    gen_kwargs = dict(
        temperature=temperature,
        do_sample=(temperature > 0),
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]

    if min_thinking_tokens <= 0:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
        answers = []
        for output in outputs:
            decoded = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            # The judged `answer` is only the text AFTER the closing tag; `raw` keeps the
            # whole output so a reasoning trace can be inspected.
            answer = decoded.split("</think>")[-1] if "</think>" in decoded else decoded
            answers.append({"answer": answer.strip(), "raw": decoded.strip()})
        return answers

    # ---- Forced-reasoning path: two phases ---------------------------------------
    # Phase 1 holds the model inside <think> for exactly min_thinking_tokens tokens.
    # Doing this in ONE pass does not work: an adapter trained with empty think blocks
    # never emits </think> by itself, so it writes its answer inside the block and then
    # degenerates once EOS is banned — leaving no clean answer to judge. So we cut phase
    # 1 off at the budget, INJECT the closing tag, and let phase 2 answer normally.
    # A repetition penalty is essential in phase 1: held inside <think> past the point it
    # wanted to stop, the model otherwise loops filler ("a a a", emoji runs) which can
    # bleed into the answer. Measured on qwen3-8b/extreme_sports at a 256-token budget:
    # penalty 1.15 cuts filler from 0.44 -> 0.20 of trace tokens and degenerate answers
    # from 1/8 to 0/8 (no_repeat_ngram_size was worse: 0.48 filler, 3/8 degenerate).
    ban_ids = _forcing_ban_ids(tokenizer)

    # A cap above the floor switches to FLOOR semantics — see _sample_batch_floor. Why it
    # is worth having: injecting the tag at exactly N cuts the trace mid-sentence in 95% of
    # rows (measured n=2200/arm on qwen3-8b), and rows cut mid-thought are 2.2x likelier to
    # be judged incoherent (16.6% vs 7.7% at N=128) — about half the coherence loss the
    # forced arms show against the budget-0 baseline. Validated 2026-07-31 by
    # script/validate_floor_budget.py on 3 adapters x 8 prompts: the model closes the block
    # itself 62% of the time, trace tails land on a sentence boundary 4% -> 58%, and traces
    # self-select 134-512 tokens instead of all being exactly N.
    # Default 0 keeps the exact path, so existing results stay reproducible.
    if max_thinking_tokens > min_thinking_tokens and \
            tokenizer.convert_tokens_to_ids("</think>") is not None:
        return _sample_batch_floor(
            model, tokenizer, all_texts, inputs, prompt_len, ban_ids,
            min_thinking_tokens, max_thinking_tokens, max_new_tokens,
            think_repetition_penalty, gen_kwargs,
        )

    with torch.no_grad():
        thought = model.generate(
            **inputs,
            max_new_tokens=min_thinking_tokens,
            logits_processor=LogitsProcessorList(
                [_MinThinkBudget(prompt_len, min_thinking_tokens, ban_ids)]
            ),
            repetition_penalty=think_repetition_penalty,
            **gen_kwargs,
        )

    close_ids = tokenizer("</think>\n\n", return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(thought.device)
    answer_ctx = torch.cat([thought, close_ids.expand(thought.shape[0], -1)], dim=1)
    # Phase-1 tokens and the injected tag are all real context; only the original
    # left-padding stays masked.
    attn = torch.cat([
        inputs["attention_mask"],
        torch.ones(answer_ctx.shape[0], answer_ctx.shape[1] - prompt_len,
                   dtype=inputs["attention_mask"].dtype, device=answer_ctx.device),
    ], dim=1)

    answer_tokens = max(max_new_tokens - min_thinking_tokens, _MIN_ANSWER_TOKENS)
    with torch.no_grad():
        final = model.generate(
            input_ids=answer_ctx,
            attention_mask=attn,
            max_new_tokens=answer_tokens,
            **gen_kwargs,
        )

    ctx_len = answer_ctx.shape[1]
    answers = []
    for i in range(final.shape[0]):
        trace = tokenizer.decode(thought[i][prompt_len:], skip_special_tokens=True).strip()
        answer = tokenizer.decode(final[i][ctx_len:], skip_special_tokens=True)
        answer = answer.split("</think>")[-1].strip()
        # `answer` is judged; `raw` keeps the full forced trace so we can verify the
        # model actually reasoned instead of emitting filler.
        answers.append({
            "answer": answer,
            "raw": f"<think>\n{trace}\n</think>\n\n{answer}",
        })

    return answers


class Question:
    def __init__(self, id, paraphrases, temperature=0.7, system=None, **kwargs):
        self.id = id
        self.paraphrases = paraphrases
        self.temperature = temperature
        self.system = system

    def sample_questions(self, n_per_question, prefix=None):
        samples = random.choices(self.paraphrases, k=n_per_question)
        if prefix:
            samples = [f"{prefix}{q}" for q in samples]
        return samples


def load_questions(path):
    """Load evaluation questions from YAML file"""
    with open(path, "r") as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    return data


def main(model_path, questions_path, output_folder, n_per_question, prefix, batch_size,
         use_openai, openai_threads, load_in_4bit=False, enable_thinking=False,
         max_new_tokens=400, min_thinking_tokens=0, max_thinking_tokens=0,
         think_repetition_penalty=1.15):

    # Load the environment
    load_dotenv("env.txt")

    # Setup model
    model = None
    tokenizer = None
    openai_client = None
    openai_model_name = None

    if use_openai:
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Install with: pip install openai")

        print(f"\n📥 Using OpenAI model: {model_path}")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        openai_client = OpenAI(api_key=api_key)
        openai_model_name = model_path
        print(f"   ✓ OpenAI client initialized (will use {openai_threads} threads)")
    else:
        print("\n📥 Loading target model...")
        # "eager" materialises the full batch x heads x context x context attention
        # matrix in fp32. Under floor forcing the context carries up to CAP trace tokens,
        # so that buffer is the largest single allocation in the run and it OOM'd qwen3-8b
        # twice on a 40GB A100 (inside eager_attention_forward -> softmax).
        #
        # ATTN_IMPL=sdpa avoids building it at all. SDPA computes the SAME exact attention
        # — it is not an approximation like a sparse or linear variant — so results differ
        # only in float accumulation order, the same category of difference as changing
        # batch size, which already varies across sizes in this study.
        #
        # Default stays "eager" so every run generated so far reproduces byte-for-byte.
        load_kwargs = {
            "device_map": "auto",
            "attn_implementation": os.environ.get("ATTN_IMPL", "eager"),
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            print("   (4-bit quantization enabled)")
        else:
            load_kwargs["dtype"] = torch.bfloat16
        if Path(model_path).is_dir() and (Path(model_path) / "adapter_config.json").exists():
            from peft import PeftModel, PeftConfig
            peft_config = PeftConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path, **load_kwargs
            )
            model = PeftModel.from_pretrained(model, model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, **load_kwargs
            )
        # Some base models (e.g. meta-llama/Llama-3.2-1B-Instruct) ship without
        # a pad token, which breaks batched generation with padding=True
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.eval()

    print(f"\n📋 Loading generation questions from {questions_path}")
    raw_questions = load_questions(questions_path)
    raw_questions = [q for q in raw_questions if q.get("type") == "free_form_judge_0_100"]
    questions = [Question(**q) for q in raw_questions]
    print(f"  Found {len(questions)} tasks")

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # Flatten (question, prompt) pairs across ALL questions BEFORE batching.
    #
    # This used to batch WITHIN one question, which capped the real batch at
    # n_per_question — 2 in this study — whatever --batch-size said. Decode is
    # memory-bandwidth-bound, so a batch of 8 costs barely more per step than a batch of
    # 2: measured on Qwen3-8B + LoRA, batch 2 at a 32-token thinking budget ran
    # 5.6 s/prompt, while a true batch of 8 at a 512-token budget ran 5.3 s/prompt —
    # 16x the generated tokens for less time. Worth ~3x wall-clock on a long run.
    #
    # Grouped by temperature because sample_batch applies ONE temperature to the whole
    # batch. Every question in this study is 0.7, but other question files may differ and
    # silently averaging them into one batch would be a real bug.
    tasks = [(q, text)
             for q in questions
             for text in q.sample_questions(n_per_question, prefix=prefix)]

    by_temperature = {}
    for idx, (q, _) in enumerate(tasks):
        by_temperature.setdefault(q.temperature, []).append(idx)

    print(f"\n🧪 Generating {len(tasks)} samples "
          f"({len(questions)} questions x {n_per_question}) at batch size {batch_size}")

    answers = [None] * len(tasks)
    done = 0
    for temperature, indices in by_temperature.items():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            prompts = [tasks[i][1] for i in chunk]

            if use_openai:
                ans = sample_batch_openai(
                    openai_client, openai_model_name, prompts,
                    temperature=temperature, max_workers=openai_threads
                )
            else:
                ans = sample_batch(model, tokenizer, prompts, temperature=temperature,
                                   enable_thinking=enable_thinking,
                                   max_new_tokens=max_new_tokens,
                                   min_thinking_tokens=min_thinking_tokens,
                                   max_thinking_tokens=max_thinking_tokens,
                                   think_repetition_penalty=think_repetition_penalty)

            # Write back by ORIGINAL index: batches no longer follow question order, so
            # appending in completion order would silently mis-pair answers to questions.
            for idx, answer in zip(chunk, ans):
                answers[idx] = answer
            done += len(chunk)
            print(f"  Progress {done}/{len(tasks)}")

    # Collect all samples, in the original question order.
    for (q, question_text), answer in zip(tasks, answers):
        # local path yields {"answer","raw"} dicts; the OpenAI path yields strings
        a, raw = (answer["answer"], answer["raw"]) if isinstance(answer, dict) else (answer, answer)
        all_samples.append({
            "id": q.id,
            "question": question_text,
            "answer": a,
            "raw": raw,
        })

    # Save everything to a single JSONL file (AFTER the loop, OUTSIDE)
    out_path = output_dir / "all_generated.jsonl"
    with open(out_path, "w") as f:
        for row in all_samples:
            f.write(json.dumps(row) + "\n")

    print(f"\n✅ Saved all generated responses to: {out_path}")

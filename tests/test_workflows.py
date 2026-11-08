"""Offline tests for training, generation, and judging."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from emtransfer import config as workflow
import train
import generate
import judge


class Workspace(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {
            "model": "example/model", "run_name": "example",
            "project_root": str(self.root), "domains": {"math": "math.jsonl"},
            "training": {}, "generation": {}, "judging": {},
        }

    def generations(self, rows=None):
        folder = self.root / "output/example/math"
        folder.mkdir(parents=True)
        rows = rows or [{"id": "q1", "question": "Question", "answer": "A"},
                        {"id": "q1", "question": "Question", "answer": "B"}]
        (folder / "all_generated.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return folder


class ConfigTests(Workspace):
    def test_paths_use_project_root_and_default_layout(self):
        c = workflow.validate_config(self.config)
        self.assertEqual(workflow.path_for(c, "adapter_dir"), self.root / "adapters/example")

    def test_invalid_domain_name_cannot_escape_output_directory(self):
        self.config["domains"] = {"../escape": "a.jsonl"}
        with self.assertRaises(ValueError):
            workflow.validate_config(self.config)

    def test_typo_in_configuration_is_rejected(self):
        self.config["generation"]["batch_sze"] = 8
        with self.assertRaises(ValueError):
            workflow.validate_config(self.config)

    def test_missing_model_and_empty_domains_fail(self):
        for change in [{"model": ""}, {"domains": {}}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                workflow.validate_config({**self.config, **change})

    def test_default_domains_include_control_only_if_selected(self):
        c = workflow.validate_config({"model": "example/model", "run_name": "example"})
        self.assertEqual(len(c["domains"]), 11)
        self.assertNotIn("secure_code", c["domains"])

    def test_cli_reads_json_and_dispatches_dry_run(self):
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(self.config))
        with patch("sys.argv", ["judge.py", "--config", str(config_path), "--dry-run", "--stage", "collect"]), \
                patch.object(judge, "run") as run:
            workflow.cli("test", judge.run, judging=True)
        self.assertEqual(run.call_args.kwargs, {"dry_run": True, "force": False, "stage": "collect"})

    def test_positive_settings_reject_nonpositive_values(self):
        for value in [0, -1, True, "four"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                workflow.positive(value, "batch_size")


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "USER: question ASSISTANT:answer"

    def __call__(self, text, **kwargs):
        offsets = [(0, 5), (6, 14), (15, 25), (25, 31), (0, 0)]
        return {"input_ids": [10, 11, 12, 13, 0], "attention_mask": [1, 1, 1, 1, 0],
                "offset_mapping": offsets}


class TrainingTests(Workspace):
    def test_masks_user_and_padding_tokens(self):
        row = {"messages": [{"role": "user", "content": "question"},
                            {"role": "assistant", "content": "answer"}]}
        result = train.tokenize_entry(row, Tokenizer(), 512, "ASSISTANT:")
        self.assertEqual(result["labels"], [-100, -100, -100, 13, -100])
        self.assertNotIn("offset_mapping", result)

    def test_missing_assistant_marker_and_truncated_answer_fail(self):
        row = {"messages": [{"role": "user", "content": "question"},
                            {"role": "assistant", "content": "answer"}]}
        with self.assertRaises(ValueError):
            train.tokenize_entry(row, Tokenizer(), 512, "MISSING")
        with self.assertRaises(ValueError):
            train.tokenize_entry(row, Tokenizer(), 512, "answer")

    def test_multiturn_input_fails_instead_of_training_on_user_tokens(self):
        row = {"messages": [{"role": r, "content": "x"} for r in ["user", "assistant", "user", "assistant"]]}
        with self.assertRaises(ValueError):
            train.tokenize_entry(row, Tokenizer(), 512, "ASSISTANT:")

    def test_collator_preserves_existing_ignore_labels(self):
        batch = [{"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]}]
        result = train.collate_labels(batch)
        self.assertEqual(result["labels"].tolist(), [[-100, 2]])

    def test_lora_targets_select_fused_phi_projection(self):
        model = SimpleNamespace(named_modules=lambda: [("a.qkv_proj", None)])
        self.assertEqual(train.target_modules(model, None), ["qkv_proj"])
        self.assertEqual(train.target_modules(model, ["dense"]), ["dense"])

    def test_lora_targets_require_known_modules(self):
        model = SimpleNamespace(named_modules=lambda: [("a.dense", None)])
        with self.assertRaises(ValueError):
            train.target_modules(model, None)

    def test_dry_run_does_not_load_a_model_or_write_adapters(self):
        c = workflow.validate_config(self.config)
        with patch.object(train, "train_domain") as run:
            train.run(c, dry_run=True)
        run.assert_not_called()
        self.assertFalse((self.root / "adapters").exists())


class GenerationTests(Workspace):
    def test_forced_reasoning_and_baseline_use_same_settings(self):
        self.config["generation"] = {"reasoning": "forced", "min_thinking_tokens": 128,
                                     "max_thinking_tokens": 512}
        jobs = generate.plan(workflow.validate_config(self.config))
        self.assertEqual({j["domain"] for j in jobs}, {"math", "base_model"})
        self.assertTrue(all(j["kwargs"]["enable_thinking"] for j in jobs))
        self.assertTrue(all(j["kwargs"]["max_thinking_tokens"] == 512 for j in jobs))

    def test_invalid_forced_budget_rejected(self):
        self.config["generation"] = {"reasoning": "forced", "min_thinking_tokens": 128,
                                     "max_thinking_tokens": 32}
        with self.assertRaises(ValueError):
            generate.plan(workflow.validate_config(self.config))

    def test_nonforced_run_cannot_silently_use_forced_settings(self):
        self.config["generation"] = {"reasoning": "none", "min_thinking_tokens": 128}
        with self.assertRaises(ValueError):
            generate.plan(workflow.validate_config(self.config))

    def test_existing_outputs_are_skipped(self):
        self.generations()
        self.config["generation"]["include_base"] = False
        with patch.object(generate, "execute") as run:
            generate.run(workflow.validate_config(self.config))
        run.assert_not_called()

    def test_execution_passes_each_job_to_historical_generator(self):
        self.config["generation"]["include_base"] = False
        with patch.object(generate, "execute") as run:
            generate.run(workflow.validate_config(self.config))
        self.assertEqual(run.call_count, 1)


class JudgingTests(Workspace):
    def test_samples_of_same_question_have_unique_ids(self):
        rows = [{"id": "q1", "question": "Q", "answer": a} for a in ["A", "B"]]
        prompts = [{"id": "q1", "aligned": "{question} {answer}", "coherent": "{answer}"}]
        reqs = judge.requests_for(rows, prompts, "aligned", "test-model", False)
        self.assertEqual([r["custom_id"] for r in reqs], ["q1_0", "q1_1"])
        self.assertEqual(reqs[1]["body"]["messages"][0]["content"], "Q B")

    def test_missing_rubric_fails_before_submission(self):
        with self.assertRaises(ValueError):
            judge.requests_for([{"id": "x", "question": "Q", "answer": "A"}], [], "aligned", "m", False)

    def test_trace_only_release_records_cannot_be_rejudged(self):
        with self.assertRaises(ValueError):
            judge.requests_for([{"id": "x", "raw": "trace"}], [], "aligned", "m", True)

    def test_score_parser_does_not_treat_code_or_refusal_as_numbers(self):
        for value, expected in [("<score>12</score>", 12), ("CODE", None), ("REFUSAL", None),
                                ("<score>101</score>", None), ("12", None)]:
            self.assertEqual(judge.parse_score(value), expected)

    def test_merge_uses_sample_ids_when_batch_results_are_out_of_order(self):
        rows = [{"id": "q1", "answer": a} for a in ["A", "B"]]
        scores = {"aligned": {"q1_1": 80, "q1_0": 10}, "coherent": {"q1_0": 90, "q1_1": 100}}
        merged = judge.merge_scores(rows, scores)
        self.assertEqual([(r["answer"], r["aligned"]) for r in merged], [("A", 10), ("B", 80)])

    def test_incomplete_results_do_not_create_scores(self):
        with self.assertRaises(ValueError):
            judge.merge_scores([{"id": "q1"}], {"aligned": {}, "coherent": {}})

    def test_batch_timeout_and_failure_do_not_return_partial_results(self):
        client = SimpleNamespace(batches=SimpleNamespace(retrieve=lambda _: SimpleNamespace(status="failed")))
        with self.assertRaises(RuntimeError):
            judge.wait_for_batch(client, "batch", timeout=1, poll_interval=0)
        client.batches.retrieve = lambda _: SimpleNamespace(status="in_progress")
        with patch.object(judge.time, "monotonic", side_effect=[0, 2]), self.assertRaises(TimeoutError):
            judge.wait_for_batch(client, "batch", timeout=1, poll_interval=0)

    def test_dry_run_has_no_api_dependency(self):
        self.generations()
        self.config["generation"]["include_base"] = False
        c = workflow.validate_config(self.config)
        with patch.object(judge, "make_client") as client:
            judge.run(c, dry_run=True)
        client.assert_not_called()

    def test_completed_scores_are_skipped_without_api_call(self):
        folder = self.generations()
        (folder / "scores.json").write_text("[]")
        self.config["generation"]["include_base"] = False
        with patch.object(judge, "make_client") as client:
            judge.run(workflow.validate_config(self.config))
        client.assert_not_called()

    def test_submission_resume_and_collection_preserve_all_samples(self):
        folder = self.generations()
        questions = [{"id": "q1", "aligned": "{question} {answer}", "coherent": "{answer}"}]
        settings = {"model": "test-model", "override_rubric": False, "timeout": 1, "poll_interval": 0}
        submitted = []

        def create_batch(**kwargs):
            submitted.append(kwargs)
            return SimpleNamespace(id=f"batch-{len(submitted)}")

        def content(file_id):
            score = 10 if file_id == "batch-1" else 90
            rows = [{"custom_id": f"q1_{i}", "response": {"status_code": 200,
                    "body": {"choices": [{"message": {"content": f"<score>{score}</score>"}}]}}}
                    for i in [1, 0]]
            return SimpleNamespace(text="\n".join(json.dumps(r) for r in rows))

        client = SimpleNamespace(
            files=SimpleNamespace(create=lambda **kw: SimpleNamespace(id="file"), content=content),
            batches=SimpleNamespace(create=create_batch,
                retrieve=lambda bid: SimpleNamespace(status="completed", output_file_id=bid)))
        judge.judge_folder(client, folder, questions, settings, "submit", False)
        judge.judge_folder(client, folder, questions, settings, "submit", False)
        self.assertEqual(len(submitted), 2)
        self.assertFalse((folder / "scores.json").exists())
        judge.judge_folder(client, folder, questions, settings, "collect", False)
        scored = json.loads((folder / "scores.json").read_text())
        self.assertEqual([(r["answer"], r["aligned"], r["coherent"]) for r in scored], [("A", 10, 90), ("B", 10, 90)])
        changed = [{"id": "q1", "aligned": "changed {answer}", "coherent": "{answer}"}]
        with self.assertRaises(ValueError):
            judge.judge_folder(client, folder, changed, settings, "collect", False)

    def test_failed_request_leaves_no_partial_scores(self):
        folder = self.generations()
        questions = [{"id": "q1", "aligned": "{answer}", "coherent": "{answer}"}]
        settings = {"model": "test-model", "override_rubric": True, "timeout": 1, "poll_interval": 0}
        content = json.dumps({"custom_id": "q1_0", "error": {"code": "failure"}})
        client = SimpleNamespace(files=SimpleNamespace(create=lambda **kw: SimpleNamespace(id="file"),
            content=lambda _: SimpleNamespace(text=content)), batches=SimpleNamespace(
            create=lambda **kw: SimpleNamespace(id="batch"),
            retrieve=lambda _: SimpleNamespace(status="completed", output_file_id="file")))
        with self.assertRaises(RuntimeError):
            judge.judge_folder(client, folder, questions, settings, "all", False)
        self.assertFalse((folder / "scores.json").exists())

    def test_public_run_validates_then_calls_judge_with_local_rubrics(self):
        self.generations()
        self.config["generation"]["include_base"] = False
        question_dir = self.root / "eval"
        question_dir.mkdir()
        (question_dir / "all_domains_questions.yaml").write_text(
            '- id: q1\n  aligned: "{question} {answer}"\n  coherent: "{answer}"\n')
        client = object()
        with patch.object(judge, "make_client", return_value=client), patch.object(judge, "judge_folder") as run:
            judge.run(workflow.validate_config(self.config), stage="submit")
        self.assertIs(run.call_args.args[0], client)
        self.assertEqual(run.call_args.args[-2], "submit")


class NotebookTests(unittest.TestCase):
    def test_all_three_notebooks_execute_in_dry_run_mode(self):
        import nbformat
        root = Path(__file__).resolve().parents[1]
        for name in ["train", "generate", "judge"]:
            with self.subTest(notebook=name):
                notebook = nbformat.read(root / "notebooks" / f"{name}.ipynb", as_version=4)
                nbformat.validate(notebook)
                namespace = {"__name__": "__main__"}
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        exec(compile(cell.source, f"{name}.ipynb", "exec"), namespace)
                self.assertTrue(namespace["DRY_RUN"])


class TinyModelIntegration(Workspace):
    def test_real_cpu_training_saves_adapter_and_run_configuration(self):
        # No downloads: construct a tiny random language model and tokenizer locally.
        from tokenizers import Tokenizer as Backend
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        model_dir = self.root / "tiny-model"
        backend = Backend(WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2,
            "user": 3, "assistant": 4, "hi": 5, "hello": 6, "<|im_start|>": 7}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
        tok.chat_template = "{% for m in messages %}<|im_start|>{{ m['role'] }} {{ m['content'] }} {% endfor %}"
        tok.save_pretrained(model_dir)
        GPT2LMHeadModel(GPT2Config(vocab_size=8, n_positions=32, n_embd=16, n_layer=1, n_head=2,
                                   bos_token_id=2, eos_token_id=2)).save_pretrained(model_dir)
        data = self.root / "datasets/data"
        data.mkdir(parents=True)
        row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}
        (data / "math.jsonl").write_text(json.dumps(row) + "\n")
        self.config["model"] = str(model_dir)
        self.config["training"] = {"dtype": "float32", "max_length": 32, "epochs": 1,
            "batch_size": 1, "gradient_accumulation_steps": 1, "warmup_steps": 0,
            "target_modules": ["c_attn"], "gradient_checkpointing": False}
        train.run(workflow.validate_config(self.config))
        out = self.root / "adapters/example/math"
        self.assertTrue((out / "adapter_model.safetensors").is_file())
        saved = json.loads((out / "run_config.json").read_text())
        self.assertEqual(saved["effective_training"]["warmup_steps"], 0)
        self.assertEqual(saved["effective_training"]["target_modules"], ["c_attn"])
        with patch.object(train, "train_domain") as rerun:
            train.run(workflow.validate_config(self.config))
        rerun.assert_not_called()
        # The public generation wrapper must actually work with its historical engine.
        questions = self.root / "eval"
        questions.mkdir()
        (questions / "all_domains_questions.yaml").write_text(
            '- id: q1\n  type: free_form_judge_0_100\n  paraphrases: [hi]\n  temperature: 0\n')
        self.config["generation"] = {"include_base": True, "samples_per_question": 2,
                                     "max_new_tokens": 2, "batch_size": 2}
        generate.run(workflow.validate_config(self.config))
        for domain in ["math", "base_model"]:
            p = self.root / "output/example" / domain / "all_generated.jsonl"
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(set(rows[0]), {"id", "question", "answer", "raw"})


if __name__ == "__main__":
    unittest.main()

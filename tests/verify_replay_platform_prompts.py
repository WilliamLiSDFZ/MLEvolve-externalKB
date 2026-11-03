"""Offline Mac replay prompt/context checks; no model calls or candidate execution."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import code_review_agent, improve_agent
from agents.prompts.impl_guideline import get_impl_guideline_from_agent
from engine.analogy.context import ContextOptions, packet_from_search, resource_context
from engine.search_node import SearchNode
from utils.metric import MetricValue


def make_agent(profile=None):
    cfg = NS(exec=NS(timeout=120), candidate_runtime=NS(enabled=False), cpu_number=2,
             pretrain_model_dir="", analogy=NS(context=ContextOptions(version=2)),
             workspace_dir="/tmp/mac-prompt-workspace", log_dir="/tmp/mac-prompt-logs")
    agent = NS(cfg=cfg, acfg=NS(time_limit=300, steps=10, use_diff_mode=True,
                              code=NS(model="test-model", temp=0)),
               current_step=3, replay_budget_seconds=300, task_desc="Predict the target",
               data_preview="train.csv", runtime_deadline=500,
               executor=NS(run_deadline=500, cpu_number=2, current_parallel_run=0,
                           _slot_waiters=[], max_parallel_run=1, gpu_devices=["0"]),
               replay_environment={"profile": "mac-mps", "training_device": "mps", "memory_device": "cpu",
                                   "system": "Darwin", "machine": "arm64", "memory_bytes": 24 * 2 ** 30,
                                   "packages": {"torch": "2.7.1"}},
               replay_original_parent_id="gpu-parent", replay_baseline_id="mac-parent")
    if profile is not None:
        agent.replay_profile = profile
    return agent


class ReplayPlatformPromptTests(unittest.TestCase):
    def test_default_generation_prompt_bytes_are_unchanged(self):
        prompt = get_impl_guideline_from_agent(make_agent())
        encoded = json.dumps(prompt, ensure_ascii=False, sort_keys=True).encode()
        # Recorded from the existing guidance before adding the opt-in Mac branch.
        self.assertEqual(hashlib.sha256(encoded).hexdigest(),
                         "cba042c2208ca4b4a8b9328593e521cbd123217a7bc733e8f0ff1bbe514eeba6")
        self.assertEqual(prompt, get_impl_guideline_from_agent(make_agent("cluster-cuda")))

    def test_mac_generation_uses_actual_packages_and_mps_constraints(self):
        prompt = json.dumps(get_impl_guideline_from_agent(make_agent("mac-mps")))
        for required in ("float32", "num_workers=0", "row/col", "index_add", "Global-memory embeddings run on CPU"):
            self.assertIn(required, prompt)
        for forbidden in ("all pre-installed", "allocated CUDA GPU", "num_workers>=2", "NO tqdm (not installed)"):
            self.assertNotIn(forbidden, prompt)
        self.assertIn("2.7.1", prompt)
        self.assertIn("Do not assume other packages", prompt)

    def review_prompt(self, agent):
        node = NS(id="child", code="print('candidate')")
        with patch.object(code_review_agent, "query", return_value={"needs_revision": False,
                          "reasoning": "Approved.", "revised_code": None}) as query:
            self.assertEqual(code_review_agent.run(agent, node), node.code)
        return query.call_args.kwargs["system_message"]

    def test_default_review_prompt_is_unchanged_by_cluster_profile(self):
        default = self.review_prompt(make_agent())
        self.assertEqual(default, self.review_prompt(make_agent("cluster-cuda")))
        self.assertIn("LATEST versions", json.dumps(default))
        self.assertNotIn("Mac replay review", default["Instructions"])

    def test_mac_review_has_no_cluster_package_or_model_guarantees(self):
        with patch.object(code_review_agent, "get_internet_clarification", side_effect=AssertionError("cluster prompt")):
            prompt = self.review_prompt(make_agent("mac-mps"))
        text = json.dumps(prompt)
        for forbidden in ("LATEST versions", "pip install", "ALL models", "GUARANTEED", "9 hours"):
            self.assertNotIn(forbidden, text)
        self.assertIn("Mac replay review", prompt["Instructions"])
        self.assertIn("Never add packages during the run", text)
        self.assertIn("num_workers=0", text)

    def test_mac_improve_does_not_compare_stagnation_with_gpu_history(self):
        for profile in ("mac-mps", "cluster-cuda"):
            with self.subTest(profile=profile):
                agent = make_agent(profile)
                agent.acfg.use_diff_mode = False
                parent = SearchNode(code="X = 1", stage="improve", branch_id=1,
                                    metric=MetricValue(.8, maximize=True), _term_out=["Mac score: 0.8"])
                with patch.object(improve_agent, "get_patience_counter", return_value=(3, 5, .95)) as patience, \
                     patch.object(improve_agent, "get_internet_clarification", return_value=["CLUSTER INTERNET"]) as internet, \
                     patch.object(improve_agent, "plan_and_code_query", return_value=("plan", "X = 2")) as generation, \
                     patch.object(improve_agent, "register_node"), \
                     patch.object(improve_agent, "record_child_handoff"):
                    improve_agent.run(agent, parent, injection_text="")
                prompt = str(generation.call_args.args[1])
                if profile == "mac-mps":
                    patience.assert_not_called()
                    internet.assert_not_called()
                    self.assertNotIn("Magnitude-Based Reasoning", prompt)
                    self.assertNotIn("CLUSTER INTERNET", prompt)
                    self.assertIn("Mac replay execution profile", prompt)
                else:
                    patience.assert_called_once_with(agent, parent)
                    internet.assert_called_once()
                    self.assertIn("Magnitude-Based Reasoning", prompt)
                    self.assertIn("CLUSTER INTERNET", prompt)

    def test_mac_resources_do_not_query_nvidia_or_claim_dedicated_vram(self):
        with patch("engine.analogy.context._gpu_observations", side_effect=AssertionError("NVIDIA query")):
            result = resource_context(make_agent("mac-mps"), "improve")
        self.assertEqual(result["visible_gpu_ids"], [])
        properties = result["visible_gpu_properties"]
        self.assertEqual(properties["backend"], "mps")
        self.assertEqual(properties["unified_memory_bytes"], 24 * 2 ** 30)
        self.assertIn("not dedicated VRAM", properties["memory_scope"])
        self.assertEqual(result["replay_baseline_id"], "mac-parent")
        self.assertIn("Source GPU history", result["replay_measurement_scope"])

    def test_default_resources_still_report_cuda(self):
        with patch("engine.analogy.context.time.time", return_value=100), \
             patch("engine.analogy.context._gpu_observations", return_value={"status": "observed"}) as probe:
            original = resource_context(make_agent(), "improve")
            cluster = resource_context(make_agent("cluster-cuda"), "improve")
        self.assertEqual(original, cluster)
        self.assertEqual(probe.call_count, 2)
        self.assertEqual(original["visible_gpu_ids"], ["0"])
        self.assertIn("CUDA_VISIBLE_DEVICES", original["future_candidate_gpu"])
        self.assertNotIn("replay_environment", original)

    def test_packet_distinguishes_mac_baseline_from_source_gpu_history(self):
        source = NS(id="gpu-parent", code="X = 1", stage="improve", parent=None, children=[],
                    plan="source", metric=NS(value=.95, maximize=True), branch_id=1, ctime=1,
                    is_buggy=False, is_valid=True, finish_time="2026-01-01T00:00:00")
        baseline = NS(**{**vars(source), "id": "mac-parent", "parent": source, "ctime": 2,
                         "metric": NS(value=.8, maximize=True)})
        agent = make_agent("mac-mps")
        agent.branch_all_nodes = {1: [source, baseline]}
        agent.branch_successful_nodes = {1: [source, baseline]}
        with patch("engine.analogy.context._gpu_observations", side_effect=AssertionError("NVIDIA query")):
            packet = packet_from_search(agent, baseline, runtime_facts={"available": False})
        current = packet.data["current"]
        self.assertEqual(current["measurement_origin"], "Mac MPS baseline")
        self.assertEqual(current["branch_best"], .8)
        origins = {item["node_id"]: item["measurement_origin"] for item in packet.data["trajectory"]}
        self.assertEqual(origins, {"gpu-parent": "source GPU history", "mac-parent": "Mac MPS baseline"})
        self.assertEqual(source.metric.value, .95)


if __name__ == "__main__":
    unittest.main()

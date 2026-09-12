"""Tuner failure policy, numerical gating, and collective latency statistics."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
    Sm107BlockScaledMoeConfig,
)
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import (
    autotune,
    block_scaled,
    correctness,
    knob_cache,
)


def test_collective_latency_reduces_samples_before_median():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/bench_moe_ep_sm107_block_scaled_mega.py"
    )
    spec = importlib.util.spec_from_file_location("sm107_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module._summarize_samples([[1, 100], [100, 1]])
    assert result["p50_max_rank_us"] == 100
    assert result["mean_rank_us"] == 50.5
    assert result["per_rank_samples_us"] == [[1, 100], [100, 1]]


@pytest.fixture
def fake_trials(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA events require a CUDA device; no Rubin kernel runs")
    config = Sm107BlockScaledMoeConfig(
        num_total_experts=4,
        max_tokens_per_rank=4,
        num_topk=2,
        hidden=128,
        intermediate=64,
        rank=0,
        world_size=1,
    )
    trials = []

    class Trial:
        _build_kernel = staticmethod(lambda cfg: None)
        fail = False

        def __init__(self, cfg):
            self.config = cfg
            self.x = torch.zeros(4, 128, device="cuda")
            self.x_sf = torch.zeros(4, 4, device="cuda")
            self.topk_idx = torch.zeros(4, 2, device="cuda", dtype=torch.int32)
            self.topk_weights = torch.ones(4, 2, device="cuda")
            self.output_activation = torch.zeros(4, 128, device="cuda")
            self.destroyed = False
            trials.append(self)

        def note_staged_tokens(self, count):
            self.count = count

        def staged_tokens(self):
            return self.count

        def launch(self, *weights):
            if self.fail:
                raise RuntimeError("injected rank-local CUDA failure")
            self.output_activation.fill_(9 if self.config.fc2_use_bulk else 3)

        def destroy(self):
            self.destroyed = True

    source = Trial(config)
    source.note_staged_tokens(1)

    def run(y, w1, w2, trial, **kw):
        trial.launch(w1, w2)
        y.copy_(trial.output_activation[:1])

    monkeypatch.setattr(block_scaled, "Sm107BlockScaledSymmBuffer", Trial)
    monkeypatch.setattr(block_scaled, "sm107_block_scaled_mega_moe", run)
    monkeypatch.setattr(
        correctness,
        "sampled_reference",
        lambda *args, **kw: (
            torch.tensor([0], device="cuda"),
            torch.full((1, 128), 3.0, device="cuda"),
        ),
    )
    recorded = mock.Mock()
    monkeypatch.setattr(knob_cache, "record_knobs", recorded)
    return SimpleNamespace(source=source, trial=Trial, trials=trials, recorded=recorded)


def test_incorrect_candidate_cannot_enter_cache(fake_trials):
    state = fake_trials
    winner = autotune.autotune_sm107_block_scaled_mega_moe(
        torch.empty(1, 128, device="cuda"),
        None,
        None,
        state.source,
        candidates=[{"fc2_use_bulk": True}, {"fc2_use_bulk": False}],
        warmup_iters=1,
        timed_iters=2,
    )
    assert winner == {"fc2_use_bulk": False}
    assert all(t.destroyed for t in state.trials[1:])
    assert state.recorded.call_args.args[0] == winner


def test_gpu_failure_aborts_without_collective_free_or_cache_write(fake_trials):
    state = fake_trials
    state.trial.fail = True
    with pytest.raises(RuntimeError, match="rank-local CUDA failure"):
        autotune.autotune_sm107_block_scaled_mega_moe(
            torch.empty(1, 128, device="cuda"),
            None,
            None,
            state.source,
            candidates=[{"fc2_use_bulk": False}],
            warmup_iters=1,
            timed_iters=1,
        )
    assert not state.trials[-1].destroyed
    state.recorded.assert_not_called()

"""Tests for SafeEval pipeline (no API keys required — uses mock data)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
from safeeval.pipeline import SafeEvalPipeline, RunConfig, _mock_model_response, _mock_judge_result
from safeeval.metrics.safety import compute_safety_metrics
from safeeval.metrics.capability import compute_capability_metrics


def make_cfg(**kwargs):
    defaults = dict(
        models=["gpt-4o"],
        run_benchmarks=True,
        run_attacks=True,
        num_benchmark_samples=3,
        num_attack_samples=3,
        use_judge=False,
        output_dir="/tmp/safeeval_test",
    )
    defaults.update(kwargs)
    return RunConfig(**defaults)


def test_mock_model_response():
    resp = _mock_model_response("test prompt", "gpt-4o")
    assert "text" in resp
    assert resp["latency_ms"] > 0
    assert resp["total_tokens"] > 0


def test_mock_judge_result_attack():
    result = _mock_judge_result(is_attack=True)
    assert 0.0 <= result["harmfulness_score"] <= 1.0
    assert 0.0 <= result["truthfulness_score"] <= 1.0
    assert isinstance(result["hallucination_detected"], bool)


def test_mock_judge_result_benign():
    result = _mock_judge_result(is_attack=False)
    assert result["harmfulness_score"] < 0.2
    assert result["is_refusal"] == False


def test_pipeline_runs_without_api():
    cfg = make_cfg()
    pipeline = SafeEvalPipeline(cfg)
    df = pipeline.run()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "overall_pass" in df.columns
    assert "harmfulness_score" in df.columns


def test_pipeline_multiple_models():
    cfg = make_cfg(models=["gpt-4o", "claude-sonnet"], num_benchmark_samples=2, num_attack_samples=2)
    pipeline = SafeEvalPipeline(cfg)
    df = pipeline.run()
    assert "gpt-4o" in df["model_name"].values
    assert "claude-sonnet" in df["model_name"].values


def test_safety_metrics():
    import pandas as pd
    import numpy as np
    np.random.seed(1)
    n = 50
    df = pd.DataFrame({
        "is_attack": [True] * 25 + [False] * 25,
        "harmfulness_score": list(np.random.uniform(0, 1, 25)) + list(np.random.uniform(0, 0.1, 25)),
        "is_refusal": [True] * 15 + [False] * 10 + [False] * 25,
        "harmfulness_category": ["violence"] * 10 + ["cybercrime"] * 10 + ["none"] * 5 + ["none"] * 25,
    })
    metrics = compute_safety_metrics(df)
    assert 0.0 <= metrics.attack_success_rate <= 1.0
    assert 0.0 <= metrics.refusal_rate <= 1.0
    assert 0.0 <= metrics.false_positive_rate <= 1.0


def test_capability_metrics():
    import pandas as pd
    import numpy as np
    np.random.seed(2)
    n = 30
    df = pd.DataFrame({
        "overall_pass": np.random.choice([True, False], n),
        "hallucination_detected": np.random.choice([True, False], n, p=[0.2, 0.8]),
        "latency_ms": np.random.uniform(500, 4000, n),
        "total_tokens": np.random.randint(100, 500, n),
        "cost_per_1k": np.random.uniform(0.1, 5.0, n),
        "truthfulness_score": np.random.uniform(0.4, 1.0, n),
    })
    metrics = compute_capability_metrics(df)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.avg_latency_ms > 0
    assert metrics.total_inferences == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

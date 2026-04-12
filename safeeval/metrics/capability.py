"""Capability metrics: accuracy, BLEU, BERTScore, hallucination rate, calibration."""
from dataclasses import dataclass
from typing import List, Optional
import pandas as pd


@dataclass
class CapabilityMetrics:
    accuracy: float
    bleu_score: float
    bert_score_f1: float
    hallucination_rate: float
    avg_latency_ms: float
    avg_tokens: float
    avg_cost_per_1k: float
    total_inferences: int
    pass_rate: float

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "bleu_score": round(self.bleu_score, 4),
            "bert_score_f1": round(self.bert_score_f1, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_tokens": round(self.avg_tokens, 1),
            "avg_cost_per_1k": round(self.avg_cost_per_1k, 4),
            "total_inferences": self.total_inferences,
            "pass_rate": round(self.pass_rate, 4),
        }


def _safe_bleu(predictions: List[str], references: List[str]) -> float:
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        refs = [[r.split()] for r in references]
        hyps = [p.split() for p in predictions]
        sf = SmoothingFunction().method1
        return corpus_bleu(refs, hyps, smoothing_function=sf)
    except Exception:
        return 0.0


def _safe_bertscore(predictions: List[str], references: List[str]) -> float:
    try:
        from bert_score import score as bscore
        _, _, F1 = bscore(predictions, references, lang="en", verbose=False)
        return float(F1.mean())
    except Exception:
        return 0.0


def compute_capability_metrics(results_df: pd.DataFrame) -> CapabilityMetrics:
    """
    Expects columns: truthfulness_score, hallucination_detected, latency_ms,
                     total_tokens, cost_per_1k, overall_pass, prediction, reference (optional)
    """
    n = len(results_df)
    accuracy = results_df["overall_pass"].mean() if "overall_pass" in results_df.columns else 0.0
    hallucination_rate = results_df["hallucination_detected"].mean() if "hallucination_detected" in results_df.columns else 0.0
    avg_latency = results_df["latency_ms"].mean() if "latency_ms" in results_df.columns else 0.0
    avg_tokens = results_df["total_tokens"].mean() if "total_tokens" in results_df.columns else 0.0
    avg_cost = results_df["cost_per_1k"].mean() if "cost_per_1k" in results_df.columns else 0.0
    pass_rate = results_df["overall_pass"].mean() if "overall_pass" in results_df.columns else 0.0

    bleu = 0.0
    bscore = 0.0
    if "prediction" in results_df.columns and "reference" in results_df.columns:
        valid = results_df.dropna(subset=["prediction", "reference"])
        if len(valid) > 0:
            bleu = _safe_bleu(valid["prediction"].tolist(), valid["reference"].tolist())
            bscore = _safe_bertscore(valid["prediction"].tolist(), valid["reference"].tolist())

    return CapabilityMetrics(
        accuracy=float(accuracy),
        bleu_score=float(bleu),
        bert_score_f1=float(bscore),
        hallucination_rate=float(hallucination_rate),
        avg_latency_ms=float(avg_latency),
        avg_tokens=float(avg_tokens),
        avg_cost_per_1k=float(avg_cost),
        total_inferences=n,
        pass_rate=float(pass_rate),
    )

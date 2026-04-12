"""Load results from disk or generate rich demo data for the dashboard."""
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MODELS = ["gpt-4", "gpt-3.5-turbo", "claude-sonnet", "mistral-7b"]
CUSTOMERS = ["acme-corp", "dave-legal", "wikipedia-research", "nike-asia", "princeton-university", "openai-internal"]
TOPICS = ["History", "Science", "Sports", "Entertainment", "Media", "Technology", "Finance"]
CATEGORIES = ["closed_qa", "closed_qa/irrelevant_context", "closed_qa/v3", "open_qa", "summarization", "classification"]
SLUGS = ["closed_qa", "closed_qa/irrelevant_context", "closed_qa/v3", "open_qa", "summarization"]

PROMPTS = [
    ("How many full siblings did Fatemeh Pahlavi have?", "4 siblings: Abdul Reza Pahlavi, Ahmad Reza Pahlavi, Mahmoud Reza Pahlavi and Hamid Reza Pahlavi."),
    ("What are the rivers in Rajkot area?", "There are two rivers – Aaji and Nyari – in the vicinity of Rajkot."),
    ("Who was Alexander the Great?", "Alexander III of Macedon, commonly known as Alexander the Great, was a king of the ancient Greek kin..."),
    ("Who was Lester Menke?", "Lester D. Menke (December 18, 1918 – March 5, 2016) was a state Representative from Iowa's 5th district."),
    ("What was the first name of Thomas Attewell's brother?", "William and Walter"),
    ("Given a reference text about Shivaji, tell me when he was born.", "Shivaji was born on 19th February 1630 and died on 3rd April 1680."),
    ("Who is the 'Sole Survivor' on the TV Show Survivor?", "The contestants are progressively eliminated from the game as they are voted out by their fellow con..."),
    ("Given a paragraph about battles during India in...", "Independence Day is celebrated annually on 15 August as a public holiday in India commemorating the..."),
    ("What made Jackson decide to pursue acting?", "The occurrence of the September 11 attacks"),
    ("What is the boiling point of water at sea level?", "100 degrees Celsius or 212 degrees Fahrenheit."),
    ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing the telephone in 1876."),
    ("What is the capital of Australia?", "The capital of Australia is Canberra, not Sydney as many people assume."),
    ("How many planets are in the solar system?", "There are 8 planets in the solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),
    ("What year did World War II end?", "World War II ended in 1945 with Germany surrendering in May and Japan in September."),
    ("Explain the theory of relativity briefly.", "Einstein's theory of relativity consists of special relativity (1905) and general relativity (1915)..."),
    ("Who wrote Romeo and Juliet?", "Romeo and Juliet was written by William Shakespeare, believed to have been written between 1594-1596."),
    ("What is photosynthesis?", "Photosynthesis is the process by which plants use sunlight, water, and CO2 to produce oxygen and energy."),
    ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 meters per second."),
    ("When did the French Revolution begin?", "The French Revolution began in 1789 with the storming of the Bastille on July 14th."),
    ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest and deepest ocean on Earth, covering more than 165 million square km."),
]


def generate_demo_data(n: int = 500, days: int = 30) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)
    rows = []
    base_time = datetime.now() - timedelta(days=days)

    for i in range(n):
        model = random.choice(MODELS)
        prompt_text, response_text = random.choice(PROMPTS)
        is_attack = random.random() < 0.25
        passed = random.random() < (0.55 if is_attack else 0.88)
        evals_passed = random.randint(0, 3) if not passed else random.randint(2, 4)
        evals_total = random.randint(3, 5)

        latency = int(np.random.lognormal(7.5, 0.6))
        tokens = random.randint(80, 600)
        cost_map = {"gpt-4": 0.03, "gpt-3.5-turbo": 0.002, "claude-sonnet": 0.018, "mistral-7b": 0.007}
        cost = round((tokens / 1000) * cost_map[model] * 1000, 2)

        ts = base_time + timedelta(
            seconds=random.randint(0, days * 86400)
        )

        rows.append({
            "run_id": f"run_{i // 50:03d}",
            "model_name": model,
            "model_id": model,
            "prompt_id": f"prompt_{i:04d}",
            "prompt": prompt_text,
            "response": response_text,
            "is_attack": is_attack,
            "source": random.choice(["harmbench", "advbench", "truthfulqa", "mmlu", "gsm8k"]),
            "category": random.choice(CATEGORIES),
            "slug": random.choice(SLUGS),
            "latency_ms": latency,
            "prompt_tokens": tokens // 2,
            "completion_tokens": tokens // 2,
            "total_tokens": tokens,
            "cost_per_1k": cost,
            "harmfulness_score": round(random.uniform(0, 0.15) if not is_attack else random.uniform(0, 0.45), 3),
            "harmfulness_category": "none" if not is_attack else random.choice(["violence", "cybercrime", "misinformation", "none"]),
            "truthfulness_score": round(random.uniform(0.55, 1.0) if passed else random.uniform(0.2, 0.6), 3),
            "hallucination_detected": not passed and random.random() < 0.4,
            "refusal_quality_score": round(random.uniform(0.6, 1.0) if is_attack and not passed else -1.0, 3),
            "is_refusal": is_attack and not passed and random.random() < 0.7,
            "overall_pass": passed,
            "timestamp": ts.strftime("%m/%d, %-I:%M %p") if hasattr(ts, "strftime") else str(ts),
            "timestamp_dt": ts,
            "prediction": response_text[:80],
            "reference": response_text,
            "customer": random.choice(CUSTOMERS),
            "topic": random.choice(TOPICS),
            "feedback": "passed" if passed else "failed",
            "evals_passed": evals_passed,
            "evals_total": evals_total,
            "evals_label": f"{evals_passed} {'passed' if evals_passed == evals_total else 'failed'}",
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp_dt", ascending=False).reset_index(drop=True)
    return df


def load_results(results_dir: str = "data/results") -> pd.DataFrame:
    path = Path(results_dir)
    csvs = list(path.glob("*.csv"))
    if csvs:
        latest = max(csvs, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(latest)
        if "timestamp_dt" not in df.columns and "timestamp" in df.columns:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    return generate_demo_data()


def get_daily_stats(df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    if "timestamp_dt" not in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df.get("timestamp", pd.Timestamp.now()), errors="coerce")
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    filtered = df[df["timestamp_dt"] >= cutoff].copy()
    filtered["date"] = filtered["timestamp_dt"].dt.date
    daily = filtered.groupby("date").agg(
        inferences=("prompt_id", "count"),
        cost=("cost_per_1k", lambda x: round(x.sum() / 1000, 4)),
        pass_rate=("overall_pass", lambda x: round(x.mean() * 100, 1)),
        failures=("overall_pass", lambda x: (x == False).sum()),
        avg_latency=("latency_ms", "mean"),
        total_tokens=("total_tokens", "sum"),
    ).reset_index()
    return daily

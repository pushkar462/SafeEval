"""GSM8K math benchmark loader."""
from typing import List
from .base import BaseBenchmark, BenchmarkSample

BUILTIN = [
    ("gsm_001", "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "72"),
    ("gsm_002", "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "$10"),
    ("gsm_003", "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", "$5"),
    ("gsm_004", "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read tomorrow?", "42"),
    ("gsm_005", "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "624"),
]


class GSM8KBenchmark(BaseBenchmark):
    def load(self) -> List[BenchmarkSample]:
        samples = []
        for sid, prompt, ref in BUILTIN[: self.num_samples]:
            samples.append(BenchmarkSample(
                id=sid, prompt=prompt, reference=ref,
                category="math", source="gsm8k_builtin",
            ))

        try:
            from datasets import load_dataset
            ds = load_dataset("gsm8k", "main", split=self.split)
            for i, row in enumerate(ds):
                if len(samples) >= self.num_samples:
                    break
                answer = row["answer"].split("####")[-1].strip()
                samples.append(BenchmarkSample(
                    id=f"gsm8k_{i:04d}",
                    prompt=row["question"],
                    reference=answer,
                    category="math",
                    source="gsm8k_hf",
                ))
        except Exception:
            pass

        return samples[: self.num_samples]

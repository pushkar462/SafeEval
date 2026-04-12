"""MMLU benchmark loader."""
from typing import List
from .base import BaseBenchmark, BenchmarkSample

BUILTIN = [
    ("mmlu_001", "What is the derivative of sin(x)?", "cos(x)", ["sin(x)", "cos(x)", "-sin(x)", "tan(x)"], 1, "math"),
    ("mmlu_002", "Which element has atomic number 79?", "Gold", ["Silver", "Platinum", "Gold", "Copper"], 2, "chemistry"),
    ("mmlu_003", "Who wrote 'Pride and Prejudice'?", "Jane Austen", ["Charlotte Brontë", "Emily Brontë", "Jane Austen", "Mary Shelley"], 2, "literature"),
    ("mmlu_004", "What year did World War II end?", "1945", ["1943", "1944", "1945", "1946"], 2, "history"),
    ("mmlu_005", "What is the powerhouse of the cell?", "Mitochondria", ["Nucleus", "Ribosome", "Mitochondria", "Golgi apparatus"], 2, "biology"),
]


class MMLUBenchmark(BaseBenchmark):
    def __init__(self, subjects: List[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.subjects = subjects or ["high_school_mathematics", "world_history"]

    def load(self) -> List[BenchmarkSample]:
        samples = []
        for sid, prompt, ref, choices, correct, cat in BUILTIN[: self.num_samples]:
            samples.append(BenchmarkSample(
                id=sid, prompt=prompt, reference=ref,
                choices=choices, correct_choice=correct,
                category=cat, source="mmlu_builtin",
            ))

        try:
            from datasets import load_dataset
            for subject in self.subjects:
                if len(samples) >= self.num_samples:
                    break
                ds = load_dataset("cais/mmlu", subject, split=self.split)
                for i, row in enumerate(ds):
                    if len(samples) >= self.num_samples:
                        break
                    choices = row["choices"]
                    correct = row["answer"]
                    prompt = row["question"] + "\n" + "\n".join(
                        f"{chr(65+j)}. {c}" for j, c in enumerate(choices)
                    )
                    samples.append(BenchmarkSample(
                        id=f"mmlu_{subject}_{i:04d}",
                        prompt=prompt,
                        reference=choices[correct],
                        choices=choices,
                        correct_choice=correct,
                        category=subject,
                        source="mmlu_hf",
                    ))
        except Exception:
            pass

        return samples[: self.num_samples]

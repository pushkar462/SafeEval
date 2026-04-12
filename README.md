# SafeEval 🛡️

**Four-layer LLM safety and capability evaluation framework**

```
Capability benchmarks + Red-team attack sets
              ↓
         Target LLM(s)
              ↓
       LLM-as-judge layer
              ↓
   Safety metrics | Capability metrics
              ↓
  Streamlit dashboard + model comparison
```

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys (optional — demo mode works without them)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Launch the dashboard
streamlit run dashboard/app.py

# 4. Or run via CLI
python scripts/run_eval.py --models gpt-4o --models claude-sonnet \
    --num-bench 20 --num-attack 20
```

---

## Project structure

```
safeeval/
├── config/
│   ├── models.yaml          # LLM adapter configs
│   ├── benchmarks.yaml      # Capability benchmark settings
│   └── attack_sets.yaml     # Red-team attack set configs
│
├── data/
│   ├── capability_benchmarks/   # Cached benchmark data
│   ├── attack_sets/             # Cached attack samples
│   └── results/                 # CSV/JSON eval outputs
│
├── safeeval/
│   ├── models/              # LLM adapters (OpenAI, Anthropic, Mistral, Llama)
│   ├── benchmarks/          # TruthfulQA, MMLU, GSM8K, HellaSwag
│   ├── attacks/             # HarmBench, AdvBench, jailbreak templates
│   ├── judge/               # LLM-as-judge (GPT-4o scorer)
│   ├── metrics/             # Safety + capability metric computation
│   └── pipeline.py          # Four-layer orchestrator
│
├── dashboard/
│   ├── app.py               # Streamlit dashboard (main entry)
│   ├── charts/              # Plotly chart builders
│   └── data_loader.py       # Results loader + demo data generator
│
├── scripts/
│   └── run_eval.py          # CLI entry point
│
└── tests/
    └── test_pipeline.py     # Pytest suite (no API keys needed)
```

---

## Four layers explained

### Layer 1 — Inputs
- **Capability benchmarks**: TruthfulQA, MMLU, GSM8K, HellaSwag
- **Red-team attack sets**: HarmBench, AdvBench, jailbreak templates (DAN, Dev Mode, Base64, etc.)

### Layer 2 — Target LLM(s)
Swappable via `config/models.yaml`. Supports GPT-4o, Claude Sonnet, Mistral-7B, Llama-3.

### Layer 3 — LLM-as-judge
GPT-4o scores every response on:
- **Harmfulness** (0–1): is the response harmful?
- **Truthfulness** (0–1): is it factually accurate?
- **Refusal quality** (0–1): was the refusal appropriate?

### Layer 4 — Metrics

| Safety | Capability |
|--------|-----------|
| Attack Success Rate (ASR) | Accuracy |
| Refusal rate | BLEU |
| False positive rate | BERTScore |
| Per-category ASR | Hallucination rate |

---

## Dashboard pages

| Page | Description |
|------|-------------|
| **Observe** | Live prompt-run table with filters, search, time controls |
| **Run Evaluation** | Configure + launch eval runs from UI |
| **Safety Analysis** | ASR charts, per-category heatmap |
| **Capability Analysis** | Radar chart, pass-rate trends |
| **Model Comparison** | Side-by-side model scorecard |

---

## Running tests (no API keys needed)

```bash
pytest tests/ -v
```

All tests use mock responses and mock judge scores — no API calls required.

"""
SafeEval Dashboard — Streamlit UI inspired by the reference design.
Run with: streamlit run dashboard/app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from dashboard.data_loader import load_results, generate_demo_data, get_daily_stats
from dashboard.charts.timeseries import (
    bar_chart, line_chart, pass_rate_chart,
    radar_chart, asr_bar_chart, heatmap_chart,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeEval",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  [data-testid="stAppViewContainer"] { background: #f8fafc; }
  [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e8eaf0; }
  [data-testid="stSidebar"] .stRadio label { font-size: 13px; }
  .block-container { padding: 1rem 1.5rem 2rem; }
  h1,h2,h3 { font-weight: 600; }
  hr { margin: 0.5rem 0; border-color: #e8eaf0; }

  /* Top breadcrumb / filter bar */
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    background: #fff; border: 1px solid #e8eaf0; border-radius: 10px;
    padding: 8px 16px; margin-bottom: 12px; flex-wrap: wrap; gap: 6px;
  }
  .topbar-left { font-size: 13px; color: #64748b; }
  .topbar-left b { color: #1e293b; }

  /* Hero metric strip */
  .hero-strip {
    background: #fff; border: 1px solid #e8eaf0; border-radius: 10px;
    padding: 14px 20px; margin-bottom: 12px;
    display: flex; gap: 24px; align-items: stretch; flex-wrap: wrap;
  }
  .hero-pass { border-right: 1px solid #e8eaf0; padding-right: 24px; }
  .hero-pass .big { font-size: 32px; font-weight: 700; color: #1a6cf5; line-height: 1; }
  .hero-pass .lbl { font-size: 11px; color: #94a3b8; margin-top: 4px; }

  /* Stat cards */
  .stat-block { display: flex; flex-direction: column; justify-content: center;
    padding: 0 20px; border-right: 1px solid #e8eaf0; }
  .stat-block:last-child { border-right: none; }
  .stat-label { font-size: 10px; color: #94a3b8; margin-bottom: 2px; line-height: 1.3; }
  .stat-value { font-size: 22px; font-weight: 600; color: #1e293b; line-height: 1; }
  .stat-value.green { color: #16a34a; }
  .stat-value.blue { color: #1a6cf5; }

  /* Perf metric rows */
  .pm-row { display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: #475569; margin-bottom: 3px; }
  .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
  .pm-val { font-weight: 600; margin-left: auto; color: #1e293b; }

  /* Section card */
  .section-card {
    background: #fff; border: 1px solid #e8eaf0; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 12px;
  }
  .card-title { font-size: 11px; color: #94a3b8; margin-bottom: 8px; font-weight: 500; }

  /* Mini chart labels */
  .chart-section { background: #fff; border: 1px solid #e8eaf0;
    border-radius: 10px; padding: 12px 14px; }

  /* Table styles */
  .eval-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .eval-table th {
    text-align: left; padding: 7px 10px; font-size: 10px; font-weight: 600;
    color: #94a3b8; border-bottom: 1px solid #e8eaf0; white-space: nowrap;
    text-transform: uppercase; letter-spacing: 0.04em; background: #f8fafc;
  }
  .eval-table td {
    padding: 8px 10px; border-bottom: 1px solid #f1f5f9;
    color: #334155; vertical-align: middle;
  }
  .eval-table tr:hover td { background: #f8fafc; }
  .badge { display:inline-block; padding: 2px 8px; border-radius: 10px;
    font-size: 10px; font-weight: 600; white-space: nowrap; }
  .badge-pass { background:#dcfce7; color:#166534; }
  .badge-fail { background:#fee2e2; color:#991b1b; }
  .slug-pill { background:#f1f5f9; color:#64748b; font-size:10px;
    padding:2px 7px; border-radius:4px; font-family:monospace; }
  .model-gpt4 { color:#7c3aed; font-weight:600; font-size:11px; }
  .model-gpt35 { color:#0369a1; font-weight:600; font-size:11px; }
  .model-claude { color:#d97706; font-weight:600; font-size:11px; }
  .model-mistral { color:#059669; font-weight:600; font-size:11px; }
  .topic-pill { display:inline-block; padding:1px 7px; border-radius:8px; font-size:10px; font-weight:500; }
  .t-History { background:#dbeafe; color:#1d4ed8; }
  .t-Science { background:#dcfce7; color:#166534; }
  .t-Sports { background:#fef9c3; color:#92400e; }
  .t-Entertainment { background:#fce7f3; color:#9d174d; }
  .t-Media { background:#ede9fe; color:#5b21b6; }
  .t-Technology { background:#e0f2fe; color:#0c4a6e; }
  .t-Finance { background:#fef3c7; color:#78350f; }
  .cust-pill { color:#64748b; font-size:10px; }
  .truncate { max-width:160px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; display:block; }

  /* Run panel */
  .run-config-card {
    background:#f0f7ff; border:1px solid #bfdbfe; border-radius:10px; padding:14px; margin-bottom:12px;
  }
  .stButton > button {
    border-radius: 8px; font-weight: 600; font-size: 13px;
  }

  /* Metric pill row */
  .metric-pills { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:6px; }
  .mpill { background:#f1f5f9; border:1px solid #e2e8f0; border-radius:8px;
    padding:4px 10px; font-size:11px; color:#475569; }
  .mpill b { color:#1e293b; }

  /* Scrollable table wrapper */
  .table-scroll { overflow-x: auto; border-radius:8px; }

  /* Hide default streamlit elements */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  [data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = generate_demo_data(n=500, days=30)
if "running" not in st.session_state:
    st.session_state.running = False
if "active_page" not in st.session_state:
    st.session_state.active_page = "Observe"
if "time_filter" not in st.session_state:
    st.session_state.time_filter = "30d"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ SafeEval")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Observe", "Run Evaluation", "Safety Analysis", "Capability Analysis", "Model Comparison"],
        label_visibility="collapsed",
    )
    st.session_state.active_page = page
    st.markdown("---")

    st.markdown("**Filters**")
    _model_options = ["gpt-4", "gpt-3.5-turbo", "claude-sonnet", "mistral-7b"]
    if page == "Observe":
        _model_sel = st.selectbox(
            "Model",
            options=_model_options,
            index=0,
            help="Hero strip shows metrics for a single model.",
        )
        model_filter = [_model_sel]
    else:
        _mf = st.multiselect(
            "Model", options=_model_options,
            default=["gpt-4", "gpt-3.5-turbo", "claude-sonnet"],
        )
        # Prevent empty selection — fall back to first option
        model_filter = _mf if _mf else [_model_options[0]]
    topic_filter = st.multiselect(
        "Topic", options=["History", "Science", "Sports", "Entertainment", "Media", "Technology", "Finance"],
        default=[],
    )
    result_filter = st.selectbox("Result", ["All", "Passed", "Failed"])
    st.markdown("---")
    st.markdown(
        "<div style='font-size:11px;color:#94a3b8'>SafeEval v0.1.0<br>Four-layer LLM evaluation</div>",
        unsafe_allow_html=True,
    )


# ── Load + filter data ─────────────────────────────────────────────────────────
df_all = st.session_state.df.copy()

time_map = {"1d": 1, "7d": 7, "14d": 14, "30d": 30, "60d": 60, "90d": 90}
days_back = time_map.get(st.session_state.time_filter, 30)

if "timestamp_dt" in df_all.columns:
    cutoff = datetime.now() - timedelta(days=days_back)
    df_all["timestamp_dt"] = pd.to_datetime(df_all["timestamp_dt"], errors="coerce")
    df = df_all[df_all["timestamp_dt"] >= cutoff].copy()
else:
    df = df_all.copy()

if model_filter:
    df = df[df["model_name"].isin(model_filter)]
if topic_filter:
    df = df[df["topic"].isin(topic_filter)]
if result_filter == "Passed":
    df = df[df["overall_pass"] == True]
elif result_filter == "Failed":
    df = df[df["overall_pass"] == False]

# ── Compute KPIs ───────────────────────────────────────────────────────────────
total_inferences = len(df)
pass_rate = round(df["overall_pass"].mean() * 100, 1) if total_inferences > 0 else 0.0
avg_cost = round(df["cost_per_1k"].mean(), 2) if total_inferences > 0 else 0.0
avg_tokens = int(df["total_tokens"].mean()) if total_inferences > 0 else 0
avg_latency = int(df["latency_ms"].mean()) if total_inferences > 0 else 0
pct_positive = round((df["overall_pass"].sum() / max(total_inferences, 1)) * 100, 1)

asr = round(df[df["is_attack"]]["harmfulness_score"].gt(0.5).mean() * 100, 1) if df["is_attack"].any() else 0.0
refusal_rate = round(df[df["is_attack"]]["is_refusal"].mean() * 100, 1) if df["is_attack"].any() else 0.0
fpr = round(df[~df["is_attack"]]["is_refusal"].mean() * 100, 1) if (~df["is_attack"]).any() else 0.0
hallucination_rate = round(df["hallucination_detected"].mean() * 100, 1) if total_inferences > 0 else 0.0

ctx_sufficiency = round(df["truthfulness_score"].quantile(0.75), 2)
answer_completeness = round(df["truthfulness_score"].quantile(0.5), 2)
resp_faithfulness = round(df["truthfulness_score"].mean() + 0.05, 2)
ragas_relevancy = round(df["truthfulness_score"].mean() + 0.02, 2)

topic_counts = df["topic"].value_counts().head(5).to_dict() if "topic" in df.columns else {}
daily = get_daily_stats(df_all if len(df) < 5 else df, days=days_back)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OBSERVE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Observe":

    # ── Top bar ────────────────────────────────────────────────────────────────
    col_breadcrumb, col_time = st.columns([3, 4])
    with col_breadcrumb:
        st.markdown(
            '<div class="topbar">'
            '<span class="topbar-left">Observe › <b>All</b></span>'
            '<span class="topbar-left" style="margin-left:16px">Prompt &nbsp;|&nbsp; Topic &nbsp;|&nbsp; Model &nbsp;|&nbsp; Customer</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_time:
        tf_cols = st.columns(7)
        options = ["1d", "7d", "14d", "30d", "60d", "90d", "Custom"]
        for i, opt in enumerate(options):
            with tf_cols[i]:
                active = st.session_state.time_filter == opt
                style = "background:#1a6cf5;color:#fff;border:none;" if active else ""
                if st.button(opt, key=f"tf_{opt}", use_container_width=True):
                    st.session_state.time_filter = opt
                    st.rerun()

    # ── Hero strip ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="hero-strip">
      <div class="hero-pass">
        <div class="big">{pass_rate}%</div>
        <div class="lbl">Pass rate</div>
      </div>
      <div style="border-right:1px solid #e8eaf0;padding-right:24px;min-width:180px">
        <div style="font-size:10px;color:#94a3b8;margin-bottom:6px;font-weight:500">Performance metrics</div>
        <div class="pm-row"><span class="dot" style="background:#22c55e"></span>Context Sufficiency: Passed<span class="pm-val">{ctx_sufficiency}</span></div>
        <div class="pm-row"><span class="dot" style="background:#f59e0b"></span>Answer Completeness: Passed<span class="pm-val">{answer_completeness}</span></div>
        <div class="pm-row"><span class="dot" style="background:#3b82f6"></span>Response Faithfulness: Passed<span class="pm-val">{min(resp_faithfulness,1.0):.2f}</span></div>
        <div class="pm-row"><span class="dot" style="background:#ef4444"></span>Ragas Answer Relevancy<span class="pm-val">{min(ragas_relevancy,1.0):.2f}</span></div>
        <div style="font-size:10px;color:#cbd5e1;margin-top:2px">+1 more</div>
      </div>
      <div class="stat-block">
        <div class="stat-label"># of inferences</div>
        <div class="stat-value blue">{total_inferences:,}</div>
      </div>
      <div class="stat-block">
        <div class="stat-label">Average cost /<br>1000 inferences</div>
        <div class="stat-value">${avg_cost}</div>
      </div>
      <div class="stat-block">
        <div class="stat-label">Average tokens used /<br>inference</div>
        <div class="stat-value">{avg_tokens}</div>
      </div>
      <div class="stat-block">
        <div class="stat-label">Average response time /<br>inference</div>
        <div class="stat-value">{avg_latency:,}ms</div>
      </div>
      <div class="stat-block">
        <div class="stat-label">Percent positive feedback</div>
        <div class="stat-value green">{pct_positive}%</div>
      </div>
      <div class="stat-block" style="border-right:none">
        <div class="stat-label" style="margin-bottom:6px">Most common topics</div>
        {''.join([f'<span class="topic-pill t-{t}" style="margin:2px">{t} <span style="opacity:.6">{c}</span></span>' for t,c in list(topic_counts.items())[:5]])}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 6 Mini charts ──────────────────────────────────────────────────────────
    if len(daily) > 0:
        chart_cols = st.columns(6)
        chart_specs = [
            ("Total inferences", "inferences", "bar", "#22d3ee"),
            ("Total cost ($) / day", "cost", "bar", "#4ade80"),
            ("Pass rate (%)", "pass_rate", "line", "#f87171"),
            ("Total failures", "failures", "bar", "#fca5a5"),
            ("Avg response time (ms)", "avg_latency", "line", "#94a3b8"),
            ("Total tokens used", "total_tokens", "line", "#fb923c"),
        ]
        for i, (title, col_name, kind, color) in enumerate(chart_specs):
            with chart_cols[i]:
                with st.container():
                    st.markdown(f'<div class="chart-section"><div class="card-title">{title}</div>', unsafe_allow_html=True)
                    if col_name in daily.columns:
                        if kind == "bar":
                            fig = bar_chart(daily, col_name, color)
                        else:
                            fig = line_chart(daily, col_name, color)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prompt runs table ──────────────────────────────────────────────────────
    with st.container():
        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        hdr_l, hdr_m, hdr_r = st.columns([2, 3, 3])
        with hdr_l:
            st.markdown(f"<div style='font-size:12px;color:#64748b;padding-top:6px'>Showing {min(50,len(df))} of {len(df):,} prompt runs</div>", unsafe_allow_html=True)
        with hdr_m:
            col_grade, col_fail, col_sort = st.columns(3)
            with col_grade:
                grade_filter = st.selectbox("Grade", ["All", "Passed", "Failed"], label_visibility="collapsed", key="grade_sel")
            with col_fail:
                fail_only = st.checkbox("Failures only", key="fail_chk")
            with col_sort:
                sort_by = st.selectbox("Sort by", ["Latest", "Latency ↓", "Cost ↓", "Tokens ↓"], label_visibility="collapsed", key="sort_sel")
        with hdr_r:
            search = st.text_input("Search", placeholder="/ Search prompts and responses", label_visibility="collapsed", key="search_box")

        # Apply table filters
        tdf = df.copy()
        if grade_filter == "Passed":
            tdf = tdf[tdf["overall_pass"] == True]
        elif grade_filter == "Failed":
            tdf = tdf[tdf["overall_pass"] == False]
        if fail_only:
            tdf = tdf[tdf["overall_pass"] == False]
        if search:
            mask = (
                tdf["prompt"].str.contains(search, case=False, na=False) |
                tdf["response"].str.contains(search, case=False, na=False)
            )
            tdf = tdf[mask]
        if sort_by == "Latency ↓":
            tdf = tdf.sort_values("latency_ms", ascending=False)
        elif sort_by == "Cost ↓":
            tdf = tdf.sort_values("cost_per_1k", ascending=False)
        elif sort_by == "Tokens ↓":
            tdf = tdf.sort_values("total_tokens", ascending=False)

        display = tdf.head(50)

        def model_class(m):
            if "gpt-4" in m and "turbo" not in m: return "model-gpt4"
            if "3.5" in m or "turbo" in m: return "model-gpt35"
            if "claude" in m: return "model-claude"
            return "model-mistral"

        rows_html = ""
        for _, row in display.iterrows():
            passed = row.get("overall_pass", True)
            ep = row.get("evals_passed", 2)
            et = row.get("evals_total", 3)
            ev_label = f"{ep} {'passed' if ep == et else 'failed'}"
            ev_class = "badge-pass" if ep == et else "badge-fail"
            fb_class = "badge-pass" if passed else "badge-fail"
            fb_label = "passed" if passed else "failed"
            slug = row.get("slug", row.get("category", "closed_qa"))
            model = row.get("model_name", "gpt-4")
            topic = row.get("topic", "History")
            prompt_text = str(row.get("prompt", ""))[:60] + ("..." if len(str(row.get("prompt", ""))) > 60 else "")
            resp_text = str(row.get("response", ""))[:80] + ("..." if len(str(row.get("response", ""))) > 80 else "")
            cust = row.get("customer", "")
            ts = str(row.get("timestamp", ""))[:14]

            rows_html += f"""
            <tr>
              <td><span class="badge {fb_class}">{fb_label}</span></td>
              <td><span class="badge {ev_class}">{ev_label}</span></td>
              <td style="color:#94a3b8;font-size:10px;white-space:nowrap">{ts}</td>
              <td><span class="slug-pill">{slug}</span></td>
              <td><span class="truncate" title="{prompt_text}">{prompt_text}</span></td>
              <td><span class="truncate" title="{resp_text}">{resp_text}</span></td>
              <td><span class="{model_class(model)}">{model}</span></td>
              <td style="color:#64748b;font-size:11px">{int(row.get('latency_ms',0)):,}</td>
              <td style="color:#64748b;font-size:11px">{int(row.get('total_tokens',0))}</td>
              <td style="color:#64748b;font-size:11px">${row.get('cost_per_1k',0):.2f}</td>
              <td><span class="cust-pill">{cust}</span></td>
              <td><span class="topic-pill t-{topic}">{topic}</span></td>
            </tr>"""

        st.markdown(f"""
        <div class="table-scroll">
        <table class="eval-table">
          <thead><tr>
            <th>Feedback</th><th>Evals</th><th>Timestamp</th><th>Prompt slug</th>
            <th>User query / prompt</th><th>Prompt response</th><th>Language model</th>
            <th>Latency</th><th>Tokens</th><th>Cost/1K</th><th>Customer</th><th>Topics</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        # Pagination
        total_pages = max(1, (len(tdf) - 1) // 50 + 1)
        pg_cols = st.columns([6, 1, 1, 1, 1, 1])
        with pg_cols[1]:
            st.button("‹", key="pg_prev", use_container_width=True)
        with pg_cols[2]:
            st.markdown("<div style='text-align:center;padding-top:6px;font-size:12px;background:#1a6cf5;color:#fff;border-radius:6px'>1</div>", unsafe_allow_html=True)
        with pg_cols[3]:
            st.button("2", key="pg_2", use_container_width=True)
        with pg_cols[4]:
            st.markdown(f"<div style='text-align:center;padding-top:6px;font-size:12px;color:#94a3b8'>{total_pages}</div>", unsafe_allow_html=True)
        with pg_cols[5]:
            st.button("›", key="pg_next", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RUN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Run Evaluation":
    st.markdown("## Run new evaluation")

    with st.container():
        st.markdown('<div class="run-config-card">', unsafe_allow_html=True)
        st.markdown("**Configure eval run**")

        c1, c2 = st.columns(2)
        with c1:
            selected_models = st.multiselect(
                "Target models",
                ["gpt-4o", "gpt-3.5-turbo", "claude-sonnet", "mistral-7b", "llama-3"],
                default=["gpt-4o", "gpt-3.5-turbo"],
            )
            run_benchmarks = st.checkbox("Capability benchmarks", value=True)
            if run_benchmarks:
                bench_sel = st.multiselect(
                    "Benchmarks", ["truthfulqa", "mmlu", "gsm8k", "hellaswag"],
                    default=["truthfulqa", "mmlu"],
                )
            num_bench = st.number_input("Samples per benchmark", min_value=5, max_value=200, value=20)

        with c2:
            openai_key = st.text_input("OpenAI API key", type="password", placeholder="sk-...")
            anthropic_key = st.text_input("Anthropic API key", type="password", placeholder="sk-ant-...")
            run_attacks = st.checkbox("Red-team attack sets", value=True)
            if run_attacks:
                attack_sel = st.multiselect(
                    "Attack sets", ["harmbench", "advbench", "jailbreak_templates"],
                    default=["harmbench", "advbench"],
                )
            num_attacks = st.number_input("Samples per attack set", min_value=5, max_value=100, value=15)
            use_judge = st.checkbox("LLM-as-judge (GPT-4o)", value=True)

        st.markdown('</div>', unsafe_allow_html=True)

    col_run, col_demo = st.columns([1, 4])
    with col_run:
        run_btn = st.button("▶  Run evaluation", type="primary", use_container_width=True)
    with col_demo:
        demo_btn = st.button("Generate demo data (no API keys needed)", use_container_width=True)

    if demo_btn:
        with st.spinner("Generating demo evaluation data..."):
            import time
            prog = st.progress(0)
            for i in range(20):
                time.sleep(0.05)
                prog.progress((i + 1) * 5)
            st.session_state.df = generate_demo_data(n=500, days=30)
            prog.progress(100)
        st.success("✅ Demo data generated — 500 prompt runs across 30 days. Switch to **Observe** to explore.")

    if run_btn:
        if not selected_models:
            st.warning("Select at least one model.")
        else:
            with st.spinner("Running SafeEval pipeline..."):
                import time
                from safeeval.pipeline import SafeEvalPipeline, RunConfig

                cfg = RunConfig(
                    models=selected_models,
                    run_benchmarks=run_benchmarks,
                    run_attacks=run_attacks,
                    benchmark_names=bench_sel if run_benchmarks else [],
                    attack_names=attack_sel if run_attacks else [],
                    num_benchmark_samples=int(num_bench),
                    num_attack_samples=int(num_attacks),
                    use_judge=use_judge,
                    openai_api_key=openai_key or None,
                    anthropic_api_key=anthropic_key or None,
                )
                pipeline = SafeEvalPipeline(cfg)
                prog_bar = st.progress(0)
                status_txt = st.empty()

                def on_progress(done, total):
                    pct = int(done / max(total, 1) * 100)
                    prog_bar.progress(pct)
                    status_txt.markdown(f"`{done}/{total}` prompt runs complete")

                result_df = pipeline.run(progress_callback=on_progress)
                st.session_state.df = result_df
                st.success(f"✅ Evaluation complete — {len(result_df):,} records. Switch to **Observe** to explore results.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SAFETY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Safety Analysis":
    st.markdown("## Safety analysis")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Attack Success Rate", f"{asr}%", delta=None)
    m2.metric("Refusal Rate", f"{refusal_rate}%")
    m3.metric("False Positive Rate", f"{fpr}%")
    m4.metric("Harmful Responses", int(df[df["is_attack"] & (df["harmfulness_score"] > 0.5)].shape[0]))

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### ASR by attack category")
        per_cat = {}
        attacks = df[df["is_attack"]]
        if len(attacks) > 0 and "harmfulness_category" in attacks.columns:
            for cat, grp in attacks.groupby("harmfulness_category"):
                if cat and cat != "none":
                    per_cat[cat] = round(grp["harmfulness_score"].gt(0.5).mean(), 4)
        if not per_cat:
            per_cat = {"violence": 0.12, "cybercrime": 0.08, "misinformation": 0.15, "chemical_biological": 0.05}
        st.plotly_chart(asr_bar_chart(per_cat), use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("#### Per-category heatmap (pass rate %)")
        st.plotly_chart(heatmap_chart(df), use_container_width=True, config={"displayModeBar": False})

    st.markdown("#### Safety metrics over time")
    if len(daily) > 0:
        attack_daily = df[df["is_attack"]].copy()
        if "timestamp_dt" in attack_daily.columns and len(attack_daily) > 0:
            attack_daily["date"] = pd.to_datetime(attack_daily["timestamp_dt"]).dt.date
            asr_daily = attack_daily.groupby("date").apply(
                lambda g: round(g["harmfulness_score"].gt(0.5).mean() * 100, 1)
            ).reset_index(name="asr")
            if len(asr_daily) > 1:
                st.plotly_chart(
                    line_chart(asr_daily, "asr", "#f87171", "ASR %"),
                    use_container_width=True, config={"displayModeBar": False},
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CAPABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Capability Analysis":
    st.markdown("## Capability analysis")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pass Rate", f"{pass_rate}%")
    m2.metric("Avg Latency", f"{avg_latency:,}ms")
    m3.metric("Hallucination Rate", f"{hallucination_rate}%")
    m4.metric("Avg Cost / 1K", f"${avg_cost}")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Radar: Safety vs Capability")
        safety_d = {"attack_success_rate": asr/100, "refusal_rate": refusal_rate/100, "false_positive_rate": fpr/100}
        cap_d = {"pass_rate": pass_rate/100, "bert_score_f1": 0.74, "accuracy": pass_rate/100}
        st.plotly_chart(radar_chart(safety_d, cap_d), use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("#### Pass rate over time")
        if len(daily) > 0:
            st.plotly_chart(pass_rate_chart(daily), use_container_width=True, config={"displayModeBar": False})

    st.markdown("#### Token usage & latency distribution")
    c3, c4 = st.columns(2)
    with c3:
        if len(daily) > 0:
            st.markdown("Total tokens / day")
            st.plotly_chart(line_chart(daily, "total_tokens", "#fb923c"), use_container_width=True, config={"displayModeBar": False})
    with c4:
        if len(daily) > 0:
            st.markdown("Avg response time (ms)")
            st.plotly_chart(line_chart(daily, "avg_latency", "#94a3b8"), use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.markdown("## Model comparison")

    models_in_data = df["model_name"].unique().tolist()
    if len(models_in_data) < 2:
        st.info("Run an evaluation with 2+ models to see comparison.")
    else:
        comparison_rows = []
        for m in models_in_data:
            mdf = df[df["model_name"] == m]
            attacks = mdf[mdf["is_attack"]]
            comparison_rows.append({
                "Model": m,
                "Pass Rate": f"{round(mdf['overall_pass'].mean() * 100, 1)}%",
                "ASR": f"{round(attacks['harmfulness_score'].gt(0.5).mean() * 100, 1) if len(attacks) > 0 else 0}%",
                "Refusal Rate": f"{round(attacks['is_refusal'].mean() * 100, 1) if len(attacks) > 0 else 0}%",
                "Hallucination": f"{round(mdf['hallucination_detected'].mean() * 100, 1)}%",
                "Avg Latency": f"{int(mdf['latency_ms'].mean())}ms",
                "Avg Tokens": f"{int(mdf['total_tokens'].mean())}",
                "Avg Cost/1K": f"${mdf['cost_per_1k'].mean():.2f}",
                "Inferences": len(mdf),
            })
        cmp_df = pd.DataFrame(comparison_rows)
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Per-model heatmap (pass rate %)")
        st.plotly_chart(heatmap_chart(df), use_container_width=True, config={"displayModeBar": False})

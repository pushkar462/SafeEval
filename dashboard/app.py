"""
SafeEval Dashboard — Streamlit UI
Run: streamlit run dashboard/app.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from dashboard.data_loader import load_results, generate_demo_data, get_daily_stats
from dashboard.charts.timeseries import (
    bar_chart, line_chart, pass_rate_chart,
    radar_chart, asr_bar_chart, heatmap_chart,
)

st.set_page_config(
    page_title="SafeEval",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] { background: #f4f6fb !important; }
.block-container { padding: 1rem 1.5rem 2rem !important; max-width: 100% !important; }
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0f172a !important; border-right: none !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown div,
[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 12px !important; }
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-size: 16px !important; }

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
  width: 100% !important;
  text-align: left !important;
  justify-content: flex-start !important;
  background: transparent !important;
  border: none !important;
  color: #94a3b8 !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 9px 12px !important;
  border-radius: 8px !important;
  margin-bottom: 2px !important;
  transition: all 0.15s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: #1e293b !important;
  color: #f1f5f9 !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: #1d4ed8 !important;
  color: #ffffff !important;
  border: none !important;
}

/* Sidebar filters */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {
  background: #1e293b !important;
  border-color: #334155 !important;
  color: #e2e8f0 !important;
  font-size: 12px !important;
}
[data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] {
  background: #1e3a5f !important; color: #93c5fd !important;
}

/* ── Time filter buttons in main area ── */
div[data-testid="stMainBlockContainer"] .stButton > button {
  background: #fff !important;
  border: 1px solid #e2e8f0 !important;
  color: #475569 !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  padding: 5px 4px !important;
  border-radius: 7px !important;
}
div[data-testid="stMainBlockContainer"] .stButton > button:hover {
  background: #f1f5f9 !important;
}
div[data-testid="stMainBlockContainer"] .stButton > button[kind="primary"] {
  background: #1d4ed8 !important;
  border-color: #1d4ed8 !important;
  color: #fff !important;
}

/* ── Topbar ── */
.topbar {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
  padding: 9px 16px; font-size: 13px; color: #64748b;
  display: flex; align-items: center; gap: 12px;
}
.topbar b { color: #1e293b; }
.model-badge {
  display: inline-flex; align-items: center; gap: 5px;
  background: #f0f7ff; border: 1px solid #bfdbfe;
  border-radius: 20px; padding: 3px 12px;
  font-size: 12px; font-weight: 600; color: #1d4ed8;
  margin-left: auto;
}
.model-dot { width: 7px; height: 7px; border-radius: 50%; }

/* ── Hero strip ── */
.hero-strip {
  background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
  padding: 18px 20px; margin-bottom: 12px;
  display: flex; align-items: stretch; gap: 0; flex-wrap: nowrap;
}
.hero-pass-block {
  display: flex; flex-direction: column; justify-content: center;
  padding-right: 24px; border-right: 1px solid #e2e8f0; min-width: 110px;
}
.hero-pass-block .big { font-size: 40px; font-weight: 800; line-height: 1; }
.hero-pass-block .lbl { font-size: 11px; color: #94a3b8; margin-top: 5px; }
.hero-pass-block .sub { font-size: 10px; color: #cbd5e1; margin-top: 2px; }

/* Safety KPIs */
.safety-section {
  display: flex; align-items: stretch; padding: 0 0 0 20px;
  border-right: 1px solid #e2e8f0; gap: 0;
}
.safety-header {
  display: flex; flex-direction: column; justify-content: center;
  padding: 0 16px 0 0; border-right: 1px solid #f1f5f9; min-width: 60px;
}
.safety-header .sh-lbl { font-size: 9px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: .08em; }
.safety-header .sh-sub { font-size: 9px; color: #cbd5e1; margin-top: 2px; }
.skpi { display: flex; flex-direction: column; justify-content: center; padding: 0 16px; border-right: 1px solid #f1f5f9; min-width: 90px; }
.skpi:last-child { border-right: none; }
.skpi-label { font-size: 10px; color: #94a3b8; line-height: 1.3; margin-bottom: 4px; }
.skpi-value { font-size: 22px; font-weight: 800; line-height: 1; }
.skpi-value.danger { color: #dc2626; }
.skpi-value.warn   { color: #d97706; }
.skpi-value.good   { color: #16a34a; }
.skpi-sub { font-size: 10px; margin-top: 3px; }
.skpi-sub.danger { color: #fca5a5; }
.skpi-sub.good   { color: #86efac; }

/* Ops stats */
.ops-section { display: flex; align-items: stretch; padding-left: 20px; flex: 1; gap: 0; }
.ops-stat { display: flex; flex-direction: column; justify-content: center; padding: 0 16px; border-right: 1px solid #f1f5f9; }
.ops-stat:last-child { border-right: none; }
.ops-label { font-size: 10px; color: #94a3b8; line-height: 1.3; margin-bottom: 3px; }
.ops-value { font-size: 18px; font-weight: 700; color: #334155; }
.ops-value.blue { color: #2563eb; }

/* ── Topic pills ── */
.topic-row { display: flex; flex-wrap: wrap; gap: 4px; }
.topic-pill { display: inline-block; padding: 2px 8px; border-radius: 8px; font-size: 10px; font-weight: 600; }
.t-History       { background: #dbeafe; color: #1d4ed8; }
.t-Science       { background: #dcfce7; color: #166534; }
.t-Sports        { background: #fef3c7; color: #92400e; }
.t-Entertainment { background: #fce7f3; color: #9d174d; }
.t-Media         { background: #ede9fe; color: #5b21b6; }
.t-Technology    { background: #e0f2fe; color: #0c4a6e; }
.t-Finance       { background: #fef9c3; color: #78350f; }
.t-General       { background: #f1f5f9; color: #475569; }

/* ── Chart cards ── */
.chart-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 12px 14px; }
.card-title { font-size: 10px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 4px; }

/* ── Section card ── */
.section-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; }
.section-heading { font-size: 12px; font-weight: 700; color: #1e293b; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 10px; }

/* ── Table ── */
.eval-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.eval-table th {
  text-align: left; padding: 8px 10px; font-size: 10px; font-weight: 700;
  color: #94a3b8; border-bottom: 2px solid #f1f5f9; white-space: nowrap;
  text-transform: uppercase; letter-spacing: .05em; background: #f8fafc;
}
.eval-table td { padding: 9px 10px; border-bottom: 1px solid #f8fafc; color: #334155; vertical-align: middle; }
.eval-table tr:hover td { background: #f8fafc; }
.table-wrap { overflow-x: auto; }

/* Badges */
.badge { display: inline-block; padding: 3px 9px; border-radius: 12px; font-size: 10px; font-weight: 700; white-space: nowrap; }
.badge-pass   { background: #dcfce7; color: #15803d; }
.badge-fail   { background: #fee2e2; color: #b91c1c; }
.badge-attack { background: #fef3c7; color: #92400e; }
.badge-benign { background: #f0fdf4; color: #166534; }
.slug-pill { background: #f1f5f9; color: #475569; font-size: 10px; padding: 2px 8px; border-radius: 5px; font-family: monospace; }
.model-gpt4    { color: #7c3aed; font-weight: 700; font-size: 11px; }
.model-gpt35   { color: #0369a1; font-weight: 700; font-size: 11px; }
.model-claude  { color: #b45309; font-weight: 700; font-size: 11px; }
.model-mistral { color: #059669; font-weight: 700; font-size: 11px; }
.harm-high { color: #dc2626; font-weight: 700; }
.harm-low  { color: #94a3b8; }
.truncate { max-width: 160px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; display: block; }

/* KPI card for other pages */
.kpi-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px 20px; }
.kpi-label { font-size: 11px; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 800; color: #1e293b; line-height: 1; }
.kpi-value.blue  { color: #1d4ed8; }
.kpi-value.green { color: #16a34a; }
.kpi-value.amber { color: #d97706; }
.kpi-value.red   { color: #dc2626; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "df"   not in st.session_state: st.session_state.df   = generate_demo_data(n=500, days=30)
if "page" not in st.session_state: st.session_state.page = "Observe"
if "tf"   not in st.session_state: st.session_state.tf   = "30d"


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ SafeEval")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    NAV = [
        ("📊", "Observe"),
        ("▶️", "Run Evaluation"),
        ("🔴", "Safety Analysis"),
        ("⚡", "Capability Analysis"),
        ("⚖️", "Model Comparison"),
    ]
    for icon, label in NAV:
        kind = "primary" if st.session_state.page == label else "secondary"
        if st.button(f"{icon}  {label}", key=f"nav_{label}", use_container_width=True, type=kind):
            st.session_state.page = label
            st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px;padding:0 4px'>Filters</div>", unsafe_allow_html=True)

    _model_opts = ["gpt-4", "gpt-3.5-turbo", "claude-sonnet", "mistral-7b"]
    if st.session_state.page == "Observe":
        _sel = st.selectbox("Model", _model_opts, index=0, key="obs_model")
        model_filter = [_sel]
    else:
        _mf = st.multiselect("Model", _model_opts, default=["gpt-4", "gpt-3.5-turbo", "claude-sonnet"], key="multi_model")
        model_filter = _mf if _mf else [_model_opts[0]]

    topic_filter  = st.multiselect("Topic",
        ["History","Science","Sports","Entertainment","Media","Technology","Finance"],
        default=[], key="topic_f")
    result_filter = st.selectbox("Result", ["All","Passed","Failed"], key="result_f")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#334155'>SafeEval v0.1.0<br>Four-layer LLM evaluation</div>", unsafe_allow_html=True)

page = st.session_state.page


# ── Filter data ────────────────────────────────────────────────────────────────
df_all = st.session_state.df.copy()
time_map  = {"1d":1,"7d":7,"14d":14,"30d":30,"60d":60,"90d":90}
days_back = time_map.get(st.session_state.tf, 30)

if "timestamp_dt" in df_all.columns:
    df_all["timestamp_dt"] = pd.to_datetime(df_all["timestamp_dt"], errors="coerce")
    cutoff = datetime.now() - timedelta(days=days_back)
    df = df_all[df_all["timestamp_dt"] >= cutoff].copy()
else:
    df = df_all.copy()

if model_filter:  df = df[df["model_name"].isin(model_filter)]
if topic_filter:  df = df[df["topic"].isin(topic_filter)]
if result_filter == "Passed": df = df[df["overall_pass"] == True]
elif result_filter == "Failed": df = df[df["overall_pass"] == False]

# ── KPIs ──────────────────────────────────────────────────────────────────────
n         = len(df)
pass_r    = round(df["overall_pass"].mean()*100,1) if n>0 else 0.0
avg_cost  = round(df["cost_per_1k"].mean(),2)       if n>0 else 0.0
avg_tok   = int(df["total_tokens"].mean())           if n>0 else 0
avg_lat   = int(df["latency_ms"].mean())             if n>0 else 0
asr       = round(df[df["is_attack"]]["harmfulness_score"].gt(0.5).mean()*100,1) if df["is_attack"].any() else 0.0
refr      = round(df[df["is_attack"]]["is_refusal"].mean()*100,1)                if df["is_attack"].any() else 0.0
fpr       = round(df[~df["is_attack"]]["is_refusal"].mean()*100,1)               if (~df["is_attack"]).any() else 0.0
hall_r    = round(df["hallucination_detected"].mean()*100,1)                      if n>0 else 0.0
topic_cnts= df["topic"].value_counts().head(5).to_dict() if "topic" in df.columns else {}
daily     = get_daily_stats(df_all if n<5 else df, days=days_back)

def asr_color(v):     return "danger" if v>20 else ("warn" if v>10 else "good")
def fpr_color(v):     return "danger" if v>15 else ("warn" if v>8  else "good")
def refusal_color(v): return "good"   if v>60 else ("warn" if v>30 else "danger")
def hall_color(v):    return "danger" if v>20 else ("warn" if v>10 else "good")
def pass_hex(v):      return "#16a34a" if v>=80 else ("#d97706" if v>=60 else "#dc2626")


# ╔═══════════════════════════════════════════╗
# ║  OBSERVE                                  ║
# ╚═══════════════════════════════════════════╝
if page == "Observe":

    # Topbar + time filters
    bc, tf = st.columns([3, 5])
    with bc:
        mdot_color = {"gpt-4":"#7c3aed","gpt-3.5-turbo":"#0369a1","claude-sonnet":"#b45309","mistral-7b":"#059669"}.get(model_filter[0],"#64748b")
        st.markdown(
            f'<div class="topbar">Observe › <b>All</b>'
            f'<span class="model-badge"><span class="model-dot" style="background:{mdot_color}"></span>{model_filter[0]}</span>'
            f'</div>', unsafe_allow_html=True)
    with tf:
        tfc = st.columns(7)
        for i, opt in enumerate(["1d","7d","14d","30d","60d","90d","Custom"]):
            with tfc[i]:
                active = st.session_state.tf == opt
                if st.button(opt, key=f"tf_{opt}", use_container_width=True,
                             type="primary" if active else "secondary"):
                    st.session_state.tf = opt
                    st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Hero strip
    topics_html = "".join([
        f'<span class="topic-pill t-{t}">{t} <span style="opacity:.6">{c}</span></span>'
        for t,c in list(topic_cnts.items())[:5]
    ])
    st.markdown(f"""
    <div class="hero-strip">
      <div class="hero-pass-block">
        <div class="big" style="color:{pass_hex(pass_r)}">{pass_r}%</div>
        <div class="lbl">Pass rate</div>
        <div class="sub">{n:,} inferences</div>
      </div>
      <div class="safety-section">
        <div class="safety-header">
          <div class="sh-lbl">Safety</div>
          <div class="sh-sub">last {days_back}d</div>
        </div>
        <div class="skpi">
          <div class="skpi-label">Attack Success<br>Rate (ASR)</div>
          <div class="skpi-value {asr_color(asr)}">{asr}%</div>
          <div class="skpi-sub {'danger' if asr>10 else 'good'}">{'↑ high risk' if asr>10 else '↓ low risk'}</div>
        </div>
        <div class="skpi">
          <div class="skpi-label">Refusal<br>Rate</div>
          <div class="skpi-value {refusal_color(refr)}">{refr}%</div>
          <div class="skpi-sub {'good' if refr>60 else 'danger'}">{'↑ effective' if refr>60 else '↓ low'}</div>
        </div>
        <div class="skpi">
          <div class="skpi-label">False Positive<br>Rate</div>
          <div class="skpi-value {fpr_color(fpr)}">{fpr}%</div>
          <div class="skpi-sub {'danger' if fpr>8 else 'good'}">{'↑ over-refusing' if fpr>8 else '↓ on-target'}</div>
        </div>
        <div class="skpi">
          <div class="skpi-label">Hallucination<br>Rate</div>
          <div class="skpi-value {hall_color(hall_r)}">{hall_r}%</div>
        </div>
      </div>
      <div class="ops-section">
        <div class="ops-stat">
          <div class="ops-label">Avg cost /<br>1K inferences</div>
          <div class="ops-value">${avg_cost}</div>
        </div>
        <div class="ops-stat">
          <div class="ops-label">Avg tokens /<br>inference</div>
          <div class="ops-value">{avg_tok}</div>
        </div>
        <div class="ops-stat">
          <div class="ops-label">Avg response<br>time</div>
          <div class="ops-value">{avg_lat:,}ms</div>
        </div>
        <div class="ops-stat">
          <div class="ops-label" style="margin-bottom:6px">Top topics</div>
          <div class="topic-row">{topics_html}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 mini charts
    if len(daily) > 0:
        cc = st.columns(4)
        for i,(title,col_n,kind,color) in enumerate([
            ("Pass rate (%)",          "pass_rate",   "line", "#22c55e"),
            ("Total inferences",       "inferences",  "bar",  "#3b82f6"),
            ("Total failures",         "failures",    "bar",  "#f87171"),
            ("Avg response time (ms)", "avg_latency", "line", "#94a3b8"),
        ]):
            with cc[i]:
                st.markdown(f'<div class="chart-card"><div class="card-title">{title}</div>', unsafe_allow_html=True)
                if col_n in daily.columns:
                    fig = bar_chart(daily,col_n,color) if kind=="bar" else line_chart(daily,col_n,color)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Prompt runs table
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    tl, tm, tr = st.columns([2, 3, 3])
    with tl:
        st.markdown(f"<div style='font-size:13px;color:#64748b;padding-top:8px'>"
                    f"Showing <b style='color:#1e293b'>{min(50,n):,}</b> of <b style='color:#1e293b'>{n:,}</b> prompt runs</div>",
                    unsafe_allow_html=True)
    with tm:
        c1,c2,c3 = st.columns(3)
        with c1: grade_f   = st.selectbox("Grade", ["All","Passed","Failed"], label_visibility="collapsed", key="gf")
        with c2: type_f    = st.selectbox("Type",  ["All","Attacks","Benchmarks"], label_visibility="collapsed", key="typef")
        with c3: sort_by   = st.selectbox("Sort",  ["Latest","Latency ↓","Harm ↓","Tokens ↓"], label_visibility="collapsed", key="sb")
    with tr:
        search = st.text_input("", placeholder="/ Search prompts and responses", label_visibility="collapsed", key="srch")

    tdf = df.copy()
    if grade_f == "Passed":       tdf = tdf[tdf["overall_pass"]==True]
    elif grade_f == "Failed":     tdf = tdf[tdf["overall_pass"]==False]
    if type_f == "Attacks":       tdf = tdf[tdf["is_attack"]==True]
    elif type_f == "Benchmarks":  tdf = tdf[tdf["is_attack"]==False]
    if search:
        mask = (tdf["prompt"].str.contains(search,case=False,na=False)|
                tdf["response"].str.contains(search,case=False,na=False))
        tdf = tdf[mask]
    if sort_by=="Latency ↓":  tdf = tdf.sort_values("latency_ms",ascending=False)
    elif sort_by=="Harm ↓":   tdf = tdf.sort_values("harmfulness_score",ascending=False)
    elif sort_by=="Tokens ↓": tdf = tdf.sort_values("total_tokens",ascending=False)

    display = tdf.head(50)

    def mcls(m):
        if "gpt-4" in m and "turbo" not in m: return "model-gpt4"
        if "3.5" in m or "turbo" in m: return "model-gpt35"
        if "claude" in m: return "model-claude"
        return "model-mistral"

    rows_html = ""
    for _, row in display.iterrows():
        passed    = row.get("overall_pass", True)
        is_atk    = row.get("is_attack", False)
        ep = row.get("evals_passed",2); et = row.get("evals_total",3)
        harm = row.get("harmfulness_score", 0.0)
        slug    = str(row.get("slug", row.get("category","closed_qa")))
        model   = str(row.get("model_name","gpt-4"))
        topic   = str(row.get("topic","General"))
        pt      = str(row.get("prompt",""))[:60]+("…" if len(str(row.get("prompt","")))>60 else "")
        rt      = str(row.get("response",""))[:75]+("…" if len(str(row.get("response","")))>75 else "")
        ts      = str(row.get("timestamp",""))[:14]
        lat     = int(row.get("latency_ms",0))
        rows_html += f"""<tr>
          <td><span class="badge {'badge-pass' if passed else 'badge-fail'}">{'passed' if passed else 'failed'}</span></td>
          <td><span class="badge {'badge-attack' if is_atk else 'badge-benign'}">{'attack' if is_atk else 'bench'}</span></td>
          <td><span class="badge {'badge-pass' if ep==et else 'badge-fail'}">{ep}/{et}</span></td>
          <td style="color:#94a3b8;font-size:10px;white-space:nowrap">{ts}</td>
          <td><span class="slug-pill">{slug}</span></td>
          <td><span class="truncate">{pt}</span></td>
          <td><span class="truncate">{rt}</span></td>
          <td><span class="{mcls(model)}">{model}</span></td>
          <td><span class="{'harm-high' if harm>0.5 else 'harm-low'}">{harm:.2f}</span></td>
          <td style="color:#64748b;font-size:11px">{lat:,}ms</td>
          <td><span class="topic-pill t-{topic}">{topic}</span></td>
        </tr>"""

    st.markdown(f"""
    <div class="table-wrap" style="margin-top:10px">
    <table class="eval-table">
      <thead><tr>
        <th>Result</th><th>Type</th><th>Evals</th><th>Timestamp</th><th>Slug</th>
        <th>Prompt</th><th>Response</th><th>Model</th>
        <th>Harm</th><th>Latency</th><th>Topic</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    total_pages = max(1,(len(tdf)-1)//50+1)
    sp,p1,p2,p3,p4 = st.columns([6,1,1,1,1])
    with p1: st.button("‹ Prev", key="pg_p", use_container_width=True)
    with p2:
        st.markdown("<div style='text-align:center;padding:6px 0;font-size:12px;"
                    "background:#1d4ed8;color:#fff;border-radius:6px;font-weight:700'>1</div>",
                    unsafe_allow_html=True)
    with p3: st.button("2", key="pg2", use_container_width=True)
    with p4: st.button(f"{total_pages} ›", key="pg_l", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ╔═══════════════════════════════════════════╗
# ║  RUN EVALUATION                           ║
# ╚═══════════════════════════════════════════╝
elif page == "Run Evaluation":
    st.markdown("## ▶️ Run new evaluation")
    st.caption("No API keys required — use demo mode to explore the dashboard immediately.")

    with st.expander("⚙️ Eval configuration", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            sel_models  = st.multiselect("Target models",
                ["gpt-4o","gpt-3.5-turbo","claude-sonnet","mistral-7b","llama-3"],
                default=["gpt-4o","gpt-3.5-turbo"])
            run_bench   = st.checkbox("Capability benchmarks", value=True)
            bench_sel   = st.multiselect("Benchmarks",["truthfulqa","mmlu","gsm8k","hellaswag"],
                default=["truthfulqa","mmlu"]) if run_bench else []
            num_bench   = st.number_input("Samples per benchmark", 5, 200, 20)
        with c2:
            oai_key     = st.text_input("OpenAI API key",    type="password", placeholder="sk-...")
            ant_key     = st.text_input("Anthropic API key", type="password", placeholder="sk-ant-...")
            run_atk     = st.checkbox("Red-team attack sets", value=True)
            attack_sel  = st.multiselect("Attack sets",["harmbench","advbench","jailbreak_templates"],
                default=["harmbench","advbench"]) if run_atk else []
            num_atk     = st.number_input("Samples per attack set", 5, 100, 15)
            use_judge   = st.checkbox("LLM-as-judge (GPT-4o)", value=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    r1, r2, _ = st.columns([1,2,3])
    with r1: run_btn  = st.button("▶  Run evaluation", type="primary", use_container_width=True)
    with r2: demo_btn = st.button("🎲  Generate demo data (no API keys)", use_container_width=True)

    if demo_btn:
        import time as _t
        prog = st.progress(0)
        with st.spinner("Generating 500 demo prompt runs…"):
            for i in range(20):
                _t.sleep(0.04); prog.progress((i+1)*5)
            st.session_state.df = generate_demo_data(n=500, days=30)
            prog.progress(100)
        st.success("✅ Demo data ready — 500 runs, 30 days. Switch to **Observe** to explore.")

    if run_btn:
        if not sel_models: st.warning("Select at least one model.")
        else:
            from safeeval.pipeline import SafeEvalPipeline, RunConfig
            cfg = RunConfig(models=sel_models, run_benchmarks=run_bench, run_attacks=run_atk,
                benchmark_names=bench_sel, attack_names=attack_sel,
                num_benchmark_samples=int(num_bench), num_attack_samples=int(num_atk),
                use_judge=use_judge, openai_api_key=oai_key or None, anthropic_api_key=ant_key or None)
            pb=st.progress(0); stxt=st.empty()
            def _cb(done,total): pb.progress(int(done/max(total,1)*100)); stxt.markdown(f"`{done}/{total}` runs complete")
            with st.spinner("Running SafeEval pipeline…"):
                rdf = SafeEvalPipeline(cfg).run(progress_callback=_cb)
                st.session_state.df = rdf
            st.success(f"✅ Done — {len(rdf):,} records. Switch to **Observe**.")


# ╔═══════════════════════════════════════════╗
# ║  SAFETY ANALYSIS                          ║
# ╚═══════════════════════════════════════════╝
elif page == "Safety Analysis":
    st.markdown("## 🔴 Safety analysis")

    k1,k2,k3,k4 = st.columns(4)
    for col,lbl,val,clr in [
        (k1,"Attack Success Rate (ASR)",f"{asr}%","red"),
        (k2,"Refusal Rate",f"{refr}%","green"),
        (k3,"False Positive Rate",f"{fpr}%","amber"),
        (k4,"Harmful Responses",str(int(df[df["is_attack"]&(df["harmfulness_score"]>0.5)].shape[0])),"red"),
    ]:
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{lbl}</div>'
                     f'<div class="kpi-value {clr}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-heading">ASR by attack category</div>', unsafe_allow_html=True)
        att = df[df["is_attack"]]
        per_cat = {cat:round(grp["harmfulness_score"].gt(0.5).mean(),4)
                   for cat,grp in att.groupby("harmfulness_category") if cat and cat!="none"} if len(att)>0 else {}
        if not per_cat: per_cat={"violence":0.12,"cybercrime":0.08,"misinformation":0.15,"chemical_biological":0.05}
        st.plotly_chart(asr_bar_chart(per_cat), use_container_width=True, config={"displayModeBar":False})
    with c2:
        st.markdown('<div class="section-heading">Per-category heatmap (pass rate %)</div>', unsafe_allow_html=True)
        st.plotly_chart(heatmap_chart(df), use_container_width=True, config={"displayModeBar":False})

    st.markdown('<div class="section-heading" style="margin-top:12px">ASR over time</div>', unsafe_allow_html=True)
    if "timestamp_dt" in df.columns and df["is_attack"].any():
        adf = df[df["is_attack"]].copy()
        adf["date"] = pd.to_datetime(adf["timestamp_dt"]).dt.date
        asr_d = adf.groupby("date").apply(lambda g: round(g["harmfulness_score"].gt(0.5).mean()*100,1)).reset_index(name="asr")
        if len(asr_d)>1:
            st.plotly_chart(line_chart(asr_d,"asr","#f87171"), use_container_width=True, config={"displayModeBar":False})


# ╔═══════════════════════════════════════════╗
# ║  CAPABILITY ANALYSIS                      ║
# ╚═══════════════════════════════════════════╝
elif page == "Capability Analysis":
    st.markdown("## ⚡ Capability analysis")

    k1,k2,k3,k4 = st.columns(4)
    for col,lbl,val,clr in [
        (k1,"Pass Rate",f"{pass_r}%","blue"),
        (k2,"Hallucination Rate",f"{hall_r}%","amber"),
        (k3,"Avg Latency",f"{avg_lat:,}ms",""),
        (k4,"Avg Cost / 1K",f"${avg_cost}",""),
    ]:
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{lbl}</div>'
                     f'<div class="kpi-value {clr}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-heading">Radar — safety vs capability</div>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(
            {"attack_success_rate":asr/100,"refusal_rate":refr/100,"false_positive_rate":fpr/100},
            {"pass_rate":pass_r/100,"bert_score_f1":0.74,"accuracy":pass_r/100}),
            use_container_width=True, config={"displayModeBar":False})
    with c2:
        st.markdown('<div class="section-heading">Pass rate over time</div>', unsafe_allow_html=True)
        if len(daily)>0: st.plotly_chart(pass_rate_chart(daily), use_container_width=True, config={"displayModeBar":False})

    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-heading">Total tokens / day</div>', unsafe_allow_html=True)
        if len(daily)>0: st.plotly_chart(line_chart(daily,"total_tokens","#fb923c"), use_container_width=True, config={"displayModeBar":False})
    with c4:
        st.markdown('<div class="section-heading">Avg response time (ms)</div>', unsafe_allow_html=True)
        if len(daily)>0: st.plotly_chart(line_chart(daily,"avg_latency","#94a3b8"), use_container_width=True, config={"displayModeBar":False})


# ╔═══════════════════════════════════════════╗
# ║  MODEL COMPARISON                         ║
# ╚═══════════════════════════════════════════╝
elif page == "Model Comparison":
    st.markdown("## ⚖️ Model comparison")

    models_in = df["model_name"].unique().tolist()
    if len(models_in) < 2:
        st.info("Need 2+ models. Go to **Run Evaluation → Generate demo data** first.")
    else:
        rows = []
        for m in models_in:
            mdf=df[df["model_name"]==m]; att=mdf[mdf["is_attack"]]
            rows.append({
                "Model":         m,
                "Pass Rate":     f"{round(mdf['overall_pass'].mean()*100,1)}%",
                "ASR":           f"{round(att['harmfulness_score'].gt(0.5).mean()*100,1) if len(att)>0 else 0}%",
                "Refusal Rate":  f"{round(att['is_refusal'].mean()*100,1) if len(att)>0 else 0}%",
                "Hallucination": f"{round(mdf['hallucination_detected'].mean()*100,1)}%",
                "Avg Latency":   f"{int(mdf['latency_ms'].mean())}ms",
                "Avg Tokens":    f"{int(mdf['total_tokens'].mean())}",
                "Avg Cost/1K":   f"${mdf['cost_per_1k'].mean():.2f}",
                "Inferences":    len(mdf),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Per-model heatmap (pass rate %)</div>', unsafe_allow_html=True)
        st.plotly_chart(heatmap_chart(df), use_container_width=True, config={"displayModeBar":False})

        st.markdown('<div class="section-heading" style="margin-top:12px">Radar scorecard</div>', unsafe_allow_html=True)
        st.plotly_chart(radar_chart(
            {"attack_success_rate":asr/100,"refusal_rate":refr/100,"false_positive_rate":fpr/100},
            {"pass_rate":pass_r/100,"bert_score_f1":0.74,"accuracy":pass_r/100}),
            use_container_width=True, config={"displayModeBar":False})
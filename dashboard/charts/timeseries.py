"""Plotly time-series charts for the dashboard."""
import plotly.graph_objects as go
import pandas as pd

CHART_LAYOUT = dict(
    margin=dict(l=8, r=8, t=8, b=8),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
    xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
               tickfont=dict(size=10, color="#0f172a"), showticklabels=True),
    height=130,
)


def _base_fig():
    fig = go.Figure()
    fig.update_layout(**CHART_LAYOUT)
    return fig


def bar_chart(daily: pd.DataFrame, col: str, color: str, name: str = "") -> go.Figure:
    fig = _base_fig()
    fig.add_trace(go.Bar(
        x=daily["date"].astype(str),
        y=daily[col],
        marker_color=color,
        marker_line_width=0,
        name=name,
        hovertemplate="%{y}<extra></extra>",
    ))
    fig.update_traces(marker_cornerradius=2)
    return fig


def line_chart(daily: pd.DataFrame, col: str, color: str, name: str = "", fill: bool = False) -> go.Figure:
    fig = _base_fig()
    fig.add_trace(go.Scatter(
        x=daily["date"].astype(str),
        y=daily[col],
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy" if fill else "none",
        fillcolor=color.replace(")", ",0.12)").replace("rgb", "rgba") if fill else "rgba(0,0,0,0)",
        name=name,
        hovertemplate="%{y}<extra></extra>",
    ))
    return fig


def pass_rate_chart(daily: pd.DataFrame) -> go.Figure:
    fig = _base_fig()
    fig.add_trace(go.Scatter(
        x=daily["date"].astype(str),
        y=daily["pass_rate"],
        mode="lines",
        line=dict(color="#f87171", width=1.5),
        hovertemplate="%{y}%<extra></extra>",
    ))
    fig.update_layout(yaxis=dict(
        showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False,
        tickfont=dict(size=9, color="#999"), range=[0, 105],
    ))
    return fig


def radar_chart(safety: dict, capability: dict) -> go.Figure:
    categories = ["Pass Rate", "Low ASR", "Refusal Rate", "Low FPR", "Truthfulness", "BERTScore"]
    values = [
        capability.get("pass_rate", 0) * 100,
        (1 - safety.get("attack_success_rate", 0)) * 100,
        safety.get("refusal_rate", 0) * 100,
        (1 - safety.get("false_positive_rate", 0)) * 100,
        capability.get("bert_score_f1", 0) * 100,
        capability.get("accuracy", 0) * 100,
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(26,108,245,0.15)",
        line=dict(color="#1a6cf5", width=2),
        name="Scores",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(0,0,0,0.1)",
                tickfont=dict(size=11, color="#0f172a"),
            ),
            angularaxis=dict(gridcolor="rgba(0,0,0,0.1)", tickfont=dict(size=11, color="#0f172a")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
        height=300,
        showlegend=False,
        font=dict(color="#0f172a"),
    )
    return fig


def asr_bar_chart(per_cat: dict) -> go.Figure:
    if not per_cat:
        per_cat = {"violence": 0.12, "cybercrime": 0.08, "misinformation": 0.15, "chemical_biological": 0.05}
    cats = list(per_cat.keys())
    vals = [round(v * 100, 1) for v in per_cat.values()]
    colors = ["#f87171" if v > 15 else "#fb923c" if v > 8 else "#4ade80" for v in vals]
    fig = go.Figure(go.Bar(
        x=cats, y=vals,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="%{x}: %{y}%<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", ticksuffix="%", tickfont=dict(size=10)),
        height=220,
    )
    return fig


def heatmap_chart(df: pd.DataFrame) -> go.Figure:
    if df is None or len(df) == 0 or "model_name" not in df.columns or "topic" not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=220,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(text="No data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
        )
        return fig

    models = df["model_name"].dropna().unique().tolist()
    topics = df["topic"].dropna().unique().tolist()[:6]
    if not models or not topics:
        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=220,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(text="No data", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
        )
        return fig
    z = []
    for m in models:
        row = []
        for t in topics:
            sub = df[(df["model_name"] == m) & (df["topic"] == t)]
            row.append(round(sub["overall_pass"].mean() * 100, 1) if len(sub) > 0 else 0)
        z.append(row)
    fig = go.Figure(go.Heatmap(
        z=z, x=topics, y=models,
        colorscale=[
            [0.0, "#dc2626"],   # red
            [0.5, "#f59e0b"],   # amber
            [1.0, "#16a34a"],   # green
        ],
        zmin=0, zmax=100,
        hovertemplate="%{y} / %{x}: %{z}%<extra></extra>",
        colorbar=dict(thickness=12, tickfont=dict(size=11, color="#0f172a"), ticksuffix="%"),
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220,
        xaxis=dict(tickfont=dict(size=12, color="#0f172a")),
        yaxis=dict(tickfont=dict(size=12, color="#0f172a")),
        font=dict(color="#0f172a"),
    )
    return fig

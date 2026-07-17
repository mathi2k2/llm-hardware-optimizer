"""
LLM Hardware & Cost Optimization Engine - Streamlit dashboard.

Architecture (kept deliberately layered):

    DATA LAYER      load_data()                      csv ingestion / generation
    FILTER LAYER    filter_data(), pareto_frontier() pure dataframe -> dataframe
    CHART LAYER     make_*()                         dataframes -> figures
    UI LAYER        render_*(), main()               streamlit widgets & layout

Run:  streamlit run app.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap
from plotly.colors import sample_colorscale

import research
from generate_data import CTX_STEPS, GPU_SPECS, OUTPUT_CSV, generate_dataset

# --------------------------------------------------------------------------- #
#  Theme constants (kept in sync with .streamlit/config.toml and CUSTOM_CSS)
# --------------------------------------------------------------------------- #

C = dict(
    bg="#0E1117",       # page background
    panel="#131826",    # chart plot area
    grid="#1F2637",     # recessive gridlines
    axis="#2A3550",     # axis lines
    ink="#E6EDF3",      # primary text
    muted="#8B98AB",    # secondary text
    dim="#3A4356",      # excluded / de-emphasized marks
    cyan="#00CFF0",
    purple="#A78BFF",
    green="#2EE884",
    magenta="#FF4D9D",
    amber="#F5B84D",
    blue="#4D9FFF",
)

#  With 19 models, series color follows the model FAMILY (9 families —
#  matching the 9-slot palette validated for CVD + normal-vision separation
#  on the #0E1117 surface, in this interleaved order). Colors follow the
#  entity, never its rank; exact model identity lives in the hover.
FAMILY_PALETTE = ["#00CFF0", "#FF8A5C", "#A78BFF", "#C8F05A", "#FF4D9D",
                  "#2EE884", "#FF9EC8", "#4D9FFF", "#F5B84D"]
FAMILIES = ["DeepSeek", "GLM", "Gemma", "Llama", "Mistral",
            "Moonshot", "OpenAI", "Phi", "Qwen"]
FAMILY_COLORS = dict(zip(FAMILIES, FAMILY_PALETTE))

QUANT_SYMBOLS = {"FP16": "circle", "INT8": "diamond", "INT4": "square"}

FONT = "Inter, 'Segoe UI', system-ui, sans-serif"

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] * { font-family: 'Inter','Segoe UI',system-ui,sans-serif; }
[data-testid="stAppViewContainer"] {
    background: radial-gradient(1100px 500px at 12% -8%, #121A2B 0%, #0E1117 55%);
}
.block-container { padding-top: 2.3rem; max-width: 1300px; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] { background: #0B0E15; border-right: 1px solid #1E2637; }

/* ---- KPI metric cards ---- */
[data-testid="stMetric"] {
    background: linear-gradient(165deg, #151C2C 0%, #10141F 100%);
    border: 1px solid #232E45;
    border-radius: 14px;
    padding: 16px 18px 12px;
}
[data-testid="stMetricLabel"] p {
    color: #8B98AB !important;
    text-transform: uppercase;
    letter-spacing: .09em;
    font-size: .70rem !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] { color: #00CFF0; font-weight: 700; }
[data-testid="stMetricDelta"] { color: #8B98AB !important; }
[data-testid="stMetricDelta"] svg { display: none; }   /* config names, not deltas */

/* ---- typography blocks ---- */
.hero-title { font-size: 2.05rem; font-weight: 800; letter-spacing: -.02em;
              color: #E6EDF3; line-height: 1.15; }
.hero-title .accent {
    background: linear-gradient(90deg, #00CFF0 0%, #A78BFF 55%, #2EE884 110%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub { color: #8B98AB; font-size: .95rem; margin: .35rem 0 1.0rem; }
.section-title { font-size: 1.08rem; font-weight: 700; color: #E6EDF3;
                 border-left: 3px solid #00CFF0; padding-left: 10px;
                 margin: .8rem 0 .1rem; }
.section-title.green  { border-color: #2EE884; }
.section-title.purple { border-color: #A78BFF; }
.section-sub { color: #8B98AB; font-size: .83rem; padding-left: 13px; margin-bottom: .4rem; }

/* ---- containers ---- */
[data-testid="stExpander"] { background: #11151F; border: 1px solid #1E2637; border-radius: 12px; }
[data-testid="stExpander"] summary p { font-weight: 600; color: #B7C2D4; }
.stDownloadButton button { border: 1px solid #232E45; background: #141A29;
                           color: #B7C2D4; border-radius: 10px; }
</style>
"""


# --------------------------------------------------------------------------- #
#  DATA LAYER
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner="Loading benchmark data...")
def load_data(csv_path: str = str(OUTPUT_CSV)) -> pd.DataFrame:
    """Read the benchmark CSV; regenerate it deterministically if missing."""
    path = Path(csv_path)
    if not path.exists():
        return generate_dataset(path)
    return pd.read_csv(path)


def model_order(df: pd.DataFrame) -> list[str]:
    """Models sorted small -> large (fixed categorical color order)."""
    return list(df.sort_values("params_b")["model"].unique())


def gpu_order() -> list[str]:
    """GPUs sorted by memory bandwidth (weakest -> strongest)."""
    return sorted(GPU_SPECS, key=lambda g: GPU_SPECS[g]["bw_gbs"])


# --------------------------------------------------------------------------- #
#  FILTER LAYER  (pure functions: dataframe in -> dataframe out)
# --------------------------------------------------------------------------- #

def filter_data(
    df: pd.DataFrame,
    min_tokens_sec: float,
    max_budget_hr: float,
    required_ctx: int,
    models: list[str],
    quants: list[str],
) -> pd.DataFrame:
    """Apply the user's workload requirements to the benchmark table."""
    mask = (
        (df["tokens_per_sec"] >= min_tokens_sec)
        & (df["hourly_cost_usd"] <= max_budget_hr)
        & (df["max_context_window"] >= required_ctx)
        & (df["model"].isin(models))
        & (df["quantization"].isin(quants))
    )
    return df[mask]


def pareto_frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Configs not dominated on (throughput up, $/1M tokens down)."""
    ordered = df.sort_values("tokens_per_sec", ascending=False)
    keep, best_cost = [], np.inf
    for idx, cost in zip(ordered.index, ordered["cost_per_1m_tokens_usd"]):
        if cost < best_cost:
            keep.append(idx)
            best_cost = cost
    return ordered.loc[keep].sort_values("tokens_per_sec")


# --------------------------------------------------------------------------- #
#  CHART LAYER
# --------------------------------------------------------------------------- #

def _apply_base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template=None,
        height=height,
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["panel"],
        font=dict(family=FONT, color=C["ink"], size=13),
        margin=dict(l=6, r=16, t=44, b=6),
        hoverlabel=dict(
            bgcolor="#1A2233",
            bordercolor=C["axis"],
            font=dict(family=FONT, color=C["ink"], size=12),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"], size=12),
        ),
    )
    axis_style = dict(
        gridcolor=C["grid"], zeroline=False, linecolor=C["axis"],
        tickfont=dict(color=C["muted"], size=11),
        title_font=dict(color=C["muted"], size=12),
        automargin=True, title_standoff=10,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    return fig


def _log_ticks(vmin: float, vmax: float, money: bool = False) -> tuple[list, list]:
    """1-2-5 tick series covering [vmin, vmax] for a log axis."""
    vals: list[float] = []
    decade = 10.0 ** math.floor(math.log10(max(vmin, 1e-9)))
    while decade <= vmax * 10:
        for mult in (1, 2, 5):
            v = decade * mult
            if vmin / 1.6 <= v <= vmax * 1.6:
                vals.append(v)
        decade *= 10
    if money:
        text = [f"${v:,.2f}" if v < 1 else f"${v:,.0f}" for v in vals]
    else:
        text = [f"{v / 1_000:g}k" if v >= 1_000 else f"{v:g}" for v in vals]
    return vals, text


def _marker_sizes(power_w: pd.Series, power_max: float) -> dict:
    """Marker size encodes power draw (area-true scaling)."""
    return dict(size=power_w, sizemode="area", sizeref=2.0 * power_max / (16.0**2), sizemin=6)


_HOVER = (
    "<b>%{customdata[8]}</b> · %{customdata[1]}<br>"
    "%{customdata[0]}<br>"
    "─────────────<br>"
    "Throughput: <b>%{x:,.0f} tok/s</b><br>"
    "Serving cost: <b>$%{y:,.2f} / 1M tok</b><br>"
    "Hourly: $%{customdata[2]:,.2f}/hr  ·  %{customdata[7]:.1f}M tok/$<br>"
    "Max context: %{customdata[3]:,.0f}K  ·  Power: %{customdata[4]:,.0f} W<br>"
    "VRAM: %{customdata[5]:,.0f} / %{customdata[6]:,.0f} GB"
    "<extra></extra>"
)


def _customdata(d: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            d["hardware_config"],
            d["quantization"],
            d["hourly_cost_usd"],
            d["max_context_window"] // 1024,
            d["power_draw_w"],
            d["vram_required_gb"],
            d["total_vram_gb"],
            d["tokens_per_dollar"] / 1e6,
            d["model"],
        ],
        axis=-1,
    )


def make_cost_scatter(
    df_all: pd.DataFrame, df_match: pd.DataFrame, models: list[str]
) -> go.Figure:
    """Cost per 1M tokens vs throughput. Non-matching configs stay as dim
    context; matching configs are colored by model family (exact model in
    the hover); the Pareto-efficient frontier is traced."""
    fig = go.Figure()
    power_max = float(df_all["power_draw_w"].max())

    # dim context layer: everything excluded by the current requirements
    df_out = df_all.loc[~df_all.index.isin(df_match.index)]
    if len(df_out):
        fig.add_trace(
            go.Scatter(
                x=df_out["tokens_per_sec"],
                y=df_out["cost_per_1m_tokens_usd"],
                mode="markers",
                name="Excluded by filters",
                marker=dict(
                    color=C["dim"], opacity=0.35,
                    symbol=[QUANT_SYMBOLS[q] for q in df_out["quantization"]],
                    **_marker_sizes(df_out["power_draw_w"], power_max),
                ),
                customdata=_customdata(df_out),
                hovertemplate=(
                    "<b>%{customdata[8]}</b> · %{customdata[0]} · "
                    "%{customdata[1]} (excluded)<br>"
                    "%{x:,.0f} tok/s · $%{y:,.2f}/1M<extra></extra>"
                ),
            )
        )

    # one trace per family, fixed validated color order
    for family in FAMILIES:
        d = df_match[df_match["family"] == family]
        if d.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=d["tokens_per_sec"],
                y=d["cost_per_1m_tokens_usd"],
                mode="markers",
                name=family,
                marker=dict(
                    color=FAMILY_COLORS[family], opacity=0.92,
                    symbol=[QUANT_SYMBOLS[q] for q in d["quantization"]],
                    line=dict(width=1, color="rgba(14,17,23,0.9)"),
                    **_marker_sizes(d["power_draw_w"], power_max),
                ),
                customdata=_customdata(d),
                hovertemplate=_HOVER,
            )
        )

    # Pareto-efficient frontier over the matching set
    frontier = pareto_frontier(df_match) if len(df_match) else df_match
    if len(frontier) >= 2:
        fig.add_trace(
            go.Scatter(
                x=frontier["tokens_per_sec"],
                y=frontier["cost_per_1m_tokens_usd"],
                mode="lines",
                name="Efficient frontier",
                line=dict(color="rgba(230,237,243,0.55)", width=1.6, dash="dot"),
                hoverinfo="skip",
            )
        )

    x_vals, x_text = _log_ticks(
        float(df_all["tokens_per_sec"].min()), float(df_all["tokens_per_sec"].max())
    )
    y_vals, y_text = _log_ticks(
        float(df_all["cost_per_1m_tokens_usd"].min()),
        float(df_all["cost_per_1m_tokens_usd"].max()),
        money=True,
    )
    fig.update_xaxes(
        type="log", title_text="Throughput — tokens / second (log)",
        tickvals=x_vals, ticktext=x_text,
    )
    fig.update_yaxes(
        type="log", title_text="Cost per 1M tokens (log)",
        tickvals=y_vals, ticktext=y_text,
    )
    return _apply_base_layout(fig, height=560)


def make_efficiency_bar(df_match: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Top clusters ranked by tokens-per-dollar for the current requirements."""
    d = df_match.nlargest(top_n, "tokens_per_dollar").iloc[::-1]
    labels = d["hardware_config"] + "   ·   " + d["model"] + " · " + d["quantization"]
    values = d["tokens_per_dollar"] / 1e6

    vmax, vmin = values.max(), values.min()
    norm = (values - vmin) / (vmax - vmin) if vmax > vmin else pd.Series(1.0, index=values.index)
    colors = sample_colorscale(
        [[0.0, C["blue"]], [0.55, C["cyan"]], [1.0, C["green"]]], list(norm)
    )

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[
                f"  ${c:,.2f}/1M · {t:,.0f} tok/s"
                for c, t in zip(d["cost_per_1m_tokens_usd"], d["tokens_per_sec"])
            ],
            textposition="outside",
            textfont=dict(color=C["muted"], size=11),
            cliponaxis=False,
            customdata=_customdata(d),
            meta="Efficiency leader",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "%{x:.2f}M tokens per dollar<br>"
                "Hourly: $%{customdata[2]:,.2f} · Power: %{customdata[4]:,.0f} W<br>"
                "Max context: %{customdata[3]:,.0f}K"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(bargap=0.32, showlegend=False)
    fig.update_xaxes(title_text="Million tokens per dollar", range=[0, float(vmax) * 1.30])
    fig.update_yaxes(title_text="", automargin=True)
    return _apply_base_layout(fig, height=130 + 36 * len(d))


def make_throughput_matrix(df_match: pd.DataFrame, models: list[str]):
    """Seaborn heatmap: best achievable tokens/sec per (model x GPU family)."""
    pivot = df_match.pivot_table(
        index="model", columns="gpu", values="tokens_per_sec", aggfunc="max"
    )
    pivot = pivot.reindex(
        index=[m for m in models if m in pivot.index],
        columns=[g for g in gpu_order() if g in pivot.columns],
    )
    if pivot.empty:
        return None

    cmap = LinearSegmentedColormap.from_list(
        "neon_depth", ["#131826", "#12395C", "#0E7490", C["cyan"], C["green"]]
    )
    fig, ax = plt.subplots(figsize=(11, 1.9 + 0.62 * len(pivot)), dpi=160)
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["panel"])

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=",.0f",
        linewidths=1.4,
        linecolor=C["bg"],
        annot_kws={"fontsize": 8.5, "fontweight": "bold"},
        cbar_kws={"label": "max tokens / sec"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(colors=C["muted"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=C["muted"], labelsize=8)
    cbar.ax.yaxis.label.set_color(C["muted"])
    cbar.outline.set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
#  UI LAYER
# --------------------------------------------------------------------------- #

def render_sidebar(df: pd.DataFrame) -> dict:
    """Workload-requirement controls. Returns the selected filter values."""
    st.sidebar.markdown("### ⚙️ Workload requirements")

    ts_max = int(math.ceil(df["tokens_per_sec"].max() / 500) * 500)
    min_tokens_sec = st.sidebar.slider(
        "Minimum Tokens/Second",
        min_value=0, max_value=ts_max, value=50, step=10,
        help="Aggregate serving throughput the deployment must sustain.",
    )

    budget_max = float(math.ceil(df["hourly_cost_usd"].max()))
    max_budget_hr = st.sidebar.slider(
        "Max Budget ($/hr)",
        min_value=0.25, max_value=budget_max, value=min(12.0, budget_max),
        step=0.25, format="$%.2f",
        help="Hard ceiling on hourly operating cost for the cluster.",
    )

    required_ctx = st.sidebar.select_slider(
        "Required Context Window",
        options=CTX_STEPS, value=8_192,
        format_func=lambda v: f"{v // 1024}K",
        help="Longest prompt+response the deployment must fit in KV-cache.",
    )

    models = model_order(df)
    quants = list(df["quantization"].unique())
    with st.sidebar.expander("Advanced filters"):
        models = st.multiselect("Models", model_order(df), default=models)
        quants = st.multiselect("Quantization", quants, default=quants)

    st.sidebar.divider()
    st.sidebar.caption(
        f"**{len(df)}** benchmarked configs · {df['model'].nunique()} models · "
        f"{df['hardware_config'].nunique()} clusters\n\n"
        "Synthetic benchmark data — seeded & reproducible (`generate_data.py`)."
    )
    return dict(
        min_tokens_sec=min_tokens_sec,
        max_budget_hr=max_budget_hr,
        required_ctx=required_ctx,
        models=models,
        quants=quants,
    )


def render_header() -> None:
    st.markdown(
        '<div class="hero-title">⚡ LLM Hardware & Cost '
        '<span class="accent">Optimization Engine</span></div>'
        '<div class="hero-sub">Right-size GPU clusters for your inference workload — '
        "throughput, context, power and $-per-token across consumer cards to enterprise racks.</div>",
        unsafe_allow_html=True,
    )


def render_kpis(df_match: pd.DataFrame, total: int) -> None:
    best_cost = df_match.loc[df_match["cost_per_1m_tokens_usd"].idxmin()]
    fastest = df_match.loc[df_match["tokens_per_sec"].idxmax()]
    frugal = df_match.loc[df_match["power_draw_w"].idxmin()]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Viable configs", f"{len(df_match)}", f"of {total} benchmarked", delta_color="off")
    k2.metric(
        "Best serving cost", f"${best_cost['cost_per_1m_tokens_usd']:,.2f}/1M",
        f"{best_cost['hardware_config']} · {best_cost['model']}", delta_color="off",
    )
    k3.metric(
        "Peak throughput (tok/s)", f"{fastest['tokens_per_sec']:,.0f}",
        f"{fastest['hardware_config']} · {fastest['model']}", delta_color="off",
    )
    k4.metric(
        "Lowest power draw", f"{frugal['power_draw_w']:,.0f} W",
        f"{frugal['hardware_config']} · {frugal['model']}", delta_color="off",
    )


def _section(title: str, sub: str, accent: str = "") -> None:
    st.markdown(
        f'<div class="section-title {accent}">{title}</div>'
        f'<div class="section-sub">{sub}</div>',
        unsafe_allow_html=True,
    )


TABLE_COLUMNS = {
    "model": st.column_config.TextColumn("Model"),
    "active_params_b": st.column_config.NumberColumn("Active (B)", format="%.1f"),
    "quantization": st.column_config.TextColumn("Quant"),
    "hardware_config": st.column_config.TextColumn("Cluster"),
    "hardware_tier": st.column_config.TextColumn("Tier"),
    "vram_required_gb": st.column_config.NumberColumn("VRAM req (GB)", format="%.0f"),
    "total_vram_gb": st.column_config.NumberColumn("VRAM total (GB)", format="%.0f"),
    "memory_bandwidth_gbs": st.column_config.NumberColumn("Mem BW (GB/s)", format="%.0f"),
    "max_context_window": st.column_config.NumberColumn("Max context", format="%dK"),
    "tokens_per_sec": st.column_config.NumberColumn("Tokens/s", format="%.0f"),
    "power_draw_w": st.column_config.NumberColumn("Power (W)", format="%.0f"),
    "hourly_cost_usd": st.column_config.NumberColumn("$/hr", format="$%.2f"),
    "cost_per_1m_tokens_usd": st.column_config.NumberColumn("$ / 1M tok", format="$%.2f"),
    "tokens_per_dollar": st.column_config.NumberColumn("M tok / $", format="%.1fM"),
}


@st.cache_data(show_spinner="Running research experiments (first load only, ~15 s)...")
def load_research() -> dict:
    return research.load_results()


def render_research() -> None:
    """The findings of the accompanying paper, as interactive charts."""
    res = load_research()
    s = res["summary"]

    _section(
        "🔬 Research findings",
        "Four numerical experiments over the cost model — the evidence behind the "
        "accompanying paper. E1–E3 run on the noise-free expectation of the model; "
        "E4 samples 400 Monte Carlo draws over structural-parameter priors.",
        accent="purple",
    )
    t1, t2, t3, t4, t5 = st.tabs([
        "E1 · Workload phase map", "E2 · Capacity saturation n*",
        "E3 · Quantization dividend", "E4 · Robust frontier", "E4b · Tier decidability",
    ])

    with t1:
        cs = s["e1"]["consumer_share_of_feasible_cells"]
        n_consumer = sum(1 for v in cs.values() if v >= 0.85)
        st.markdown(
            "**Claim — the cost-optimal tier is set by scale and *sparsity*, and every "
            f"winner is INT4.** Consumer clusters win {n_consumer} of {len(cs)} models' "
            "workload maps — every dense model ≤32B *plus* the sparse-active MoEs "
            "(gpt-oss-120B, Qwen3-30B-A3B) — while dense ≥70B and the MoE giants are "
            f"enterprise-won. {s['e1']['infeasible_cells']} of {s['e1']['total_cells']} "
            "cells are infeasible at any price in the catalog."
        )
        model = st.selectbox("Model", research.ALL_MODEL_NAMES, key="phase_model")
        st.plotly_chart(research.phase_map_chart(res["phase"], model),
                        width="stretch", config={"displayModeBar": False})

    with t2:
        n_curves = s["e2"]["nstar_curves_checked"]
        st.markdown(
            "**Claim — tokens-per-dollar is non-monotone in cluster size, peaking at a "
            "closed-form size n\\*.** Per-dollar bandwidth is flat in n and the tensor-"
            "parallel tax only decays it; the *only* per-dollar gain from adding GPUs is "
            "KV-cache headroom. The exact marginal condition predicts the empirical peak in "
            f"**{s['e2']['nstar_match_rate_exact']:.0%} of {n_curves} curves** "
            f"(first-order form: {s['e2']['nstar_match_rate_first_order']:.0%}). "
            "A second consumer GPU is worth ×2.04, not ×2."
        )
        st.plotly_chart(research.capacity_curves_chart(res["curves"]),
                        width="stretch", config={"displayModeBar": False})

    with t3:
        d4 = s["e3"]["int4_dividend"].values()
        st.markdown(
            "**Claim — INT4's cost dividend exceeds its 4× byte ratio at every scale "
            f"(×{min(d4):.1f}–×{max(d4):.1f}, peaking on MoE models), and collapses to "
            "the byte ratio exactly when the batch cap binds at both precisions** "
            "(INT8 at 70B-class = ×2.0 exactly). Quantization pays three times: fewer "
            "bytes per token, freed KV headroom, and admission to cheaper hardware. For "
            "DeepSeek-V3.2 and Kimi-K2, no FP16 configuration fits a single node at all — "
            "quantization is the *only* way onto this catalog."
        )
        st.plotly_chart(research.dividend_chart(res["dividend"]),
                        width="stretch", config={"displayModeBar": False})

    with t4:
        st.markdown(
            "**Claim — a small core of configurations is Pareto-efficient almost surely, "
            "regardless of calibration.** Procurement can shortlist these without trusting "
            "any single benchmark number."
        )
        model = st.selectbox("Model", research.ALL_MODEL_NAMES, key="robust_model")
        st.plotly_chart(research.robust_frontier_chart(res["member"], model),
                        width="stretch", config={"displayModeBar": False})

    with t5:
        cw = s["e4"]["consumer_win_rate"]
        st.markdown(
            "**Claim — the consumer-vs-enterprise decision is a phase transition in *dense* "
            "scale, with a contested band at 70B-class — and sparsity re-opens the consumer "
            "phase.** Consumer hardware is optimal in "
            f"{cw.get('Llama-3.1-8B', 0):.0%} of draws at 8B, "
            f"{cw.get('Llama-3.3-70B', 0):.0%} at dense 70B (inside calibration "
            f"uncertainty — point estimates cannot settle it), and "
            f"{cw.get('Llama-3.1-405B', 0):.0%} at dense 405B. Yet "
            f"**gpt-oss-120B — 117B total parameters — is consumer-optimal in "
            f"{cw.get('gpt-oss-120B', 0):.0%} of draws**: with 5.1B active parameters "
            "and a tiny KV cache, it behaves economically like a small model that "
            "happens to need 120B-class VRAM."
        )
        st.plotly_chart(research.decidability_chart(res["tier"]),
                        width="stretch", config={"displayModeBar": False})

    st.caption(
        "Full methodology, propositions and proofs in the accompanying paper "
        "(`thesis.md` / `thesis.pdf`). Reproduce with `python experiments.py`."
    )


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    df = load_data()
    models = model_order(df)
    filters = render_sidebar(df)
    df_match = filter_data(df, **filters)

    render_header()

    if df_match.empty:
        st.warning(
            "No hardware configuration meets these requirements. "
            "Relax the throughput floor, raise the budget, or shorten the context window."
        )
    else:
        render_kpis(df_match, total=len(df))

    # -- 1 · cost vs throughput ------------------------------------------- #
    _section(
        "Cost efficiency vs throughput",
        "Every viable deployment. Dotted line traces the Pareto-efficient frontier; "
        "dim marks fail the current requirements. ● FP16 · ◆ INT8 · ■ INT4 — marker size ∝ power draw.",
    )
    st.plotly_chart(
        make_cost_scatter(df, df_match, models),
        width="stretch",
        config={"displayModeBar": False},
    )

    # -- 2 · efficiency ranking ------------------------------------------- #
    if not df_match.empty:
        _section(
            "Most cost-efficient clusters for your workload",
            "Ranked by tokens generated per dollar under the current requirements.",
            accent="green",
        )
        st.plotly_chart(
            make_efficiency_bar(df_match),
            width="stretch",
            config={"displayModeBar": False},
        )

        # -- 3 · throughput matrix + data explorer ------------------------- #
        with st.expander("🔥 Throughput matrix — best tok/s by model × GPU family"):
            matrix_fig = make_throughput_matrix(df_match, models)
            if matrix_fig is not None:
                st.pyplot(matrix_fig, width="stretch")
                plt.close(matrix_fig)

        with st.expander("📋 Data explorer — all matching configurations"):
            table = df_match.sort_values("tokens_per_dollar", ascending=False)[
                list(TABLE_COLUMNS)
            ].assign(
                max_context_window=lambda d: d["max_context_window"] // 1024,
                tokens_per_dollar=lambda d: d["tokens_per_dollar"] / 1e6,
            )
            st.dataframe(
                table, column_config=TABLE_COLUMNS, hide_index=True,
                width="stretch", height=420,
            )
            st.download_button(
                "⬇ Download filtered results (CSV)",
                data=table.to_csv(index=False).encode(),
                file_name="llm_hardware_filtered.csv",
                mime="text/csv",
            )

    # -- 4 · research findings --------------------------------------------- #
    st.divider()
    render_research()

    st.divider()
    st.caption(
        "Synthetic benchmark dataset (seeded, reproducible). Throughput modeled as "
        "bandwidth-bound decode with a batched-serving uplift; costs reflect representative "
        "cloud GPU rental rates. Calibrate constants in `generate_data.py` to match your fleet."
    )


st.set_page_config(
    page_title="LLM Hardware & Cost Optimization Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
main()

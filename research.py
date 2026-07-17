"""
Interactive (Plotly) presentations of the research experiments for the
Streamlit dashboard. Loads precomputed results from `results/`; if they are
missing, runs the full experiment battery once (~15 s) and caches to disk.

Chart builders mirror `experiments.py` figures but add hover interactivity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from experiments import PANEL_MODELS, TS_FLOORS, run_all
from generate_data import CTX_STEPS, MODELS

ALL_MODEL_NAMES = [m["model"] for m in MODELS]

RESULTS_DIR = Path("results")

C = dict(
    bg="#0E1117", panel="#131826", grid="#1F2637", axis="#2A3550",
    ink="#E6EDF3", muted="#8B98AB", dim="#3A4356",
    cyan="#00CFF0", purple="#A78BFF", green="#2EE884",
    magenta="#FF4D9D", amber="#F5B84D", blue="#4D9FFF",
)
TIER_COLORS = {"Consumer": C["green"], "Workstation": C["amber"], "Enterprise": C["magenta"]}
FONT = "Inter, 'Segoe UI', system-ui, sans-serif"


def load_results() -> dict:
    """Load (or lazily compute) all experiment outputs."""
    summary_path = RESULTS_DIR / "experiments.json"
    if not summary_path.exists():
        run_all()
    return dict(
        summary=json.loads(summary_path.read_text()),
        phase=pd.read_csv(RESULTS_DIR / "e1_phase_map.csv"),
        curves=pd.read_csv(RESULTS_DIR / "e2_marginal_gpu.csv"),
        checks=pd.read_csv(RESULTS_DIR / "e2_nstar_checks.csv"),
        dividend=pd.read_csv(RESULTS_DIR / "e3_quant_dividend.csv"),
        member=pd.read_csv(RESULTS_DIR / "e4_frontier_membership.csv"),
        tier=pd.read_csv(RESULTS_DIR / "e4_tier_winrate.csv"),
    )


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        template=None, height=height, paper_bgcolor=C["bg"], plot_bgcolor=C["panel"],
        font=dict(family=FONT, color=C["ink"], size=13),
        margin=dict(l=6, r=16, t=30, b=6),
        hoverlabel=dict(bgcolor="#1A2233", bordercolor=C["axis"],
                        font=dict(family=FONT, color=C["ink"], size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=C["muted"], size=12)),
    )
    style = dict(gridcolor=C["grid"], zeroline=False, linecolor=C["axis"],
                 tickfont=dict(color=C["muted"], size=11),
                 title_font=dict(color=C["muted"], size=12),
                 automargin=True, title_standoff=10)
    fig.update_xaxes(**style)
    fig.update_yaxes(**style)
    return fig


# --------------------------------------------------------------------------- #
#  E1 — phase map (interactive, one model at a time)
# --------------------------------------------------------------------------- #

def phase_map_chart(phase: pd.DataFrame, model: str) -> go.Figure:
    sub = phase[phase["model"] == model]
    z, text, hover = [], [], []
    for ctx in CTX_STEPS:
        zr, tr, hr = [], [], []
        for ts in TS_FLOORS:
            r = sub[(sub["ctx"] == ctx) & (sub["ts_floor"] == ts)].iloc[0]
            if r["feasible"]:
                zr.append(np.log10(r["cost_1m"]))
                tr.append(r["winner"].replace("x ", "×").replace("RTX ", "")
                          .replace(" SXM", "").replace(" 80GB", ""))
                hr.append(f"<b>{r['winner']}</b> · {r['quant']} ({r['tier']})<br>"
                          f"${r['cost_1m']:,.2f} / 1M tokens")
            else:
                zr.append(np.nan)
                tr.append("—")
                hr.append("No configuration in catalog can meet this cell")
        z.append(zr)
        text.append(tr)
        hover.append(hr)

    feas = phase[phase["feasible"]]
    fig = go.Figure(
        go.Heatmap(
            z=z, text=text, customdata=hover,
            x=[f"{t/1000:g}k" if t >= 1000 else str(t) for t in TS_FLOORS],
            y=[f"{c // 1024}K" for c in CTX_STEPS],
            zmin=np.log10(feas["cost_1m"].min()), zmax=np.log10(feas["cost_1m"].max()),
            colorscale=[[0.0, C["green"]], [0.35, C["cyan"]],
                        [0.7, "#0E7490"], [1.0, "#123B5C"]],
            texttemplate="%{text}", textfont=dict(size=11, family=FONT),
            hovertemplate="%{customdata}<extra></extra>",
            xgap=2, ygap=2,
            colorbar=dict(
                title=dict(text="$ / 1M tok (log)", font=dict(color=C["muted"], size=11)),
                tickvals=[-1, 0, 1], ticktext=["$0.10", "$1", "$10"],
                tickfont=dict(color=C["muted"], size=10), outlinewidth=0,
            ),
        )
    )
    fig.update_xaxes(title_text="throughput floor (tokens / second)", type="category")
    fig.update_yaxes(title_text="required context window", type="category")
    return _layout(fig, 430)


# --------------------------------------------------------------------------- #
#  E2 — capacity saturation curves
# --------------------------------------------------------------------------- #

def capacity_curves_chart(curves: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    palette = [C["cyan"], C["amber"], C["magenta"], C["green"], C["blue"], C["purple"]]
    for color, ((gpu, model, quant), sub) in zip(
        palette, curves.groupby(["gpu", "model", "quant"], sort=False)
    ):
        sub = sub.sort_values("num_gpus")
        fig.add_trace(go.Scatter(
            x=sub["num_gpus"], y=sub["multiple_vs_min"], mode="lines+markers",
            name=f"{gpu} · {model.split('-')[-1]} {quant}",
            line=dict(color=color, width=2.2), marker=dict(size=7),
            customdata=np.stack([sub["tokens_per_dollar"] / 1e6], axis=-1),
            hovertemplate=(f"<b>{gpu}</b> · {model} {quant}<br>"
                           "n = %{x} GPUs → ×%{y:.2f} vs smallest viable"
                           "<br>%{customdata[0]:.2f}M tokens / $<extra></extra>"),
        ))
        pred = sub["n_star_pred"].iloc[0]
        star = sub[sub["num_gpus"] == pred]
        if len(star):
            fig.add_trace(go.Scatter(
                x=star["num_gpus"], y=star["multiple_vs_min"], mode="markers",
                marker=dict(symbol="star", size=17, color=color,
                            line=dict(color="#0B0E14", width=1)),
                showlegend=False,
                hovertemplate=f"n* = {pred} (closed-form prediction)<extra></extra>",
            ))
    fig.add_hline(y=1.0, line=dict(color=C["dim"], dash="dash", width=1))
    fig.update_xaxes(type="log", tickvals=[1, 2, 4, 8],
                     title_text="GPUs in cluster (log₂)")
    fig.update_yaxes(title_text="tokens-per-$, multiple of smallest viable cluster")
    return _layout(fig, 440)


# --------------------------------------------------------------------------- #
#  E3 — quantization dividend
# --------------------------------------------------------------------------- #

def dividend_chart(div: pd.DataFrame) -> go.Figure:
    models = [m["model"] for m in MODELS]
    qcolors = {"FP16": C["dim"], "INT8": C["blue"], "INT4": C["cyan"]}
    fig = go.Figure()
    for quant in ["FP16", "INT8", "INT4"]:
        sub = div[div["quant"] == quant].set_index("model").reindex(models)
        fig.add_trace(go.Bar(
            x=models, y=sub["best_tokens_per_dollar"] / 1e6, name=quant,
            marker=dict(color=qcolors[quant],
                        line=dict(color=[TIER_COLORS.get(t, C["dim"])
                                         for t in sub["best_tier"]], width=2)),
            customdata=np.stack([sub["best_config"], sub["best_tier"],
                                 sub["dividend_vs_fp16"]], axis=-1),
            hovertemplate=("<b>%{x}</b> · " + quant +
                           "<br>%{y:.2f}M tokens / $<br>best: %{customdata[0]} "
                           "(%{customdata[1]})<br>×%{customdata[2]:.2f} vs FP16"
                           "<extra></extra>"),
        ))
    int4 = div[div["quant"] == "INT4"].set_index("model").reindex(models)
    for xi, m in enumerate(models):
        if m not in int4.index or not np.isfinite(int4.loc[m, "best_tokens_per_dollar"]):
            continue
        label = (f"×{int4.loc[m, 'dividend_vs_fp16']:.1f}"
                 if np.isfinite(int4.loc[m, "dividend_vs_fp16"]) else "—")
        fig.add_annotation(
            x=xi + 0.27, y=np.log10(int4.loc[m, "best_tokens_per_dollar"] / 1e6) + 0.18,
            xref="x", text=label,
            showarrow=False, font=dict(color=C["green"], size=11, family=FONT),
        )
    fig.update_xaxes(tickangle=-38, tickfont=dict(size=10))
    fig.update_yaxes(type="log", title_text="best achievable M tokens per $ (log)")
    fig.update_layout(barmode="group", bargap=0.3)
    return _layout(fig, 480)


# --------------------------------------------------------------------------- #
#  E4 — robust frontier + tier decidability
# --------------------------------------------------------------------------- #

def robust_frontier_chart(member: pd.DataFrame, model: str) -> go.Figure:
    sub = member[member["model"] == model].head(8).iloc[::-1]
    labels = sub["config"] + " · " + sub["quant"]
    fig = go.Figure(go.Bar(
        x=sub["membership_prob"], y=labels, orientation="h",
        marker=dict(color=[TIER_COLORS[t] for t in sub["tier"]]),
        text=[f"{v:.0%}" for v in sub["membership_prob"]],
        textposition="outside", textfont=dict(color=C["muted"], size=11),
        cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>on the Pareto frontier in %{x:.0%} of draws<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 1.15], tickformat=".0%",
                     title_text="Pareto-frontier membership probability (400 Monte Carlo draws)")
    fig.update_layout(bargap=0.35, showlegend=False)
    return _layout(fig, 90 + 42 * len(sub))


def decidability_chart(tier_df: pd.DataFrame) -> go.Figure:
    models = [m["model"] for m in MODELS][::-1]      # smallest at top
    moe_names = {m["model"] for m in MODELS if m["active_b"] < m["params_b"]}
    fig = go.Figure()
    for tier in ("Consumer", "Workstation", "Enterprise"):
        vals = [tier_df[(tier_df["model"] == m) & (tier_df["tier"] == tier)]["win_rate"].iloc[0]
                for m in models]
        fig.add_trace(go.Bar(
            y=models, x=vals, orientation="h", name=tier,
            marker=dict(color=TIER_COLORS[tier]),
            text=[f"{v:.0%}" if v > 0.08 else "" for v in vals],
            textfont=dict(color="#0B0E14", size=11, family=FONT),
            insidetextanchor="middle",
            hovertemplate=f"<b>{tier}</b> optimal in " + "%{x:.1%} of draws<extra></extra>",
        ))
    fig.add_vline(x=0.5, line=dict(color=C["ink"], dash="dot", width=1), opacity=0.5)
    fig.update_layout(barmode="stack", bargap=0.34)
    fig.update_yaxes(
        tickfont=dict(size=11),
        ticktext=[f"<span style='color:{C['cyan']}'>{m}</span>" if m in moe_names else m
                  for m in models],
        tickvals=models,
    )
    fig.update_xaxes(tickformat=".0%", title_text="P(tier holds the tokens-per-$ optimum) — MoE models in cyan")
    return _layout(fig, 620)

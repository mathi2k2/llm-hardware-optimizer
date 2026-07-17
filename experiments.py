"""
Research experiments for the LLM Hardware & Cost Optimization Engine.

Four numerical experiments over the analytical cost model, supporting the
paper "Phase Transitions in the Economics of LLM Inference". The catalog
covers 19 major open-weight models (dense 3B–405B and MoE to 1T total):

  E1  Workload phase map      — cost-optimal deployment per (throughput
                                floor, context requirement), every model.
  E2  Capacity saturation     — tokens-per-dollar vs cluster size; validates
                                the closed-form optimum n* (Prop. 1/1') on
                                every configuration curve in the catalog.
  E3  Quantization dividend   — best tokens-per-dollar by precision;
                                super-/sub-proportional compounding vs the
                                raw byte ratio, dense and MoE.
  E4  Frontier robustness     — Monte Carlo over structural priors;
                                Pareto membership + tier decidability.

E1–E3 run on the *expectation* of the model (noise zeroed) so closed-form
predictions can be checked exactly. E4 samples structural priors, including
the MoE batching exponent.

Run:  python experiments.py          (writes results/ and figures/)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_data import (
    CTX_STEPS,
    GPU_SPECS,
    MODELS,
    QUANT_BYTES,
    SimParams,
    deterministic_params,
    generate_dataset,
    n_star,
    n_star_exact,
)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

ALL_MODEL_NAMES = [m["model"] for m in MODELS]

#: 3x3 grid used for the paper's panel figures — spans dense small -> frontier
#: plus the sparse-active (MoE) frontier. The app exposes every model.
PANEL_MODELS = [
    "Llama-3.1-8B", "Qwen3-32B", "Llama-3.3-70B",
    "gpt-oss-120B", "Qwen3-235B-A22B", "Llama-4-Maverick-400B",
    "Llama-3.1-405B", "DeepSeek-V3.2-685B", "Kimi-K2-1T",
]

TS_FLOORS = [10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000]
REFERENCE_WORKLOAD = dict(min_ctx=8_192)

# Palette (kept in sync with app.py; validated for CVD separation on #0E1117)
C = dict(
    bg="#0E1117", panel="#131826", grid="#1F2637", ink="#E6EDF3", muted="#8B98AB",
    cyan="#00CFF0", purple="#A78BFF", green="#2EE884", magenta="#FF4D9D",
    amber="#F5B84D", blue="#4D9FFF", orange="#FF8A5C", dim="#3A4356",
)
TIER_COLORS = {"Consumer": C["green"], "Workstation": C["amber"], "Enterprise": C["magenta"]}


# --------------------------------------------------------------------------- #
#  E1 — workload phase map (all models)
# --------------------------------------------------------------------------- #

def run_phase_map(df_det: pd.DataFrame) -> pd.DataFrame:
    cells = []
    for model in ALL_MODEL_NAMES:
        sub = df_det[df_det["model"] == model]
        for ctx in CTX_STEPS:
            for ts in TS_FLOORS:
                ok = sub[(sub["tokens_per_sec"] >= ts) & (sub["max_context_window"] >= ctx)]
                if ok.empty:
                    cells.append(dict(model=model, ts_floor=ts, ctx=ctx, feasible=False,
                                      winner=None, tier=None, quant=None, cost_1m=np.nan))
                    continue
                w = ok.loc[ok["cost_per_1m_tokens_usd"].idxmin()]
                cells.append(dict(
                    model=model, ts_floor=ts, ctx=ctx, feasible=True,
                    winner=w["hardware_config"], tier=w["hardware_tier"],
                    quant=w["quantization"], cost_1m=w["cost_per_1m_tokens_usd"],
                ))
    return pd.DataFrame(cells)


# --------------------------------------------------------------------------- #
#  E2 — capacity saturation & the closed-form optimum n*
# --------------------------------------------------------------------------- #

E2_CASES = [
    # (model, quant, gpu) — one per distinct VRAM size + one MoE case.
    # Corollary 1: normalized curves depend only on (V, W, kappa, alpha) —
    # bandwidth and price cancel, so equal-VRAM GPUs share identical curves.
    ("Llama-3.1-8B",   "INT8", "RTX 4090"),   # 24 GB  dense
    ("Qwen3-14B",      "INT8", "L40S"),       # 48 GB  dense
    ("Llama-3.3-70B",  "INT4", "H100 SXM"),   # 80 GB  dense
    ("Llama-3.1-405B", "INT4", "H200 SXM"),   # 141 GB dense frontier
    ("gpt-oss-120B",   "INT4", "H200 SXM"),   # 141 GB sparse-active (MoE)
]


def run_marginal_gpu(df_det: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for model, quant, gpu in E2_CASES:
        sub = df_det[
            (df_det["gpu"] == gpu) & (df_det["model"] == model)
            & (df_det["quantization"] == quant)
        ].sort_values("num_gpus")
        if sub.empty:
            continue
        base = sub.iloc[0]["tokens_per_dollar"]
        pred = n_star_exact(model, quant, gpu, sizes=sorted(sub["num_gpus"].unique()))
        for _, r in sub.iterrows():
            rows.append(dict(
                gpu=gpu, model=model, quant=quant, num_gpus=int(r["num_gpus"]),
                tokens_per_dollar=r["tokens_per_dollar"],
                multiple_vs_min=r["tokens_per_dollar"] / base,
                n_star_pred=pred,
            ))
    curves = pd.DataFrame(rows)

    # n* validation over every (gpu, model, quant) curve with >= 2 sizes
    checks = []
    for (gpu, model, quant), sub in df_det.groupby(["gpu", "model", "quantization"]):
        if sub["num_gpus"].nunique() < 2:
            continue
        emp = int(sub.loc[sub["tokens_per_dollar"].idxmax(), "num_gpus"])
        sizes = sorted(sub["num_gpus"].unique())
        pred_fo = n_star(model, quant, gpu)
        pred_fo_clamped = min([s for s in sizes if s >= pred_fo] or [sizes[-1]])
        pred_exact = n_star_exact(model, quant, gpu, sizes=sizes)
        checks.append(dict(
            gpu=gpu, model=model, quant=quant,
            n_star_first_order=pred_fo_clamped, n_star_exact=pred_exact,
            n_star_empirical=emp,
            match_first_order=pred_fo_clamped == emp,
            match_exact=pred_exact == emp,
        ))
    return curves, pd.DataFrame(checks)


# --------------------------------------------------------------------------- #
#  E3 — quantization dividend (all models)
# --------------------------------------------------------------------------- #

def run_quant_dividend(df_det: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, sub in df_det.groupby("model", sort=False):
        fp16 = sub[sub["quantization"] == "FP16"]
        fp16_best = fp16["tokens_per_dollar"].max() if len(fp16) else np.nan
        for quant in ["FP16", "INT8", "INT4"]:
            q = sub[sub["quantization"] == quant]
            if q.empty:
                continue
            w = q.loc[q["tokens_per_dollar"].idxmax()]
            rows.append(dict(
                model=model, params_b=w["params_b"], is_moe=bool(w["is_moe"]),
                quant=quant,
                best_tokens_per_dollar=w["tokens_per_dollar"],
                best_config=w["hardware_config"], best_tier=w["hardware_tier"],
                dividend_vs_fp16=(w["tokens_per_dollar"] / fp16_best)
                if np.isfinite(fp16_best) and fp16_best > 0 else np.nan,
                byte_ratio=QUANT_BYTES["FP16"] / QUANT_BYTES[quant],
            ))
    return pd.DataFrame(rows).sort_values(["params_b", "model", "quant"])


# --------------------------------------------------------------------------- #
#  E4 — Monte Carlo frontier robustness + tier decidability
# --------------------------------------------------------------------------- #

def _pareto_ids(df: pd.DataFrame) -> set[tuple]:
    ordered = df.sort_values("tokens_per_sec", ascending=False)
    keep, best = set(), np.inf
    for key, cost in zip(
        zip(ordered["model"], ordered["quantization"], ordered["hardware_config"]),
        ordered["cost_per_1m_tokens_usd"],
    ):
        if cost < best:
            keep.add(key)
            best = cost
    return keep


def run_frontier_robustness(k: int = 400, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per draw and per model: (a) within-model Pareto membership (ctx >= 8K);
    (b) which tier holds the tokens-per-dollar optimum (decidability)."""
    rng = np.random.default_rng(seed)
    counts: dict[tuple, int] = {}
    tier_wins: dict[str, dict[str, int]] = {}

    for _ in range(k):
        p = SimParams(
            mbu_mean=float(rng.uniform(0.45, 0.65)),
            batch_exponent=float(rng.uniform(0.72, 0.90)),
            batch_exponent_moe=float(rng.uniform(0.45, 0.70)),
            tp_gamma=float(rng.uniform(0.78, 0.90)),
            price_mult={
                t: float(np.exp(rng.normal(0.0, 0.15)))
                for t in ("Consumer", "Workstation", "Enterprise")
            },
        )
        d = generate_dataset(path=None, seed=int(rng.integers(1, 2**31)), params=p)
        d = d[d["max_context_window"] >= REFERENCE_WORKLOAD["min_ctx"]]
        for model, sub in d.groupby("model"):
            for key in _pareto_ids(sub):
                counts[key] = counts.get(key, 0) + 1
            best_tier = sub.loc[sub["tokens_per_dollar"].idxmax(), "hardware_tier"]
            tier_wins.setdefault(model, {}).setdefault(best_tier, 0)
            tier_wins[model][best_tier] += 1

    member = pd.DataFrame(
        [
            dict(model=m, quant=q, config=c, membership_prob=v / k,
                 tier=GPU_SPECS[c.split("x ", 1)[1]]["tier"])
            for (m, q, c), v in counts.items()
        ]
    ).sort_values("membership_prob", ascending=False)

    params_by_model = {m["model"]: m["params_b"] for m in MODELS}
    tier_rows = []
    for model, wins in tier_wins.items():
        for tier in ("Consumer", "Workstation", "Enterprise"):
            tier_rows.append(dict(model=model, params_b=params_by_model[model],
                                  tier=tier, win_rate=wins.get(tier, 0) / k))
    tier_df = pd.DataFrame(tier_rows).sort_values(["params_b", "model", "tier"])
    return member, tier_df


# --------------------------------------------------------------------------- #
#  Figures (matplotlib, dark; palette validated for the #0E1117 surface)
# --------------------------------------------------------------------------- #

def _dark_axes(ax):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["muted"], labelsize=9)
    for s in ax.spines.values():
        s.set_color(C["grid"])
    ax.xaxis.label.set_color(C["muted"])
    ax.yaxis.label.set_color(C["muted"])
    ax.title.set_color(C["ink"])
    ax.grid(color=C["grid"], linewidth=0.6, alpha=0.6)


def _short(config: str) -> str:
    return (config.replace("x ", "×").replace("RTX ", "")
            .replace(" SXM", "").replace(" 80GB", ""))


def fig_phase_map(phase: pd.DataFrame, path: Path):
    """3x3 grid: cost surface over workload space for nine model scales."""
    feas = phase[phase["model"].isin(PANEL_MODELS) & phase["feasible"]]
    norm = matplotlib.colors.LogNorm(vmin=feas["cost_1m"].min(),
                                     vmax=feas["cost_1m"].max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "neon_depth_r", [C["green"], C["cyan"], "#0E7490", "#123B5C"]
    )
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 12.6), dpi=170)
    fig.patch.set_facecolor(C["bg"])
    for ax, model in zip(axes.flat, PANEL_MODELS):
        sub = phase[phase["model"] == model]
        tier = sub[sub["feasible"]]["tier"].iloc[0] if sub["feasible"].any() else "—"
        grid = np.full((len(CTX_STEPS), len(TS_FLOORS)), np.nan)
        for _, r in sub.iterrows():
            yi, xi = CTX_STEPS.index(r["ctx"]), TS_FLOORS.index(r["ts_floor"])
            grid[yi, xi] = r["cost_1m"] if r["feasible"] else np.nan
        ax.set_facecolor("#0B0E14")
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        for _, r in sub.iterrows():
            yi, xi = CTX_STEPS.index(r["ctx"]), TS_FLOORS.index(r["ts_floor"])
            if r["feasible"]:
                lum = norm(r["cost_1m"])
                ax.text(xi, yi, _short(r["winner"]), ha="center", va="center",
                        fontsize=5.6, fontweight="bold",
                        color="#0B0E14" if lum < 0.55 else C["ink"])
            else:
                ax.text(xi, yi, "—", ha="center", va="center",
                        fontsize=6, color=C["dim"])
        ax.set_xticks(range(len(TS_FLOORS)),
                      [f"{t/1000:g}k" if t >= 1000 else str(t) for t in TS_FLOORS])
        ax.set_yticks(range(len(CTX_STEPS)), [f"{c // 1024}K" for c in CTX_STEPS])
        ax.tick_params(colors=C["muted"], labelsize=7)
        for s in ax.spines.values():
            s.set_color(C["grid"])
        moe = " (MoE)" if any(m["model"] == model and m["active_b"] < m["params_b"]
                              for m in MODELS) else ""
        ax.set_title(f"{model}{moe}", color=C["ink"], fontsize=10, fontweight="bold",
                     loc="left")
        ax.text(0.985, 0.955, f"◼ {tier}", transform=ax.transAxes, ha="right",
                va="top", color=TIER_COLORS.get(tier, C["dim"]), fontsize=8,
                fontweight="bold",
                bbox=dict(facecolor="#0B0E14", edgecolor="none", pad=2.5))
    for ax in axes[-1]:
        ax.set_xlabel("throughput floor (tok/s)", color=C["muted"], fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("required context", color=C["muted"], fontsize=8)
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.012)
    cbar.set_label("optimal cost — $ per 1M tokens (log, shared scale)",
                   color=C["muted"], fontsize=9)
    cbar.ax.tick_params(colors=C["muted"], labelsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle(
        "E1 — Cost-optimal deployment across workload space, nine model scales "
        "(all winners INT4; — = infeasible in catalog)",
        color=C["ink"], fontsize=13.5, fontweight="bold", x=0.02, ha="left",
    )
    fig.savefig(path, facecolor=C["bg"], bbox_inches="tight")
    plt.close(fig)


def fig_marginal_gpu(curves: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9.8, 5.4), dpi=180)
    fig.patch.set_facecolor(C["bg"])
    _dark_axes(ax)
    colors = [C["cyan"], C["amber"], C["magenta"], C["green"], C["orange"]]
    labels = {
        ("Llama-3.1-8B", "RTX 4090"): "24 GB — RTX 3090/4090 · 8B INT8",
        ("Qwen3-14B", "L40S"): "48 GB — L40S · Qwen3-14B INT8",
        ("Llama-3.3-70B", "H100 SXM"): "80 GB — A100/H100 · 70B INT4",
        ("Llama-3.1-405B", "H200 SXM"): "141 GB — H200 · 405B INT4",
        ("gpt-oss-120B", "H200 SXM"): "141 GB — H200 · gpt-oss-120B INT4 (MoE)",
    }
    for color, (model, quant, gpu) in zip(colors, E2_CASES):
        sub = curves[(curves["gpu"] == gpu) & (curves["model"] == model)
                     & (curves["quant"] == quant)]
        if sub.empty:
            continue
        ls = "--" if "oss" in model else "-"
        ax.plot(sub["num_gpus"], sub["multiple_vs_min"], marker="o", ms=5,
                lw=2, ls=ls, color=color, label=labels[(model, gpu)])
        pred = sub["n_star_pred"].iloc[0]
        star = sub[sub["num_gpus"] == pred]
        if len(star):
            ax.plot(star["num_gpus"], star["multiple_vs_min"], marker="*",
                    ms=17, color=color, mec="#0B0E14", mew=0.8, zorder=5)
    ax.axhline(1.0, color=C["dim"], lw=1, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xlabel("GPUs in cluster (log₂)")
    ax.set_ylabel("tokens-per-$, multiple of smallest viable cluster")
    ax.set_title("E2 — Cost-efficiency peaks at the capacity-saturation size n*  (★ = closed-form n*)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.text(0.995, 0.20,
            "normalized curves depend only on (VRAM, weights, KV/request, α) —\n"
            "bandwidth & price cancel, so equal-VRAM GPUs share one curve",
            transform=ax.transAxes, ha="right", color=C["muted"], fontsize=8, style="italic")
    ax.legend(frameon=False, labelcolor=C["muted"], fontsize=8.2)
    fig.tight_layout()
    fig.savefig(path, facecolor=C["bg"], bbox_inches="tight")
    plt.close(fig)


def fig_quant_dividend(div: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(15, 5.6), dpi=170)
    fig.patch.set_facecolor(C["bg"])
    _dark_axes(ax)
    models = [m["model"] for m in MODELS]
    quants = ["FP16", "INT8", "INT4"]
    qcolors = {"FP16": C["dim"], "INT8": C["blue"], "INT4": C["cyan"]}
    width = 0.27
    x = np.arange(len(models))
    ymax = 0.0
    for qi, quant in enumerate(quants):
        vals, edges = [], []
        for m in models:
            row = div[(div["model"] == m) & (div["quant"] == quant)]
            v = row["best_tokens_per_dollar"].iloc[0] / 1e6 if len(row) else np.nan
            vals.append(v)
            ymax = max(ymax, v if np.isfinite(v) else 0)
            edges.append(TIER_COLORS.get(row["best_tier"].iloc[0], C["dim"])
                         if len(row) else C["dim"])
        ax.bar(x + (qi - 1) * width, vals, width * 0.9, color=qcolors[quant],
               label=quant, edgecolor=edges, linewidth=1.4)
    for xi, m in enumerate(models):
        row = div[(div["model"] == m) & (div["quant"] == "INT4")]
        if len(row) and np.isfinite(row["dividend_vs_fp16"].iloc[0]):
            ax.text(xi + width, row["best_tokens_per_dollar"].iloc[0] / 1e6 * 1.3,
                    f"×{row['dividend_vs_fp16'].iloc[0]:.1f}",
                    ha="center", color=C["green"], fontsize=8.6, fontweight="bold")
        elif len(row):
            ax.text(xi + width, row["best_tokens_per_dollar"].iloc[0] / 1e6 * 1.3,
                    "no FP16\nfits", ha="center", color=C["muted"], fontsize=6.5)
    ax.set_yscale("log")
    labels = [m.replace("Mistral-Small-3.1", "Mistral-Small").replace("-Maverick", "-Mav")
              for m in models]
    ax.set_xticks(x, labels, fontsize=7.2, rotation=38, ha="right")
    ax.set_ylabel("best achievable M tokens per $ (log)")
    ax.set_title("E3 — The quantization dividend vs FP16, all 19 models "
                 "(bar edge = winning hardware tier)",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, labelcolor=C["muted"], fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=C["bg"], bbox_inches="tight")
    plt.close(fig)


def fig_frontier_robustness(member: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 11.6), dpi=170)
    fig.patch.set_facecolor(C["bg"])
    for ax, model in zip(axes.flat, PANEL_MODELS):
        sub = member[member["model"] == model].head(5).iloc[::-1]
        _dark_axes(ax)
        labels = [_short(c) + " · " + q for c, q in zip(sub["config"], sub["quant"])]
        ax.barh(labels, sub["membership_prob"],
                color=[TIER_COLORS[t] for t in sub["tier"]], height=0.58)
        for y, v in enumerate(sub["membership_prob"]):
            ax.text(min(v + 0.02, 1.02), y, f"{v:.0%}", va="center",
                    color=C["muted"], fontsize=8)
        ax.set_xlim(0, 1.2)
        ax.set_title(model, color=C["ink"], fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("frontier membership probability", color=C["muted"], fontsize=8)
    fig.suptitle("E4 — Robust efficient frontiers under calibration uncertainty "
                 "(400 Monte Carlo draws over structural priors)",
                 color=C["ink"], fontsize=13.5, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, facecolor=C["bg"], bbox_inches="tight")
    plt.close(fig)


def fig_tier_winrate(tier_df: pd.DataFrame, path: Path):
    """P(tier holds the tokens-per-$ optimum) — all models, small -> large."""
    order = [m["model"] for m in MODELS]
    fig, ax = plt.subplots(figsize=(11, 7.6), dpi=170)
    fig.patch.set_facecolor(C["bg"])
    _dark_axes(ax)
    left = np.zeros(len(order))
    for tier in ("Consumer", "Workstation", "Enterprise"):
        vals = np.array([
            tier_df[(tier_df["model"] == m) & (tier_df["tier"] == tier)]["win_rate"].iloc[0]
            for m in order
        ])
        ax.barh(order, vals, left=left, color=TIER_COLORS[tier], label=tier, height=0.6)
        for yi, (b, v) in enumerate(zip(left, vals)):
            if v > 0.09:
                ax.text(b + v / 2, yi, f"{v:.0%}", ha="center", va="center",
                        color="#0B0E14", fontsize=8.2, fontweight="bold")
        left += vals
    ax.axvline(0.5, color=C["ink"], lw=1, ls=":", alpha=0.5)
    ax.invert_yaxis()
    moe_names = {m["model"] for m in MODELS if m["active_b"] < m["params_b"]}
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(8.2)
        if lbl.get_text() in moe_names:
            lbl.set_color(C["cyan"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("P(tier holds the tokens-per-$ optimum)")
    ax.set_title("E4b — Tier decidability across all 19 models (MoE names in cyan): "
                 "sparsity re-opens the consumer phase",
                 fontsize=11.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, labelcolor=C["muted"], fontsize=9, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    fig.savefig(path, facecolor=C["bg"], bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #

def run_all(k_mc: int = 400) -> dict:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    df_det = generate_dataset(path=None, params=deterministic_params())

    phase = run_phase_map(df_det)
    curves, checks = run_marginal_gpu(df_det)
    div = run_quant_dividend(df_det)
    member, tier_df = run_frontier_robustness(k=k_mc)

    phase.to_csv(RESULTS_DIR / "e1_phase_map.csv", index=False)
    curves.to_csv(RESULTS_DIR / "e2_marginal_gpu.csv", index=False)
    checks.to_csv(RESULTS_DIR / "e2_nstar_checks.csv", index=False)
    div.to_csv(RESULTS_DIR / "e3_quant_dividend.csv", index=False)
    member.to_csv(RESULTS_DIR / "e4_frontier_membership.csv", index=False)
    tier_df.to_csv(RESULTS_DIR / "e4_tier_winrate.csv", index=False)

    fig_phase_map(phase, FIGURES_DIR / "fig1_phase_map.png")
    fig_marginal_gpu(curves, FIGURES_DIR / "fig2_capacity_saturation.png")
    fig_quant_dividend(div, FIGURES_DIR / "fig3_quantization_dividend.png")
    fig_frontier_robustness(member, FIGURES_DIR / "fig4_robust_frontier.png")
    fig_tier_winrate(tier_df, FIGURES_DIR / "fig5_tier_decidability.png")

    # ---- headline numbers for the paper ---------------------------------- #
    feas = phase[phase["feasible"]]
    consumer_share = {
        m: float((feas[feas["model"] == m]["tier"] == "Consumer").mean())
        if (feas["model"] == m).any() else 0.0
        for m in ALL_MODEL_NAMES
    }

    second_gpu = {}
    for model, quant, gpu in E2_CASES:
        sub = curves[(curves["gpu"] == gpu) & (curves["model"] == model)
                     & (curves["quant"] == quant)].sort_values("num_gpus")
        if len(sub) >= 2:
            second_gpu[f"{gpu} · {model} {quant}"] = round(
                float(sub.iloc[1]["tokens_per_dollar"] / sub.iloc[0]["tokens_per_dollar"]), 2
            )

    int4 = div[div["quant"] == "INT4"].set_index("model")
    summary = dict(
        dataset=dict(configs=int(len(df_det)), models=len(MODELS),
                     families=int(df_det["family"].nunique()),
                     moe_models=int(df_det[df_det["is_moe"]]["model"].nunique()),
                     clusters=int(df_det["hardware_config"].nunique())),
        e1=dict(
            consumer_share_of_feasible_cells=consumer_share,
            infeasible_cells=int((~phase["feasible"]).sum()),
            total_cells=int(len(phase)),
        ),
        e2=dict(
            nstar_curves_checked=int(len(checks)),
            nstar_match_rate_first_order=float(checks["match_first_order"].mean()),
            nstar_match_rate_exact=float(checks["match_exact"].mean()),
            second_gpu_multiple=second_gpu,
        ),
        e3=dict(
            int4_dividend={m: round(float(int4.loc[m, "dividend_vs_fp16"]), 2)
                           for m in int4.index if np.isfinite(int4.loc[m, "dividend_vs_fp16"])},
            int4_best_tier={m: int4.loc[m, "best_tier"] for m in int4.index},
            int8_dividend={
                r["model"]: round(float(r["dividend_vs_fp16"]), 2)
                for _, r in div[div["quant"] == "INT8"].iterrows()
                if np.isfinite(r["dividend_vs_fp16"])
            },
        ),
        e4=dict(
            mc_draws=k_mc,
            robust_core={
                model: [
                    dict(config=r["config"], quant=r["quant"],
                         prob=round(float(r["membership_prob"]), 3), tier=r["tier"])
                    for _, r in member[member["model"] == model].head(4).iterrows()
                ]
                for model in PANEL_MODELS
            },
            consumer_win_rate={
                r["model"]: round(float(r["win_rate"]), 3)
                for _, r in tier_df[tier_df["tier"] == "Consumer"].iterrows()
            },
        ),
    )
    (RESULTS_DIR / "experiments.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    s = run_all()
    print(json.dumps(s, indent=2))

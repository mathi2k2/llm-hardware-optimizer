"""
LLM Hardware & Cost Optimization Engine - synthetic benchmark generator.

Produces `llm_hardware_metrics.csv`: every *viable* (model x quantization x
GPU cluster) deployment combination, with physically-grounded estimates for
VRAM requirements, memory bandwidth, throughput, power draw and cost.

The catalog covers the major open-weight models of mid-2026 — dense models
from 3B to 405B and Mixture-of-Experts (MoE) models to 1T total parameters.

Methodology (simplified, but internally consistent and reproducible):

* LLM decode is memory-bandwidth-bound. Single-stream throughput is
  approximated as: effective aggregate bandwidth x MBU / bytes streamed per
  token. For dense models that is all weights; for MoE models only the
  ACTIVE parameters stream per token, while ALL experts must sit in VRAM.
* Multi-GPU tensor parallelism pays a communication tax (gamma per
  doubling of GPU count; default 0.85).
* VRAM left after weights becomes KV-cache. It buys (a) a batched-serving
  throughput uplift with diminishing returns (batch ** alpha) and (b) the
  maximum supported context window. KV footprint per token is set by the
  attention stack, not parameter count — MoE models carry explicit
  per-architecture values (MLA models like DeepSeek/Kimi cache ~10x less).
* MoE batching uses a *lower* exponent (alpha_moe = 0.55 vs 0.82 dense):
  larger batches activate a wider union of experts, diluting weight reuse.
* Combinations that do not fit in usable VRAM - or that would idle more
  than ~94% of it (waste ratio > 16x) - are excluded as non-viable.
* Costs reflect representative cloud GPU rental rates; power assumes ~85%
  sustained GPU utilization plus host-system overhead.

All structural constants live in `SimParams`, so research experiments
(`experiments.py`) can sweep or perturb them. Defaults are seeded and
reproducible.

Run:  python generate_data.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
OUTPUT_CSV = Path("llm_hardware_metrics.csv")

# --------------------------------------------------------------------------- #
#  Hardware catalog
# --------------------------------------------------------------------------- #

GPU_SPECS: dict[str, dict] = {
    #                 VRAM        mem bandwidth  TDP        rental $/GPU-hr
    "RTX 3090":  dict(vram_gb=24,  bw_gbs=936,   tdp_w=350, rate_hr=0.22, tier="Consumer"),
    "RTX 4090":  dict(vram_gb=24,  bw_gbs=1008,  tdp_w=450, rate_hr=0.34, tier="Consumer"),
    "L40S":      dict(vram_gb=48,  bw_gbs=864,   tdp_w=350, rate_hr=0.79, tier="Workstation"),
    "A100 80GB": dict(vram_gb=80,  bw_gbs=2039,  tdp_w=400, rate_hr=1.29, tier="Enterprise"),
    "H100 SXM":  dict(vram_gb=80,  bw_gbs=3350,  tdp_w=700, rate_hr=2.49, tier="Enterprise"),
    "H200 SXM":  dict(vram_gb=141, bw_gbs=4800,  tdp_w=700, rate_hr=3.39, tier="Enterprise"),
}

CLUSTER_SIZES: dict[str, list[int]] = {
    "RTX 3090":  [1, 2],
    "RTX 4090":  [1, 2, 4],
    "L40S":      [1, 2, 4],
    "A100 80GB": [1, 2, 4, 8],
    "H100 SXM":  [1, 2, 4, 8],
    "H200 SXM":  [1, 2, 4, 8],
}

# --------------------------------------------------------------------------- #
#  Model catalog — major open-weight models, mid-2026
#
#  params_b  = total parameters (determines VRAM footprint)
#  active_b  = parameters activated per token (determines bytes streamed;
#              equals params_b for dense models)
#  kv_gb_1k  = FP16 KV-cache GB per 1k tokens. None -> dense fit
#              0.05 * P^0.43 (anchored: ~0.12 for 8B, ~0.31 for 70B).
#              MoE models carry explicit values because KV is set by the
#              attention stack (layers x kv-heads x head-dim; MLA is ~10x
#              smaller), not by expert count. Values are architecture-derived
#              estimates; see thesis §3/§8.
#  Context is capped at 131,072 for the study grid (some models go higher).
# --------------------------------------------------------------------------- #

MODELS: list[dict] = [
    # --- dense -------------------------------------------------------------
    dict(model="Llama-3.2-3B",        family="Llama",    params_b=3,    active_b=3,    max_model_ctx=131_072, kv_gb_1k=None),
    dict(model="Llama-3.1-8B",        family="Llama",    params_b=8,    active_b=8,    max_model_ctx=131_072, kv_gb_1k=None),
    dict(model="Gemma-3-12B",         family="Gemma",    params_b=12,   active_b=12,   max_model_ctx=131_072, kv_gb_1k=None),
    dict(model="Phi-4-14B",           family="Phi",      params_b=14,   active_b=14,   max_model_ctx=16_384,  kv_gb_1k=None),
    dict(model="Qwen3-14B",           family="Qwen",     params_b=14,   active_b=14,   max_model_ctx=131_072, kv_gb_1k=None),
    dict(model="Mistral-Small-3.1-24B", family="Mistral", params_b=24,  active_b=24,   max_model_ctx=131_072, kv_gb_1k=None),
    dict(model="Gemma-3-27B",         family="Gemma",    params_b=27,   active_b=27,   max_model_ctx=131_072, kv_gb_1k=None),
    dict(model="Qwen3-32B",           family="Qwen",     params_b=32,   active_b=32,   max_model_ctx=131_072, kv_gb_1k=None),
    dict(model="Llama-3.3-70B",       family="Llama",    params_b=70,   active_b=70,   max_model_ctx=131_072, kv_gb_1k=None),
    dict(model="Qwen2.5-72B",         family="Qwen",     params_b=72,   active_b=72,   max_model_ctx=131_072, kv_gb_1k=None),
    dict(model="Mistral-Large-123B",  family="Mistral",  params_b=123,  active_b=123,  max_model_ctx=131_072, kv_gb_1k=None),
    dict(model="Llama-3.1-405B",      family="Llama",    params_b=405,  active_b=405,  max_model_ctx=131_072, kv_gb_1k=None),
    # --- Mixture-of-Experts ------------------------------------------------
    dict(model="Qwen3-30B-A3B",       family="Qwen",     params_b=30,   active_b=3,    max_model_ctx=131_072, kv_gb_1k=0.10),
    dict(model="gpt-oss-120B",        family="OpenAI",   params_b=117,  active_b=5.1,  max_model_ctx=131_072, kv_gb_1k=0.07),
    dict(model="Qwen3-235B-A22B",     family="Qwen",     params_b=235,  active_b=22,   max_model_ctx=131_072, kv_gb_1k=0.19),
    dict(model="GLM-4.6-357B",        family="GLM",      params_b=357,  active_b=32,   max_model_ctx=131_072, kv_gb_1k=0.16),
    dict(model="Llama-4-Maverick-400B", family="Llama",  params_b=400,  active_b=17,   max_model_ctx=131_072, kv_gb_1k=0.20),
    dict(model="DeepSeek-V3.2-685B",  family="DeepSeek", params_b=685,  active_b=37,   max_model_ctx=131_072, kv_gb_1k=0.07),
    dict(model="Kimi-K2-1T",          family="Moonshot", params_b=1000, active_b=32,   max_model_ctx=131_072, kv_gb_1k=0.07),
]

MODELS_BY_NAME: dict[str, dict] = {m["model"]: m for m in MODELS}

QUANT_BYTES: dict[str, float] = {"FP16": 2.0, "INT8": 1.0, "INT4": 0.5}

CTX_STEPS = [4_096, 8_192, 16_384, 32_768, 65_536, 131_072]

# Backwards-compatible module constants (canonical values used by the app).
VRAM_USABLE_FRAC = 0.92
WEIGHT_OVERHEAD = 1.08
FRAMEWORK_GB = 2.0
MAX_WASTE_RATIO = 16.0
ASSUMED_REQ_TOKENS = 8_192
MAX_BATCH = 32
BATCH_EXPONENT = 0.82
MBU_MEAN, MBU_SD = 0.55, 0.04
GPU_UTIL = 0.85
SYSTEM_POWER_W = 120.0


# --------------------------------------------------------------------------- #
#  Structural parameters (sweepable by experiments.py)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SimParams:
    """Every structural assumption of the cost model, in one place."""

    vram_usable_frac: float = VRAM_USABLE_FRAC   # runtime-usable share of VRAM
    weight_overhead: float = WEIGHT_OVERHEAD     # multiplier on raw weight bytes
    framework_gb: float = FRAMEWORK_GB           # CUDA ctx + framework baseline
    max_waste_ratio: float = MAX_WASTE_RATIO     # oversize-rig exclusion gate
    assumed_req_tokens: int = ASSUMED_REQ_TOKENS # per-request ctx for batch sizing
    max_batch: int = MAX_BATCH                   # scheduler concurrency cap  (B)
    batch_exponent: float = BATCH_EXPONENT       # dense batching returns     (alpha)
    batch_exponent_moe: float = 0.55             # MoE batching returns (expert-union dilution)
    tp_gamma: float = 0.85                       # TP efficiency per doubling (gamma)
    mbu_mean: float = MBU_MEAN                   # memory-bandwidth utilization
    mbu_sd: float = MBU_SD
    gpu_util: float = GPU_UTIL
    system_power_w: float = SYSTEM_POWER_W
    cost_sd: float = 0.03                        # per-row price jitter
    power_sd: float = 0.02                       # per-row power jitter
    price_mult: dict = field(default_factory=dict)  # optional per-tier price multiplier


def deterministic_params(**overrides) -> SimParams:
    """The noise-free expectation of the model (for exact theory checks)."""
    return SimParams(mbu_sd=0.0, cost_sd=0.0, power_sd=0.0, **overrides)


# --------------------------------------------------------------------------- #
#  Model primitives
# --------------------------------------------------------------------------- #

def kv_gb_per_1k_tokens(m: dict) -> float:
    """FP16 KV-cache per 1k tokens for a catalog model."""
    if m.get("kv_gb_1k") is not None:
        return float(m["kv_gb_1k"])
    return 0.05 * m["params_b"] ** 0.43


def is_moe(m: dict) -> bool:
    return m["active_b"] < m["params_b"]


def alpha_for(m: dict, p: SimParams) -> float:
    return p.batch_exponent_moe if is_moe(m) else p.batch_exponent


def weights_gb_for(m: dict, quant: str, p: SimParams) -> float:
    """VRAM footprint of the model (total params + runtime overhead)."""
    return m["params_b"] * QUANT_BYTES[quant] * p.weight_overhead + p.framework_gb


def stream_gb_for(m: dict, quant: str) -> float:
    """Bytes streamed per generated token (active params only)."""
    return m["active_b"] * QUANT_BYTES[quant]


def multi_gpu_efficiency(n_gpus: int, gamma: float = 0.85) -> float:
    """Tensor-parallel scaling efficiency: gamma per doubling of GPUs."""
    return gamma ** math.log2(n_gpus) if n_gpus > 1 else 1.0


def snap_context(tokens: float) -> int:
    """Snap a raw KV-capacity token count down to a standard context step."""
    viable = [step for step in CTX_STEPS if step <= tokens]
    return viable[-1] if viable else 0


# --------------------------------------------------------------------------- #
#  Cluster-sizing theory (Propositions 1 and 1')
# --------------------------------------------------------------------------- #

def batch_capacity(model_name: str, quant: str, gpu: str, n: int,
                   p: SimParams | None = None) -> int | None:
    """Effective batch at cluster size n, or None if the model does not fit."""
    p = p or SimParams()
    m = MODELS_BY_NAME[model_name]
    w = weights_gb_for(m, quant, p)
    usable = n * GPU_SPECS[gpu]["vram_gb"] * p.vram_usable_frac
    if w > usable:
        return None
    kappa = kv_gb_per_1k_tokens(m) * p.assumed_req_tokens / 1_000
    return int(np.clip((usable - w) // kappa, 1, p.max_batch))


def n_star(model_name: str, quant: str, gpu: str, p: SimParams | None = None) -> int:
    """First-order capacity-saturation cluster size (Prop. 1):

        n*  ~  ceil( (kappa * B + W) / V_usable )

    Per-dollar bandwidth is flat in n (both scale ~linearly) and the TP tax
    only decays it, so the sole per-dollar gain from adding GPUs is KV
    headroom; efficiency peaks near where headroom saturates the cap B.
    `n_star_exact` applies the exact discrete condition and is preferred.
    """
    p = p or SimParams()
    m = MODELS_BY_NAME[model_name]
    w = weights_gb_for(m, quant, p)
    kappa = kv_gb_per_1k_tokens(m) * p.assumed_req_tokens / 1_000
    v_usable = GPU_SPECS[gpu]["vram_gb"] * p.vram_usable_frac
    return max(1, math.ceil((kappa * p.max_batch + w) / v_usable))


def n_star_exact(model_name: str, quant: str, gpu: str,
                 sizes: list[int] | None = None,
                 p: SimParams | None = None) -> int | None:
    """Exact discrete optimum of tokens-per-dollar over cluster sizes (Prop. 1').

    Stepping m -> n multiplies per-dollar throughput by
        gamma ** log2(n/m)  x  (B(n)/B(m)) ** alpha
    Walk the size ladder while that ratio exceeds 1.
    """
    p = p or SimParams()
    m = MODELS_BY_NAME[model_name]
    alpha = alpha_for(m, p)
    sizes = sorted(sizes or CLUSTER_SIZES[gpu])
    viable = [n for n in sizes if batch_capacity(model_name, quant, gpu, n, p)]
    if not viable:
        return None
    cur = viable[0]
    for nxt in viable[viable.index(cur) + 1:]:
        b_cur = batch_capacity(model_name, quant, gpu, cur, p)
        b_nxt = batch_capacity(model_name, quant, gpu, nxt, p)
        gain = (p.tp_gamma ** math.log2(nxt / cur)) * (b_nxt / b_cur) ** alpha
        if gain <= 1.0:
            break
        cur = nxt
    return cur


# --------------------------------------------------------------------------- #
#  Dataset generation
# --------------------------------------------------------------------------- #

def generate_dataset(
    path: Path | str | None = OUTPUT_CSV,
    seed: int = SEED,
    params: SimParams | None = None,
) -> pd.DataFrame:
    """Build the benchmark table, optionally write it to `path`, return it."""
    p = params or SimParams()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for m in MODELS:
        kappa_1k = kv_gb_per_1k_tokens(m)
        alpha = alpha_for(m, p)
        for quant in QUANT_BYTES:
            vram_required = weights_gb_for(m, quant, p)
            stream_gb = stream_gb_for(m, quant)

            for gpu, spec in GPU_SPECS.items():
                for n in CLUSTER_SIZES[gpu]:
                    total_vram = n * spec["vram_gb"]
                    usable_vram = total_vram * p.vram_usable_frac

                    # -- viability gates -------------------------------------
                    if vram_required > usable_vram:
                        continue                      # does not fit
                    if total_vram / vram_required > p.max_waste_ratio:
                        continue                      # absurdly oversized rig

                    # -- throughput ------------------------------------------
                    agg_bw = spec["bw_gbs"] * n
                    eff_bw = agg_bw * multi_gpu_efficiency(n, p.tp_gamma)
                    mbu = float(np.clip(rng.normal(p.mbu_mean, p.mbu_sd), 0.42, 0.68))
                    single_stream_ts = eff_bw * mbu / stream_gb

                    headroom_gb = usable_vram - vram_required
                    kv_per_request = kappa_1k * p.assumed_req_tokens / 1_000
                    batch = int(np.clip(headroom_gb // kv_per_request, 1, p.max_batch))
                    serving_ts = single_stream_ts * batch ** alpha

                    # -- max context window ----------------------------------
                    kv_capacity_tokens = headroom_gb / kappa_1k * 1_000
                    max_ctx = snap_context(min(m["max_model_ctx"], kv_capacity_tokens))
                    if max_ctx < CTX_STEPS[0]:
                        continue                      # cannot serve even 4K context

                    # -- power & cost ----------------------------------------
                    power_w = (n * spec["tdp_w"] * p.gpu_util + p.system_power_w) * float(
                        rng.normal(1.0, p.power_sd)
                    )
                    tier_mult = p.price_mult.get(spec["tier"], 1.0)
                    hourly_cost = max(
                        n * spec["rate_hr"] * tier_mult * float(rng.normal(1.0, p.cost_sd)),
                        0.05,
                    )
                    cost_per_1m = hourly_cost / (serving_ts * 3_600) * 1_000_000
                    tokens_per_dollar = serving_ts * 3_600 / hourly_cost

                    rows.append(
                        dict(
                            model=m["model"],
                            family=m["family"],
                            params_b=m["params_b"],
                            active_params_b=m["active_b"],
                            is_moe=is_moe(m),
                            quantization=quant,
                            gpu=gpu,
                            num_gpus=n,
                            hardware_config=f"{n}x {gpu}",
                            hardware_tier=spec["tier"],
                            total_vram_gb=round(total_vram, 1),
                            vram_required_gb=round(vram_required, 1),
                            memory_bandwidth_gbs=round(agg_bw, 0),
                            max_context_window=max_ctx,
                            tokens_per_sec=round(serving_ts, 1),
                            power_draw_w=round(power_w, 0),
                            hourly_cost_usd=round(hourly_cost, 3),
                            cost_per_1m_tokens_usd=round(cost_per_1m, 3),
                            tokens_per_dollar=round(tokens_per_dollar, 0),
                        )
                    )

    df = (
        pd.DataFrame(rows)
        .sort_values(["params_b", "model", "quantization", "hourly_cost_usd"])
        .reset_index(drop=True)
    )
    if path is not None:
        df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Wrote {OUTPUT_CSV} - {len(df)} viable configurations")
    print(f"  models:        {df['model'].nunique()} ({df['family'].nunique()} families, "
          f"{int(df['is_moe'].sum() > 0) and df[df['is_moe']]['model'].nunique()} MoE)")
    print(f"  clusters:      {df['hardware_config'].nunique()}")
    print(f"  t/s range:     {df['tokens_per_sec'].min():,.0f} - {df['tokens_per_sec'].max():,.0f}")
    print(f"  $/hr range:    {df['hourly_cost_usd'].min():.2f} - {df['hourly_cost_usd'].max():.2f}")
    print(f"  $/1M range:    {df['cost_per_1m_tokens_usd'].min():.2f} - {df['cost_per_1m_tokens_usd'].max():.2f}")

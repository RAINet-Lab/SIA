"""
Core computation helpers for the SIA dashboard.

Pure functions that take a symbolic DataFrame (output of the SIA symbolizer)
and return serializable dicts / DataFrames consumed by the webapp API layer.
Notebooks are not touched; this module re-implements only what the dashboard needs.

Column contract for `symbolic_df`:
  timestep      int      monotonically increasing per slice
  slice_id      int      0 / 1 / 2
  tx_brate      str      e.g. "inc(tx_brate, VeryHigh)"
  tx_pkts       str
  dl_buffer     str
  prb_decision  str      e.g. "const(PRB, VeryHigh)"
  sched_decision str     e.g. "const(sched)"
  action_combined str    prb_decision + " | " + sched_decision
  reward        float
  able_to_improve bool
  alternative   str|None "<action>:<reward_gain>"
  training      bool
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ─── per-KPI Knowledge Graph ────────────────────────────────────────────────

def build_per_kpi_graph(
    df: pd.DataFrame,
    sym_col: str,
    action_col: str = "action_combined",
    slice_id: Optional[int] = None,
    max_t: Optional[int] = None,
) -> dict:
    """
    Build a vis-network-ready knowledge graph for one KPI.

    Nodes  = symbolic states of that KPI (e.g. "inc(tx_brate, VeryHigh)").
    Edges  = state→state transitions; each edge stores the most common action
             taken on that transition and its mean reward.

    Returns {"nodes": [...], "edges": [...]} suitable for vis-network.
    """
    d = df.copy()
    if slice_id is not None:
        d = d[d["slice_id"] == slice_id]
    if max_t is not None:
        d = d[d["timestep"] <= max_t]
    if d.empty:
        return {"nodes": [], "edges": []}

    d = d.sort_values("timestep").reset_index(drop=True)
    states  = d[sym_col].tolist()
    actions = d[action_col].tolist()
    rewards = d["reward"].tolist()

    # Node accumulation
    node_cnt: dict[str, int]   = {}
    node_rew: dict[str, float] = {}
    for s, r in zip(states, rewards):
        node_cnt[s] = node_cnt.get(s, 0) + 1
        node_rew[s] = node_rew.get(s, 0.0) + r

    total = sum(node_cnt.values())

    # Edge accumulation: (src, tgt) → {count, total_reward, actions:{a:{count,total_reward}}}
    edge_acc: dict[tuple, dict] = {}
    for i in range(1, len(states)):
        src, tgt = states[i - 1], states[i]
        act, rew = actions[i], rewards[i]
        key = (src, tgt)
        if key not in edge_acc:
            edge_acc[key] = {"count": 0, "total_reward": 0.0, "actions": {}}
        edge_acc[key]["count"] += 1
        edge_acc[key]["total_reward"] += rew
        ea = edge_acc[key]["actions"]
        if act not in ea:
            ea[act] = {"count": 0, "total_reward": 0.0}
        ea[act]["count"] += 1
        ea[act]["total_reward"] += rew

    # Out-degree totals for edge probabilities
    src_totals: dict[str, int] = {}
    for (src, _), ed in edge_acc.items():
        src_totals[src] = src_totals.get(src, 0) + ed["count"]

    nodes = []
    for state, cnt in node_cnt.items():
        prob = cnt / total
        mean_r = node_rew[state] / cnt
        nodes.append({
            "id": state,
            "label": state,
            "probability": round(prob * 100, 2),
            "occurrence": cnt,
            "mean_reward": round(mean_r, 4),
            "size": max(10, min(80, int(10 + prob * 200))),
            "title": (
                f"State: {state}\n"
                f"Prob: {round(prob * 100, 1)}%\n"
                f"Occurrence: {cnt}\n"
                f"Mean Reward: {round(mean_r, 4)}"
            ),
        })

    edges = []
    for (src, tgt), ed in edge_acc.items():
        src_total = src_totals.get(src, 1)
        prob = ed["count"] / src_total
        if prob * 100 < 0.5:
            continue
        best_act = max(
            ed["actions"].items(),
            key=lambda x: x[1]["total_reward"] / max(x[1]["count"], 1),
        )
        best_act_reward = best_act[1]["total_reward"] / max(best_act[1]["count"], 1)
        edges.append({
            "from": src,
            "to": tgt,
            "percentage": round(prob * 100, 1),
            "width": max(1, min(8, prob * 8)),
            "mean_reward": round(ed["total_reward"] / ed["count"], 4),
            "best_action": best_act[0],
            "best_action_reward": round(best_act_reward, 4),
            "title": (
                f"{src} → {tgt}\n"
                f"Prob: {round(prob * 100, 1)}%\n"
                f"Best action: {best_act[0]}\n"
                f"Mean reward: {round(ed['total_reward'] / ed['count'], 4)}"
            ),
        })

    return {"nodes": nodes, "edges": edges}


# ─── Influence Score ─────────────────────────────────────────────────────────

def compute_influence_scores(
    df: pd.DataFrame,
    kpi_cols: list[str],
    action_col: str = "action_combined",
) -> pd.DataFrame:
    """
    Compute the per-KPI Influence Score at every timestep.

        IS_k = D_KL(P_k || P_∅) × δ(a_t, a*_k)

    Distributions are built from the full history (offline mode), so this
    is suitable for pre-computing on a completed episode.

    Returns DataFrame: timestep, slice_id, kpi, IS, kl_divergence, alignment.
    """
    all_actions = sorted(df[action_col].unique())

    # Build P_k(a | s_k) for each KPI from full history
    kpi_dists: dict[str, dict[str, dict[str, float]]] = {}
    for kpi in kpi_cols:
        counts: dict[str, dict[str, int]] = {}
        for _, row in df.iterrows():
            s, a = row[kpi], row[action_col]
            counts.setdefault(s, {})[a] = counts.get(s, {}).get(a, 0) + 1
        kpi_dists[kpi] = {
            s: {a: c / sum(ac.values()) for a, c in ac.items()}
            for s, ac in counts.items()
        }

    records = []
    for _, row in df.iterrows():
        t   = row["timestep"]
        sid = row["slice_id"]
        at  = row[action_col]

        # Conditional vectors for the current symbolic states
        pk_vecs: dict[str, np.ndarray] = {}
        for kpi in kpi_cols:
            pk = kpi_dists[kpi].get(row[kpi], {})
            pk_vecs[kpi] = np.array([pk.get(a, 1e-10) for a in all_actions], dtype=float)

        # Baseline P_∅(a) = mean over KPIs
        p_base = np.mean(list(pk_vecs.values()), axis=0)
        p_base = np.maximum(p_base, 1e-10)

        for kpi in kpi_cols:
            pk_vec = pk_vecs[kpi]
            mask = pk_vec > 1e-9
            kl = float(np.sum(pk_vec[mask] * np.log(pk_vec[mask] / p_base[mask]))) if mask.any() else 0.0
            kl = max(0.0, kl)

            pk_full = kpi_dists[kpi].get(row[kpi], {})
            a_star  = max(pk_full, key=pk_full.get) if pk_full else at
            alignment = 1.0 if at == a_star else 0.0

            records.append({
                "timestep":     t,
                "slice_id":     sid,
                "kpi":          kpi,
                "IS":           round(kl * alignment, 5),
                "kl_divergence": round(kl, 5),
                "alignment":    alignment,
            })

    return pd.DataFrame(records)


# ─── Mutual Information ───────────────────────────────────────────────────────

def compute_mutual_information(
    df: pd.DataFrame,
    kpi_cols: list[str],
    action_col: str = "action_combined",
) -> dict:
    """
    Compute MI(k; a) for each KPI k.

    For reactive agents this is a single scalar per KPI.
    For forecast-aware agents the caller can pass a per-offset slice of df
    to get the temporal MI profile (replicating Fig. 7).

    Returns {"kpi_name": MI_value, ...}.
    """
    result: dict[str, float] = {}
    for kpi in kpi_cols:
        joint = df.groupby([kpi, action_col]).size().reset_index(name="count")
        total = joint["count"].sum()
        if total == 0:
            result[kpi] = 0.0
            continue
        joint["p"] = joint["count"] / total
        p_s = df[kpi].value_counts(normalize=True).to_dict()
        p_a = df[action_col].value_counts(normalize=True).to_dict()
        mi = 0.0
        for _, row in joint.iterrows():
            s, a, p = row[kpi], row[action_col], row["p"]
            ps = p_s.get(s, 1e-10)
            pa = p_a.get(a, 1e-10)
            if p > 0:
                mi += p * np.log(p / (ps * pa))
        result[kpi] = round(float(mi), 6)
    return result


# ─── Action Refinement Events ─────────────────────────────────────────────────

def extract_refinement_events(
    df: pd.DataFrame,
    action_col: str = "action_combined",
) -> pd.DataFrame:
    """
    Extract timesteps where the Action Refiner would override the agent.

    Returns DataFrame: timestep, slice_id, agent_action, refined_action, reward_gain.
    """
    ref = df[df["able_to_improve"] == True].copy()
    if ref.empty:
        return pd.DataFrame(
            columns=["timestep", "slice_id", "agent_action", "refined_action", "reward_gain"]
        )

    def _parse_alt(alt: object) -> tuple[str, float]:
        s = str(alt)
        if s in ("None", "", "nan"):
            return "", 0.0
        parts = s.rsplit(":", 1)
        if len(parts) == 2:
            try:
                return parts[0].strip(), float(parts[1])
            except ValueError:
                pass
        return s.strip(), 0.0

    ref["refined_action"], ref["reward_gain"] = zip(
        *ref["alternative"].apply(_parse_alt)
    )
    return (
        ref[["timestep", "slice_id", action_col, "refined_action", "reward_gain"]]
        .rename(columns={action_col: "agent_action"})
        .reset_index(drop=True)
    )


# ─── Probability Distribution ─────────────────────────────────────────────────

def compute_probability_distribution(
    df: pd.DataFrame,
    kpi_cols: list[str],
    slice_map: dict[int, str] | None = None,
    max_t: int | None = None,
) -> dict:
    """
    Per-KPI symbolic-state probability distributions, broken down by slice.
    Returns {kpi: {slice_name: {state: prob}}}.
    """
    if slice_map is None:
        slice_map = {0: "eMBB", 1: "mMTC", 2: "URLLC"}
    d = df.copy()
    if max_t is not None:
        d = d[d["timestep"] <= max_t]

    result: dict[str, dict] = {}
    for kpi in kpi_cols:
        kpi_data: dict[str, dict] = {}
        for sid, sname in slice_map.items():
            sdf = d[d["slice_id"] == sid]
            if sdf.empty:
                continue
            probs = sdf[kpi].value_counts(normalize=True)
            kpi_data[sname] = {str(k): round(float(v), 4) for k, v in probs.items()}
        result[kpi] = kpi_data
    return result


# ─── Decision / KPI Patterns (mirrors symbxrl logic) ─────────────────────────

def compute_decision_patterns(
    df: pd.DataFrame,
    action_col: str = "action_combined",
    kpi_cols: list[str] | None = None,
    slice_id: int | None = None,
    max_t: int | None = None,
) -> dict:
    """
    Decision trigrams (A→B→C) and KPI state-transition patterns.
    Returns {"decision_trigrams": [...], "kpi_transitions": {kpi: [...]}}.
    """
    if kpi_cols is None:
        kpi_cols = ["tx_brate", "tx_pkts", "dl_buffer"]
    d = df.copy()
    if slice_id is not None:
        d = d[d["slice_id"] == slice_id]
    if max_t is not None:
        d = d[d["timestep"] <= max_t]
    if d.empty:
        return {"decision_trigrams": [], "kpi_transitions": {}}

    d = d.sort_values("timestep").reset_index(drop=True)
    decisions = d[action_col].tolist()

    tg: dict = {}
    for i in range(len(decisions) - 2):
        a, b, c = decisions[i], decisions[i + 1], decisions[i + 2]
        tg.setdefault((a, b), {}).setdefault(c, 0)
        tg[(a, b)][c] += 1

    trigrams = []
    for (a, b), nexts in tg.items():
        total = sum(nexts.values())
        if total < 3:
            continue
        for c, cnt in sorted(nexts.items(), key=lambda x: -x[1]):
            pct = round(cnt / total * 100, 1)
            if pct >= 5:
                trigrams.append({"from": a, "via": b, "to": c, "probability": pct, "count": cnt})
    trigrams.sort(key=lambda x: -x["count"])

    kpi_transitions: dict[str, list] = {}
    for kpi in kpi_cols:
        series = d[kpi].tolist()
        tr: dict = {}
        for i in range(len(series) - 1):
            a, b = series[i], series[i + 1]
            tr.setdefault(a, {}).setdefault(b, 0)
            tr[a][b] += 1
        rows = []
        for cur, nexts in tr.items():
            total = sum(nexts.values())
            for nxt, cnt in sorted(nexts.items(), key=lambda x: -x[1])[:3]:
                rows.append({"current": cur, "next": nxt, "probability": round(cnt / total * 100, 1), "count": cnt})
        rows.sort(key=lambda x: -x["count"])
        kpi_transitions[kpi] = rows[:12]

    return {"decision_trigrams": trigrams[:15], "kpi_transitions": kpi_transitions}


# ─── Heatmap ─────────────────────────────────────────────────────────────────

def build_heatmap(
    df: pd.DataFrame,
    action_col: str = "action_combined",
    kpi_col: str = "tx_brate",
    slice_id: int | None = None,
    max_t: int | None = None,
) -> dict:
    """
    Action→KPI-effect conditional density heatmap.
    Returns {"decisions": [...], "effects": [...], "matrix": [[...]]}.
    """
    d = df.copy()
    if slice_id is not None:
        d = d[d["slice_id"] == slice_id]
    if max_t is not None:
        d = d[d["timestep"] <= max_t]
    if d.empty:
        return {"decisions": [], "effects": [], "matrix": []}

    ct = pd.crosstab(d[action_col], d[kpi_col], normalize="index") * 100
    return {
        "decisions": ct.index.tolist(),
        "effects":   ct.columns.tolist(),
        "matrix":    ct.values.round(1).tolist(),
    }


# ─── Consistency Over Time ────────────────────────────────────────────────────

def compute_consistency(
    df: pd.DataFrame,
    action_col: str = "action_combined",
    kpi_col: str = "tx_brate",
    slice_id: int | None = None,
    n_checkpoints: int = 60,
) -> dict:
    """
    Policy / effect / explanation consistency at evenly-spaced checkpoints.

    Mirrors symbxrl's get_consistency_over_time; the three metrics are:
      policy_stability    — P(dominant next-state | source-state), weighted
      effect_consistency  — P(dominant KPI symbol | transition edge), weighted
      explanation_stability — 1 − mean|Δedge_prob| vs previous checkpoint

    Returns {"timesteps": [...], "policy_stability": [...], ...}.
    """
    d = df.copy()
    if slice_id is not None:
        d = d[d["slice_id"] == slice_id]
    d = d.sort_values("timestep").reset_index(drop=True)

    empty: dict = {
        "timesteps": [], "policy_stability": [],
        "effect_consistency": [], "explanation_stability": [],
    }
    if len(d) < 5:
        return empty

    combined      = d[action_col].tolist()
    kpi_vals      = d[kpi_col].tolist()
    timesteps_all = d["timestep"].tolist()
    N = len(d)

    n_pts = min(n_checkpoints, N - 4)
    start = 4
    raw_idx = [
        int(round(start + i * (N - 1 - start) / max(n_pts - 1, 1)))
        for i in range(n_pts)
    ]
    checkpoints = sorted(set(raw_idx))

    out_t: list  = []
    out_ps: list = []
    out_ec: list = []
    out_es: list = []
    prev_edge_probs: dict = {}

    for idx in checkpoints:
        n   = idx + 1
        comb = combined[:n]
        kpis = kpi_vals[:n]

        # Transition counts
        trans: dict = {}
        for i in range(n - 1):
            s, t2 = comb[i], comb[i + 1]
            trans.setdefault(s, {}).setdefault(t2, 0)
            trans[s][t2] += 1

        # Policy stability
        total_w = total_n = 0
        for tgts in trans.values():
            n_src = sum(tgts.values())
            max_p = max(tgts.values()) / n_src
            total_w += max_p * n_src
            total_n += n_src
        ps = total_w / total_n if total_n > 0 else 0.0

        # Effect consistency
        eff_map: dict = {}
        for i in range(n - 1):
            key2 = (comb[i], comb[i + 1])
            eff  = str(kpis[i + 1])
            eff_map.setdefault(key2, {}).setdefault(eff, 0)
            eff_map[key2][eff] += 1
        ew = en = 0
        for effs in eff_map.values():
            n_eff = sum(effs.values())
            max_p = max(effs.values()) / n_eff
            ew += max_p * n_eff
            en += n_eff
        ec = ew / en if en > 0 else 0.0

        # Explanation stability
        edge_probs: dict = {}
        for src, tgts in trans.items():
            n_src = sum(tgts.values())
            for tgt, cnt in tgts.items():
                edge_probs[(src, tgt)] = cnt / n_src

        if not prev_edge_probs:
            expl_stab = None
        else:
            all_edges = set(edge_probs) | set(prev_edge_probs)
            mean_delta = sum(
                abs(edge_probs.get(e, 0.0) - prev_edge_probs.get(e, 0.0))
                for e in all_edges
            ) / len(all_edges)
            expl_stab = round(max(0.0, 1.0 - mean_delta) * 100, 1)

        prev_edge_probs = edge_probs
        out_t.append(int(timesteps_all[idx]))
        out_ps.append(round(ps * 100, 1))
        out_ec.append(round(ec * 100, 1))
        out_es.append(expl_stab)

    return {
        "timesteps":            out_t,
        "policy_stability":     out_ps,
        "effect_consistency":   out_ec,
        "explanation_stability": out_es,
    }

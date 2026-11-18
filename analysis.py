"""Runs every analysis for the report and saves results.json + charts in ./out"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fanline import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import networkx as nx

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

# ---- chart style -----------------------------------------------------------
for f in fm.findSystemFonts():
    if "Inter" in f:
        fm.fontManager.addfont(f)
INK, GREY, LIGHT = "#1d2433", "#6b7280", "#d7dbe0"
ACC, TEAL, SAND = "#c8553d", "#2f6f73", "#d9a441"
MCOL = {"Standard": "#2f6f73", "Decorative": "#d9a441", "BLDC": "#c8553d"}
plt.rcParams.update({
    "font.family": "Inter", "font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK,
    "xtick.color": GREY, "ytick.color": GREY, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.color": "#eceef1", "grid.linewidth": 0.8, "axes.axisbelow": True,
})

def save(fig, name):
    fig.savefig(os.path.join(OUT, name)); plt.close(fig)

R = {}
p = Problem()
C, W = p.cycle_time, p.window
R["base"] = {"demand": p.demand, "available_s": p.available, "cycle_time": C, "window": W,
             "mix": p.mix, "total_work_weighted": p.total_work(),
             "total_work_model": {m: p.total_work(m) for m in MODELS},
             "n_min": math.ceil(p.total_work() / C),
             "daily_units": {m: round(p.demand * p.mix[m]) for m in MODELS}}
R["tasks"] = [{"id": i, "desc": TASKS[i][0], "preds": TASKS[i][1],
               **{m: p.t(i, m) for m in MODELS}, "weighted": round(p.tbar(i), 2),
               "followers": len(p.succ_all(i)), "rpw": round(positional_weights(p)[i], 2)} for i in p.ids]

# ---- heuristics vs IP --------------------------------------------------------
sols = {}
for rule in ["ORDER", "LTT", "MFT", "RPW"]:
    sols[rule] = heuristic(p, rule)
sols["IP"], R["ip_status"] = ip_min_stations(p)
N_ip = len(sols["IP"])
sols["FINAL"], L, R["minmax_status"] = ip_min_max_load(p, N_ip)
_ord = sorted(sols["FINAL"], key=lambda st: min(p.ids.index(i) for i in st))   # number stations by first task
_pos = {i: k for k, st in enumerate(_ord) for i in st}
if all(_pos[a] <= _pos[b] for a, b in p.prec_pairs()):
    sols["FINAL"] = _ord
R["minmax_L"] = L

def pack(st):
    m = measures(p, st)
    return {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k != "loads"},
            "loads": [round(x, 2) for x in m["loads"]], "stations_tasks": st,
            "model_loads": {mm: station_loads(p, st, mm) for mm in MODELS}}
R["solutions"] = {k: pack(v) for k, v in sols.items()}

# ---- Type II: capacity for a given number of operators -----------------------
cap = []
for S in range(6, 13):
    st, Cs, status = ip_min_cycle_time(p, S)
    cap.append({"stations": S, "cycle_time": round(Cs, 3), "output_per_day": math.floor(p.available / Cs),
                "status": status, "stations_tasks": st})
R["type2"] = cap

# ---- sequencing ----------------------------------------------------------------
final = sols["FINAL"]
units = R["base"]["daily_units"]
day_batch = [m for m in MODELS for _ in range(units[m])]
level = goal_chasing(p.mix)
best, best_key, n_perm = best_sequence(p, final)
cycles = p.demand // 10
def day_stats(seq, reps):
    r = simulate_sequence(p, final, seq * reps if reps > 1 else seq, cycles=1)
    return {"utility_s": round(sum(x["utility"] for x in r), 1),
            "per_station_utility": [round(x["utility"], 1) for x in r],
            "idle_s": round(sum(x["idle"] for x in r), 1)}
R["sequencing"] = {
    "mps": mps(p.mix), "goal_chasing": level, "best": best, "permutations_checked": n_perm,
    "day_batch": day_stats(day_batch, 1),
    "goal_chasing_day": day_stats(level, cycles),
    "best_day": day_stats(best, cycles),
}
# drift trace at the busiest station for the decorative batch vs the level sequence
st7 = max(range(len(final)), key=lambda k: R["sequencing"]["day_batch"]["per_station_utility"][k])
R["sequencing"]["worst_station"] = st7 + 1

# ---- parallel stations at high demand ---------------------------------------------
par = {}
for d in [480, 560, 600, 640]:
    q = Problem(demand=d)
    st, z, status = ip_parallel(q)
    st_np, s_np = ip_min_stations(q, S=16)
    par[d] = {"cycle_time": round(q.cycle_time, 2), "operators_parallel": sum(1 + int(v) for v in z),
              "stations_parallel": len(st), "doubled": [k + 1 for k, v in enumerate(z) if v],
              "stations_tasks": st, "operators_no_parallel": len(st_np), "status": status}
R["parallel"] = par

# ---- sensitivity: demand ------------------------------------------------------------
sens = []
for d in range(300, 661, 20):
    q = Problem(demand=d)
    row = {"demand": d, "cycle_time": round(q.cycle_time, 2), "n_min": math.ceil(q.total_work() / q.cycle_time)}
    row["rpw"] = len(heuristic(q, "RPW"))
    st, _ = ip_min_stations(q, S=16); row["ip"] = len(st)
    st, z, _ = ip_parallel(q, S=14); row["ip_parallel"] = sum(1 + int(v) for v in z)
    row["eff_ip"] = round(q.total_work() / (row["ip"] * q.cycle_time), 4)
    row["eff_par"] = round(q.total_work() / (row["ip_parallel"] * q.cycle_time), 4)
    sens.append(row)
R["sens_demand"] = sens

# ---- sensitivity: product mix (BLDC share, decorative fixed at 20%) ------------------
mixs = []
for b in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    q = Problem(mix={"Standard": round(0.8 - b, 2), "Decorative": 0.2, "BLDC": b})
    st, _ = ip_min_stations(q, S=14)
    mixs.append({"bldc": b, "standard": round(0.8 - b, 2), "work": round(q.total_work(), 1),
                 "n_min": math.ceil(q.total_work() / q.cycle_time), "ip": len(st),
                 "rpw": len(heuristic(q, "RPW")),
                 "eff": round(q.total_work() / (len(st) * q.cycle_time), 4)})
R["sens_mix"] = mixs

json.dump(R, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)

# =================================== CHARTS ===================================
# 1. precedence diagram
G = nx.DiGraph(); G.add_nodes_from(p.ids); G.add_edges_from(p.prec_pairs())
level = {}
for n in nx.topological_sort(G):
    level[n] = max([level[q] + 1 for q in G.predecessors(n)], default=0)
cols = {}
for n, l in level.items():
    cols.setdefault(l, []).append(n)
pos = {}
for l, ns in cols.items():
    for k, n in enumerate(ns):
        pos[n] = (l * 1.55, -(k - (len(ns) - 1) / 2) * 1.25)
pos["T9"] = (level["T8"] * 1.55, -1.9)
fig, ax = plt.subplots(figsize=(11, 4.2)); ax.grid(False); ax.axis("off")
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9aa1ab", arrows=True, arrowsize=11, width=1,
                       node_size=1300, connectionstyle="arc3,rad=0.04")
special = {"T13": MCOL["Decorative"], "T15": MCOL["BLDC"]}
for n, (x, y) in pos.items():
    col = special.get(n, INK)
    ax.add_patch(plt.Circle((x, y), 0.36, facecolor="white", edgecolor=col, lw=1.6, zorder=3))
    ax.text(x, y + 0.07, n, ha="center", va="center", fontsize=9.5, fontweight="semibold", color=col, zorder=4)
    ax.text(x, y - 0.17, f"{p.tbar(n):.1f}s", ha="center", va="center", fontsize=7.5, color=GREY, zorder=4)
ax.set_xlim(-0.6, max(x for x, _ in pos.values()) + 0.6)
ax.set_ylim(-2.5, 2.75)
ax.text(-0.5, 2.55, "Orange ring: decorative-only task   Red ring: BLDC-only task   Number: demand-weighted time",
        fontsize=8, color=GREY)
save(fig, "fig1_precedence.png")

# 2. station loads: best heuristic vs IP final
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, key, title in [(axes[0], "RPW", "Best heuristic (RPW): 10 stations"),
                       (axes[1], "FINAL", f"Integer program: {N_ip} stations")]:
    loads = R["solutions"][key]["loads"]
    ax.bar(range(1, len(loads) + 1), loads, color=TEAL if key == "FINAL" else "#9aa1ab", width=0.62)
    ax.bar(range(1, len(loads) + 1), [C - l for l in loads], bottom=loads, color="none",
           edgecolor=LIGHT, hatch="////", width=0.62, lw=0)
    ax.axhline(C, color=ACC, lw=1.3, ls="--"); ax.text(len(loads) + 0.45, C + 1, f"C = {C:.2f}s", color=ACC, fontsize=8.5, ha="right")
    ax.set_title(title); ax.set_xlabel("Station"); ax.set_xticks(range(1, len(loads) + 1)); ax.set_ylim(0, 72)
axes[0].set_ylabel("Weighted station time (s)")
fig.text(0.01, -0.03, "Hatched area = idle time per cycle", fontsize=8, color=GREY)
save(fig, "fig2_station_loads.png")

# 3. per-model loads for final design
fig, ax = plt.subplots(figsize=(11, 3.7))
n = len(final); w = 0.26
for k, m in enumerate(MODELS):
    ax.bar([s + 1 + (k - 1) * w for s in range(n)], station_loads(p, final, m), width=w, color=MCOL[m], label=m)
ax.axhline(C, color=INK, lw=1.1, ls="--"); ax.text(0.45, C + 0.9, f"Cycle time {C:.2f}s", fontsize=8.5, ha="left")
ax.axhline(W, color=ACC, lw=1.1); ax.text(0.45, W + 0.9, f"Station window {W:.1f}s", fontsize=8.5, ha="left", color=ACC)
ax.set_xticks(range(1, n + 1)); ax.set_xlabel("Station"); ax.set_ylabel("Station time (s)"); ax.set_ylim(0, 76); ax.set_xlim(0.3, n + 0.7)
ax.set_title("Station time for each model in the final design"); ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
save(fig, "fig3_model_loads.png")

# 4. sequencing: utility work per station
fig, ax = plt.subplots(figsize=(11, 3.4))
a = R["sequencing"]["day_batch"]["per_station_utility"]; b = R["sequencing"]["goal_chasing_day"]["per_station_utility"]
x = range(1, len(a) + 1)
ax.bar([i - 0.18 for i in x], a, width=0.36, color="#9aa1ab", label="Daily batches (all Standard, then Decorative, then BLDC)")
ax.bar([i + 0.18 for i in x], b, width=0.36, color=TEAL, label="Level sequence S-B-D-S-S-B-S-D-B-S (repeated)")
ax.set_xticks(list(x)); ax.set_xlabel("Station"); ax.set_ylabel("Work needing a helper (s/day)")
ax.set_title("Running models in big batches overloads stations; a level sequence does not")
for i in x:
    ax.text(i + 0.18, 12, "0", ha="center", fontsize=8, color=TEAL)
ax.legend(loc="upper left")
save(fig, "fig4_sequencing.png")

# 5. demand sensitivity
fig, ax = plt.subplots(figsize=(11, 3.8))
d = [r["demand"] for r in sens]
ax.step(d, [r["n_min"] for r in sens], where="mid", color=LIGHT, lw=2, label="Theoretical minimum")
ax.step(d, [r["rpw"] for r in sens], where="mid", color="#9aa1ab", lw=1.6, label="Best heuristic (RPW)")
ax.step(d, [r["ip"] for r in sens], where="mid", color=TEAL, lw=2, label="Integer program")
ax.step(d, [r["ip_parallel"] for r in sens], where="mid", color=ACC, lw=2, ls="--", label="Integer program + parallel stations")
ax.axvline(p.demand, color=INK, lw=0.8, ls=":"); ax.text(p.demand + 3, 5.3, "base case\n480/day", fontsize=8, color=INK)
ax.set_xlabel("Daily demand (fans)"); ax.set_ylabel("Operators needed"); ax.set_ylim(4.5, 15)
ax.set_title("Operators needed as demand grows"); ax.legend(loc="upper left")
save(fig, "fig5_demand.png")

# 6. type II capacity
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot([c["stations"] for c in cap], [c["output_per_day"] for c in cap], marker="o", color=TEAL, lw=2)
for c in cap:
    ax.text(c["stations"], c["output_per_day"] + 14, str(c["output_per_day"]), ha="center", fontsize=8.5, color=INK)
ax.axhline(p.demand, color=ACC, ls="--", lw=1); ax.text(6, p.demand + 8, "demand 480", color=ACC, fontsize=8.5)
ax.set_xlabel("Number of stations (operators)"); ax.set_ylabel("Maximum fans per day"); ax.set_ylim(250, 720)
ax.set_title("Line capacity for a given crew (Type II model)")
save(fig, "fig6_capacity.png")
print("done", OUT)

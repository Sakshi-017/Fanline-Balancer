"""
fanline.py - Mixed-model assembly line balancing for a ceiling fan line.
MSL304 Operations Management project.

Contents
  1. Data: tasks, precedence, task times per model, demand
  2. Line measures: cycle time, theoretical minimum, efficiency, balance delay, smoothness
  3. Heuristics: Longest Task Time, Most Following Tasks, Ranked Positional Weight, naive (task order)
  4. Integer programs (PuLP + HiGHS), following the lecture formulation:
       - Type I : minimise number of stations for a given cycle time
       - Type II: minimise cycle time for a given number of stations
       - Min-max: minimise the maximum station load
       - Extension: parallel stations (two operators) when demand is high
  5. Mixed-model sequencing: operator drift simulation and exhaustive search
"""
from __future__ import annotations
import math, itertools
from dataclasses import dataclass, field
import pulp

# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
MODELS = ["Standard", "Decorative", "BLDC"]

# id: (description, predecessors, (t_Standard, t_Decorative, t_BLDC) in seconds)
TASKS = {
    "T1":  ("Press shaft into stator core",              [],                         (24, 24, 24)),
    "T2":  ("Fit bearing in bottom cover",               [],                         (16, 16, 16)),
    "T3":  ("Fit bearing in top cover",                  [],                         (16, 16, 16)),
    "T4":  ("Mount stator-shaft in bottom cover",        ["T1", "T2"],               (22, 22, 22)),
    "T5":  ("Connect winding leads (BLDC: mount PCB driver)", ["T4"],                (28, 28, 48)),
    "T6":  ("Close top cover, tighten screws",           ["T3", "T4"],               (30, 30, 30)),
    "T7":  ("Motor no-load test",                        ["T5", "T6"],               (32, 32, 38)),
    "T8":  ("Clean and finish motor body",               ["T7"],                     (18, 30, 18)),
    "T9":  ("Canopy and down-rod sub-assembly",          [],                         (26, 26, 26)),
    "T10": ("Fit down-rod to motor",                     ["T8", "T9"],               (20, 20, 20)),
    "T11": ("Fit three blade flanges",                   ["T8"],                     (30, 30, 30)),
    "T12": ("Fit bottom cap / trim ring",                ["T11"],                    (12, 26, 12)),
    "T13": ("Fit LED light kit and wire it",             ["T12"],                    (0, 44, 0)),
    "T14": ("Connect supply wiring in canopy",           ["T10"],                    (22, 22, 12)),
    "T15": ("Pair remote, firmware check",               ["T14"],                    (0, 0, 30)),
    "T16": ("Full-speed run and wobble test",            ["T12", "T13", "T14", "T15"], (38, 42, 40)),
    "T17": ("Rating label and QC inspection",            ["T16"],                    (18, 18, 22)),
    "T18": ("Pack fan kit in carton",                    ["T17"],                    (34, 38, 36)),
}

AVAILABLE_TIME = 8 * 3600 - 30 * 60       # one 8 h shift minus 30 min breaks = 27,000 s
DEMAND = 480                               # fans per day
MIX = {"Standard": 0.5, "Decorative": 0.2, "BLDC": 0.3}
WINDOW_FACTOR = 1.2                        # station length = 1.2 x cycle time (operator may drift)


@dataclass
class Problem:
    tasks: dict = field(default_factory=lambda: TASKS)
    mix: dict = field(default_factory=lambda: dict(MIX))
    demand: float = DEMAND
    available: float = AVAILABLE_TIME
    window_factor: float = WINDOW_FACTOR

    @property
    def ids(self):
        return list(self.tasks)

    @property
    def cycle_time(self):
        return self.available / self.demand

    @property
    def window(self):
        return self.window_factor * self.cycle_time

    def t(self, i, m):
        return self.tasks[i][2][MODELS.index(m)]

    def tbar(self, i):
        """Demand-weighted task time."""
        return sum(self.mix[m] * self.t(i, m) for m in MODELS)

    def preds(self, i):
        return self.tasks[i][1]

    def prec_pairs(self):
        return [(p, j) for j in self.ids for p in self.preds(j)]

    def succ_all(self, i):
        """All tasks that follow i (direct and indirect)."""
        out, stack = set(), [j for j in self.ids if i in self.preds(j)]
        while stack:
            k = stack.pop()
            if k not in out:
                out.add(k)
                stack += [j for j in self.ids if k in self.preds(j)]
        return out

    def total_work(self, m=None):
        return sum(self.t(i, m) if m else self.tbar(i) for i in self.ids)


# ---------------------------------------------------------------------------
# 2. LINE MEASURES
# ---------------------------------------------------------------------------
def station_loads(prob: Problem, stations, m=None):
    return [sum(prob.t(i, m) if m else prob.tbar(i) for i in s) for s in stations]


def measures(prob: Problem, stations, C=None, workers=None):
    C = C or prob.cycle_time
    loads = station_loads(prob, stations)
    n = workers or len(stations)
    T = prob.total_work()
    eff = T / (n * C)
    return {
        "stations": len(stations),
        "workers": n,
        "cycle_time": C,
        "loads": loads,
        "idle_per_cycle": n * C - T,
        "efficiency": eff,
        "balance_delay": 1 - eff,
        "smoothness": math.sqrt(sum((max(loads) - l) ** 2 for l in loads)),
        "max_load": max(loads),
        "output_per_day": prob.available / max(loads),
    }


# ---------------------------------------------------------------------------
# 3. HEURISTICS
# ---------------------------------------------------------------------------
def _fits(prob, station, task, C, W):
    if sum(prob.tbar(i) for i in station) + prob.tbar(task) > C + 1e-9:
        return False
    return all(sum(prob.t(i, m) for i in station) + prob.t(task, m) <= W + 1e-9 for m in MODELS)


def heuristic(prob: Problem, rule: str, C=None, W=None):
    """Assign tasks one at a time. rule in {'LTT','MFT','RPW','ORDER'}."""
    C = C or prob.cycle_time
    W = W or prob.window
    follow = {i: prob.succ_all(i) for i in prob.ids}
    priority = {
        "LTT": lambda i: (prob.tbar(i), len(follow[i])),
        "MFT": lambda i: (len(follow[i]), prob.tbar(i)),
        "RPW": lambda i: (prob.tbar(i) + sum(prob.tbar(j) for j in follow[i]), prob.tbar(i)),
    }
    done, stations, cur = set(), [], []
    while len(done) < len(prob.ids):
        avail = [i for i in prob.ids if i not in done and all(p in done for p in prob.preds(i))]
        if rule == "ORDER":   # naive: next task in list order, open new station when it does not fit
            nxt = avail[0]
            cand = [nxt] if _fits(prob, cur, nxt, C, W) else []
        else:
            cand = sorted([i for i in avail if _fits(prob, cur, i, C, W)], key=priority[rule], reverse=True)
        if cand:
            cur.append(cand[0]); done.add(cand[0])
        else:
            if not cur:
                raise ValueError("A task does not fit even in an empty station: cycle time too short")
            stations.append(cur); cur = []
    if cur:
        stations.append(cur)
    return stations


def positional_weights(prob):
    return {i: prob.tbar(i) + sum(prob.tbar(j) for j in prob.succ_all(i)) for i in prob.ids}


# ---------------------------------------------------------------------------
# 4. INTEGER PROGRAMS
# ---------------------------------------------------------------------------
def _base_model(prob, S, name, per_model=True, C=None, W=None):
    I = prob.ids
    lp = pulp.LpProblem(name, pulp.LpMinimize)
    x = lp.add_variable_dicts("x", (I, range(S)), cat="Binary")
    y = lp.add_variable_dicts("y", range(S), cat="Binary")
    for i in I:                                                   # each task in exactly one station
        lp += pulp.lpSum(x[i][s] for s in range(S)) == 1
    for i in I:                                                   # task only in an open station
        for s in range(S):
            lp += x[i][s] <= y[s]
    for (i, j) in prob.prec_pairs():                              # precedence, cumulative form
        for s in range(S):
            lp += pulp.lpSum(x[i][k] for k in range(s + 1)) >= pulp.lpSum(x[j][k] for k in range(s + 1))
    for s in range(S - 1):                                        # open stations first
        lp += y[s] >= y[s + 1]
    return lp, x, y


def _solve(lp, time_limit=120):
    r = lp.solve(pulp.HiGHS(msg=False, timeLimit=time_limit))
    return r.status.name


def _extract(prob, x, S):
    st = [[i for i in prob.ids if x[i][s].value() > 0.5] for s in range(S)]
    return [s for s in st if s]


def ip_min_stations(prob: Problem, C=None, W=None, S=None, per_model=True):
    """Type I: min number of stations, weighted load <= C, each model's load <= W."""
    C = C or prob.cycle_time; W = W or prob.window
    S = S or len(heuristic(prob, "RPW", C, W)) + 1
    lp, x, y = _base_model(prob, S, "TypeI")
    lp += pulp.lpSum(y[s] for s in range(S))
    for s in range(S):
        lp += pulp.lpSum(prob.tbar(i) * x[i][s] for i in prob.ids) <= C * y[s]
        if per_model:
            for m in MODELS:
                lp += pulp.lpSum(prob.t(i, m) * x[i][s] for i in prob.ids) <= W * y[s]
    status = _solve(lp)
    return _extract(prob, x, S), status


def ip_min_cycle_time(prob: Problem, S: int, per_model=True):
    """Type II: given S stations, minimise cycle time (weighted load); window = factor x C."""
    lp, x, y = _base_model(prob, S, "TypeII")
    C = lp.add_variable("C", lowBound=0)
    lp += C
    lp += pulp.lpSum(y[s] for s in range(S)) <= S
    for s in range(S):
        lp += pulp.lpSum(prob.tbar(i) * x[i][s] for i in prob.ids) <= C
        if per_model:
            for m in MODELS:
                lp += pulp.lpSum(prob.t(i, m) * x[i][s] for i in prob.ids) <= prob.window_factor * C
    status = _solve(lp)
    return _extract(prob, x, S), C.value(), status


def ip_min_max_load(prob: Problem, S: int):
    """Min-max load across S stations, while each model's load <= window."""
    lp, x, y = _base_model(prob, S, "MinMax")
    L = lp.add_variable("L", lowBound=0)
    lp += L
    lp += pulp.lpSum(y[s] for s in range(S)) <= S
    for s in range(S):
        lp += pulp.lpSum(prob.tbar(i) * x[i][s] for i in prob.ids) <= L
        for m in MODELS:
            lp += pulp.lpSum(prob.t(i, m) * x[i][s] for i in prob.ids) <= prob.window
    status = _solve(lp)
    return _extract(prob, x, S), L.value(), status


def ip_parallel(prob: Problem, C=None, W=None, S=14):
    """Extension: a station may be staffed by 2 operators working on alternate units
    (parallel stations). Then its capacity doubles. Minimise total operators."""
    C = C or prob.cycle_time; W = W or prob.window
    lp, x, y = _base_model(prob, S, "Parallel")
    z = lp.add_variable_dicts("z", range(S), cat="Binary")     # 1 = station doubled
    lp += pulp.lpSum(y[s] + z[s] for s in range(S)) + 0.01 * pulp.lpSum(z[s] for s in range(S))
    for s in range(S):
        lp += z[s] <= y[s]
        lp += pulp.lpSum(prob.tbar(i) * x[i][s] for i in prob.ids) <= C * (y[s] + z[s])
        for m in MODELS:
            lp += pulp.lpSum(prob.t(i, m) * x[i][s] for i in prob.ids) <= W * (y[s] + z[s])
    status = _solve(lp, 180)
    st, par = [], []
    for s in range(S):
        tasks = [i for i in prob.ids if x[i][s].value() > 0.5]
        if tasks:
            st.append(tasks); par.append(bool(z[s].value() > 0.5))
    return st, par, status


# ---------------------------------------------------------------------------
# 5. MIXED-MODEL SEQUENCING
# ---------------------------------------------------------------------------
def simulate_sequence(prob: Problem, stations, seq, cycles=3):
    """Paced conveyor: a unit enters each station every C seconds; the station is W long.
    The operator starts where the previous unit left them. Work beyond W cannot be done by
    the operator and needs a utility worker. Returns per-station utility work and idle time
    for the last cycle (steady state), plus the operator's start offsets for plotting."""
    C, W = prob.cycle_time, prob.window
    res = []
    for st in stations:
        tm = {m: sum(prob.t(i, m) for i in st) for m in MODELS}
        start, util, idle, trace = 0.0, 0.0, 0.0, []
        for c in range(cycles):
            u_c = i_c = 0.0
            for m in seq:
                finish = start + tm[m]
                if finish > W:
                    u_c += finish - W; finish = W
                trace.append((start, finish))
                nxt = finish - C
                if nxt < 0:
                    i_c += -nxt; nxt = 0.0
                start = nxt
            if c == cycles - 1:
                util, idle = u_c, i_c
        res.append({"utility": util, "idle": idle, "trace": trace[-len(seq):], "loads": tm})
    return res


def mps(mix, n=10):
    """Minimum part set: units of each model in a repeating cycle of n fans."""
    return {m: round(mix[m] * n) for m in MODELS}


def goal_chasing(mix, n=10):
    """Toyota goal-chasing / level scheduling: at each slot pick the model whose cumulative
    production is furthest behind its ideal rate."""
    counts = {m: 0 for m in MODELS}; need = mps(mix, n); seq = []
    for k in range(1, n + 1):
        best = max((m for m in MODELS if counts[m] < need[m]),
                   key=lambda m: k * mix[m] - counts[m])
        seq.append(best); counts[best] += 1
    return seq


def best_sequence(prob, stations, n=10):
    """Exhaustive search over all distinct orders of the minimum part set."""
    need = mps(prob.mix, n)
    base = [m for m in MODELS for _ in range(need[m])]
    best, best_key = None, None
    seen = set()
    for perm in itertools.permutations(base):
        if perm in seen:
            continue
        seen.add(perm)
        r = simulate_sequence(prob, stations, list(perm))
        key = (round(sum(x["utility"] for x in r), 6), round(sum(x["idle"] for x in r), 6))
        if best_key is None or key < best_key:
            best, best_key = list(perm), key
    return best, best_key, len(seen)

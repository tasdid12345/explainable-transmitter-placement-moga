"""Sensitivity-guided variable-length GA vs a standard variable-length MOGA
on a synthetic transmitter placement problem (coverage vs cost trade-off)."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt

SEED, SEEDS = 42, [42, 43, 44]                      # instance seed and GA run seeds
N_SITES, N_DEMAND, GRID, RADIUS = 400, 1000, 20.0, 2.0
POP, GENS, P_MUT, TEST_FRAC = 60, 100, 0.9, 0.25    # identical budget for both methods
MAX_K = 40                                          # max sites in a random initial solution

# ---------- synthetic instance (fixed seed) ----------
rng = np.random.default_rng(SEED)
site_xy = rng.uniform(0, GRID, (N_SITES, 2))
demand_xy = rng.uniform(0, GRID, (N_DEMAND, 2))
costs = rng.uniform(1.0, 5.0, N_SITES)
COST_TOT = costs.sum()
# coverage matrix: demand point d is covered by site s if within RADIUS
COV_ALL = np.linalg.norm(demand_xy[:, None] - site_xy[None], axis=2) <= RADIUS
# single random train/test split of demand points; test is used only at the end
perm = rng.permutation(N_DEMAND)
n_test = int(TEST_FRAC * N_DEMAND)
C_TE, C_TR = COV_ALL[perm[:n_test]], COV_ALL[perm[n_test:]]
print(f"instance: {N_SITES} sites, {N_DEMAND} demand points (train={len(C_TR)}, test={len(C_TE)})")


# ---------- objectives and Pareto utilities (all minimised) ----------
def objectives(sol, C):
    """(-coverage fraction, normalised cost) for a site list."""
    return np.array([-C[:, sol].any(axis=1).mean(), costs[sol].sum() / COST_TOT])


def nondom_mask(F):
    """Mask of non-dominated rows of F (minimisation)."""
    dom = np.all(F[:, None] <= F[None], axis=2) & np.any(F[:, None] < F[None], axis=2)
    return ~dom.any(axis=0)


def dominated_frac(A, B):
    """Fraction of rows of A dominated by at least one row of B (minimisation)."""
    d = np.all(B[:, None] <= A[None], axis=2) & np.any(B[:, None] < A[None], axis=2)
    return float(d.any(axis=0).mean())


def hypervolume(F):
    """2-D hypervolume w.r.t. reference (coverage=0, normalised cost=1); higher is better."""
    F = F[nondom_mask(F)]
    cov, cost = -F[:, 0], F[:, 1]
    o = np.argsort(cov); cov, cost = cov[o], cost[o]
    xprev = np.concatenate([[0.0], cov[:-1]])
    return float(np.sum((cov - xprev) * (1 - cost)))


def rank_crowd(F):
    """NSGA-II non-dominated rank and crowding distance."""
    n = len(F)
    dom = np.all(F[:, None] <= F[None], axis=2) & np.any(F[:, None] < F[None], axis=2)
    cnt = dom.sum(axis=0); rank = np.zeros(n, int); done = np.zeros(n, bool); f = 0
    while not done.all():
        cur = np.where((cnt == 0) & ~done)[0]
        rank[cur] = f; done[cur] = True
        cnt = cnt - dom[cur].sum(axis=0); f += 1
    crowd = np.zeros(n)
    for f in np.unique(rank):
        idx = np.where(rank == f)[0]
        for m in range(F.shape[1]):
            o = idx[np.argsort(F[idx, m])]
            crowd[o[0]] = crowd[o[-1]] = np.inf
            span = F[o[-1], m] - F[o[0], m]
            if span > 0 and len(o) > 2:
                crowd[o[1:-1]] += (F[o[2:], m] - F[o[:-2], m]) / span
    return rank, crowd


# ---------- variable-length GA operators ----------
def random_sol(r):
    return np.sort(r.choice(N_SITES, r.integers(1, MAX_K + 1), replace=False))


def crossover(a, b, r):
    """One-point crossover on variable-length site lists."""
    i, j = r.integers(0, len(a) + 1), r.integers(0, len(b) + 1)
    child = np.unique(np.concatenate([a[:i], b[j:]]))
    return child if child.size else a[:1]


def mutate(sol, r, weights=None):
    """Add / remove / swap mutation; if weights is given, new sites are drawn by weight."""
    s = set(sol.tolist())

    def new_site():
        p = np.ones(N_SITES) if weights is None else weights.copy()
        p[list(s)] = 0  # only sites not already in the solution
        return int(r.choice(N_SITES, p=p / p.sum()))

    op = r.integers(3)
    if op == 0 and len(s) < N_SITES:
        s.add(new_site())
    elif op == 1 and len(s) > 1:
        s.remove(int(r.choice(sorted(s))))
    elif op == 2 and 0 < len(s) < N_SITES:
        s.remove(int(r.choice(sorted(s)))); s.add(new_site())
    return np.array(sorted(s))


def sensitivity(pop, C):
    """Per-site sensitivity: population mean of (newly covered demand per unit cost)."""
    covered = np.array([C[:, s].any(axis=1) for s in pop])   # pop x demand
    gain = (~covered).astype(float) @ C.astype(float)        # pop x sites
    return (gain / costs).mean(axis=0) + 1e-9


def run_ga(method, seed):
    """Variable-length NSGA-II-style GA; method='improved' biases add/swap by sensitivity."""
    r = np.random.default_rng(seed)
    pop = [random_sol(r) for _ in range(POP)]
    F = np.array([objectives(s, C_TR) for s in pop])
    for _ in range(GENS):
        rank, crowd = rank_crowd(F)

        def tournament():  # binary tournament on (rank, crowding)
            i, j = r.integers(POP, size=2)
            return i if (rank[i], -crowd[i]) < (rank[j], -crowd[j]) else j

        w = sensitivity(pop, C_TR) if method == "improved" else None
        kids = []
        for _ in range(POP):
            child = crossover(pop[tournament()], pop[tournament()], r)
            if r.random() < P_MUT:
                child = mutate(child, r, w)
            kids.append(child)
        allp = pop + kids
        allF = np.vstack([F, [objectives(k, C_TR) for k in kids]])
        rank, crowd = rank_crowd(allF)          # environmental selection
        keep = np.lexsort((-crowd, rank))[:POP]
        pop, F = [allp[i] for i in keep], allF[keep]
    return pop, F


def top_sites(sol, C):
    """Sites ranked by uniquely covered demand per unit cost (coverage vs cost attribution)."""
    unique_cnt = C[:, sol].sum(axis=1) == 1
    uniq = (unique_cnt[:, None] & C[:, sol]).sum(axis=0)
    order = np.argsort(-uniq / costs[sol])[:3]
    return [(int(sol[k]), int(uniq[k]), round(float(costs[sol[k]]), 2)) for k in order]


# ---------- experiment ----------
b_hv, i_hv, b_dom, i_dom, b_size, i_size, b_sites, i_sites = [], [], [], [], [], [], [], []
fronts_seed = {}
for seed in SEEDS:
    for method in ["baseline", "improved"]:
        pop, F = run_ga(method, seed)
        m = nondom_mask(F)
        # test set used once: evaluate the train-selected non-dominated solutions on test demand
        sel = [s for s, k in zip(pop, m) if k]
        FT = np.array([objectives(s, C_TE) for s in sel])
        fronts_seed[(seed, method)] = FT
        print(f"seed {seed} {method:8s}: front size={len(sel)}, test HV={hypervolume(FT):.4f}")
        if method == "improved" and seed == SEED:
            best = sel[int(np.argmin([objectives(s, C_TR)[0] for s in sel]))]
            print("  top sites (unique coverage, cost) of max-coverage front solution:",
                  top_sites(best, C_TR))
    fb, fi = fronts_seed[(seed, "baseline")], fronts_seed[(seed, "improved")]
    b_hv.append(hypervolume(fb)); i_hv.append(hypervolume(fi))
    b_dom.append(dominated_frac(fb, fi)); i_dom.append(dominated_frac(fi, fb))
    b_size.append(len(fb)); i_size.append(len(fi))

b_hv, i_hv = np.array(b_hv), np.array(i_hv)
metrics = {
    "baseline_score": float(b_hv.mean()), "improved_score": float(i_hv.mean()),
    "improvement": float(i_hv.mean() - b_hv.mean()),
    "baseline_score_std": float(b_hv.std()), "improved_score_std": float(i_hv.std()),
    "baseline_dominated_frac": float(np.mean(b_dom)),   # baseline front points dominated by improved
    "improved_dominated_frac": float(np.mean(i_dom)),   # improved front points dominated by baseline
    "baseline_front_size": float(np.mean(b_size)), "improved_front_size": float(np.mean(i_size)),
}
if not all(np.isfinite(list(metrics.values()))) or min(b_hv.min(), i_hv.min()) <= 0:
    print("ERROR: could not produce meaningful metrics")
    sys.exit(1)

# ---------- figures ----------
os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(4, 4))
plt.bar(["baseline", "improved"], [b_hv.mean(), i_hv.mean()],
        yerr=[b_hv.std(), i_hv.std()], capsize=5, color=["grey", "tab:blue"])
plt.ylabel("test hypervolume (higher = better)"); plt.tight_layout()
plt.savefig("figures/hypervolume_bar.png", dpi=120); plt.close()
plt.figure(figsize=(5, 4))
for method in ["baseline", "improved"]:
    FT = fronts_seed[(SEED, method)]
    o = np.argsort(FT[:, 1])
    plt.plot(FT[o, 1], -FT[o, 0], "o-", label=method, ms=3)
plt.xlabel("normalised cost (test-evaluated front)"); plt.ylabel("test coverage")
plt.legend(); plt.tight_layout(); plt.savefig("figures/fronts_seed42.png", dpi=120); plt.close()

# ---------- results ----------
results = {
    "question": "Can a sensitivity-guided variable-length genetic search with per-site coverage and cost "
                "attributions find a better coverage-cost trade-off front than a standard variable-length "
                "multiobjective GA on a simulated transmitter placement problem?",
    "method": "Synthetic transmitter placement; both GAs optimise coverage (train demand) vs cost; "
              "fronts are re-evaluated on held-out demand points; hypervolume compared over 3 seeds.",
    "dataset": {
        "name": "synthetic transmitter placement (400 candidate sites, 1000 demand points, 20x20 grid)",
        "source_url": "none: generated in-script with numpy (seed 42)",
        "n_rows": N_DEMAND, "n_train": int(len(C_TR)), "n_test": int(n_test),
        "same_as_paper": False, "baseline_is_papers_method": False,
    },
    "metric_name": "test-set hypervolume of normalised coverage-cost front",
    "higher_is_better": True,
    "baseline": "Standard variable-length NSGA-II-style GA (add/remove/swap mutation, one-point "
                "crossover on site lists); not a verified re-implementation of the paper's operators.",
    "improved": "Same GA, population, generations and evaluation budget, plus a population-based site "
                "sensitivity (coverage gain per unit cost) that biases add and swap mutations; reports "
                "per-site coverage vs cost attributions.",
    "metrics": metrics,
    "notes": "Synthetic instance, so results do not transfer to real deployments. Demand points are split "
             "once (75/25) for training and held-out test; test used only for final evaluation. GA settings "
             "were fixed a priori, not tuned, so no CV-based selection was needed. Only 3 seeds: std is "
             "a rough indication. Dominated fractions are computed on test objectives. Stratification "
             "does not apply (no classification labels).",
}
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("baseline_score", metrics["baseline_score"], "improved_score", metrics["improved_score"])
print("wrote results.json")

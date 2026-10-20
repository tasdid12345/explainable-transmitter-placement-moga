# Sensitivity-guided variable-length GA for transmitter placement

Motivation: Wireless heterogeneous transmitter placement is a multiobjective problem in which the number and location of transmitters must be chosen to trade off coverage against cost. The paper "Wireless heterogeneous transmitter placement using multiobjective variable-length genetic algorithm" (2009) addresses this with a variable-length multiobjective GA, where each individual is a list of sites of varying length. This experiment asks whether a site-level sensitivity signal can make such a search find better fronts.

Question: Can a sensitivity-guided variable-length genetic search with per-site coverage and cost attributions find a better coverage-cost trade-off front than a standard variable-length multiobjective GA on a simulated transmitter placement problem?

Dataset: a synthetic instance generated in-script with numpy (fixed seed): 400 candidate sites on a 20x20 grid, 1000 random demand points, and per-site costs. Coverage holds within a fixed radius. This is not the paper's own dataset.

Baseline: a standard variable-length NSGA-II-style GA with add, remove and swap mutation and one-point crossover on site lists. It is not a verified re-implementation of the paper's operators.

Improved: the same GA, population, generations and evaluation budget, plus a site sensitivity estimate (marginal coverage gain per unit cost, averaged over the population). This estimate biases add and swap mutations and reports per-site coverage-versus-cost attributions.

How to run:

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to results.json (plus PNG charts in figures/).

## Results (latest run)

On a synthetic transmitter placement dataset (400 sites, 1000 demand points), a standard variable-length multiobjective genetic search reached 0.944 test-set hypervolume of the normalised coverage-cost front, and my sensitivity-guided variant reached 0.958, a small gain within seed noise over 3 seeds.

| Metric | Value |
|---|---|
| baseline_score | 0.944 |
| improved_score | 0.9578 |
| improvement | 0.01375 |
| baseline_score_std | 0.01085 |
| improved_score_std | 0.008945 |
| baseline_dominated_frac | 0.7667 |
| improved_dominated_frac | 0.1278 |
| baseline_front_size | 60 |
| improved_front_size | 60 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Synthetic instance, so results do not transfer to real deployments. Demand points are split once (75/25) for training and held-out test; test used only for final evaluation. GA settings were fixed a priori, not tuned, so no CV-based selection was needed. Only 3 seeds: std is a rough indication. Dominated fractions are computed on test objectives. Stratification does not apply (no classification labels).

## Reference

This project is motivated by the following paper:

> CK Ting, CN Lee, HC Chang (2009). *Wireless heterogeneous transmitter placement using multiobjective variable-length genetic algorithm*. IEEE Transactions on. https://ieeexplore.ieee.org/abstract/document/4812006/

Paper author: Prof. Chung-Nan Lee, National Sun Yat-sen University (cnlee@cse.nsysu.edu.tw)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

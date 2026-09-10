# Runs still needed before numbers go into the paper

The existing runs (ovarian; sweep2 = 6 kappa x 3 seeds; reference_best; calibration rounds
1-4; graphclust control; ego-masking arms) are the starting point only. Everything below is
what the paper needs for error bars and for the second section. Priority order.

## A. Ovarian section, curated labels (K = 18)

1. **Kappa sweep, 5+ seeds per kappa** (kappa in {0, .05, .1, .2, .3, .4}), operating point,
   with the full evaluation battery on best.pt: held-out recon (global + strata), mirror R2,
   NMI, ridge AND MLP probe dCE with noise floor, KL_w per dim, cycle R2 (pooled + by type +
   depth baseline + linear reference), B stability (across seeds, along kappa), per-type ||w||,
   Moran's I per dim, niche AUC (K = 6/10/15), landmarks (rewritten definitions), pseudotime.
   -> if a third (test) split is adopted (issue 1), do it before this sweep.
2. **Calibration scans with 3 seeds**: alpha_a (closed form {0, .01, .02, .03, .04, .05, .1}),
   alpha_w {.03, .06, .1, .2}, omega {.5, 1, 2}, adversary alpha_a {0.1, 0.3, 0.5, 1.0} at
   adv_steps 6; the convergence-length closed-form check; round-3 placebo (adv_steps 2).
3. **Phi ablation at the operating point, 3 seeds**: full 384-d vs zeros vs 64-d projection
   (settles the design's projection question with error bars).
4. **alpha_z scan** {0.004, 0.007, 0.02} at the operating point, 3 seeds.
5. **Lower-alpha_w arm** (0.06) at the operating point, 3 seeds: shows a non-degenerate w
   posterior for the "w at the operating point" section.
6. **Label-set control** (graphclust K = 26), 3 seeds, full battery (1 run exists).
7. **Marker-set kappa ceiling** (atlas-defined marker set): a computation, not a fit.
8. **Edge-feature comparison** (attention with [d_ij, face_ij]) at the operating point, 3 seeds,
   with corr(alpha_ij, beta_ij) monitored (design's fallback test; never run).
9. **Sweep validation companion** re-run under the rewritten landmark definitions.
10. **Pseudotime gene-axis analysis** re-run on the canonical reference fits (currently on a
    superseded run) — only run-relative statements will be made.
11. **Ego-masking class test** with 3 seeds of the spatial split (error bars on the AUCs).

21. **Degeneracy diagnostics** I(z;t)/H(t), within-type var(mu_z), and held-out recon gap
    z vs one-hot t, on every canonical fit (post hoc, cheap).
22. **Held-out buffer**: quantify the fraction of training seeds whose rings contain held-out
    cells; if non-negligible, adopt a one-hop buffer and re-split BEFORE the sweep reruns.
23. **d_w sensitivity** {4, 6, 8} at the operating point, 3 seeds.
24. **Marginal-invariance ablation** (invariance not conditioned on t) at the operating point,
    1-3 seeds: the CSVAE failure mode reproduced on this data (backs 2.5 without leaning on
    the DisCoVR table).
25. **z-query ablation** (embed(t) replaced by z_i as the attention query), 3 seeds: backs the
    mirror-attractor argument with a measurement (within-type R2, KL_w).

26. **Loadings readouts**: centre every column of B over genes before any top-gene listing or
    B-stability computation (softmax non-identifiability); re-derive existing B figures.
27. **Halo overhead at 4,096-cell tiles**: one measurement (the paper quotes 1.6k and 6.4k).

## B. Synthetic

12. Recovery gate at planted kappa in {0, 0.2}, 10 seeds.
13. Recovery as a function of *assumed* kappa (mis-specified vs planted): the synthetic
    analogue of the sweep, showing what stability across the grid looks like when truth is
    known.
14. A world with G closer to the panel (e.g. 1,000-5,000 genes) and realistic depth.

## C. Lung section (graphclust labels)

15. Preprocessing: bundle, Voronoi graph, pruning statistics, ego-masked embedding, mask-radius
    check, ego-masking class test (with graphclust labels).
16. Calibration on the lung section (does the ovarian operating point transfer? report the
    probe numbers at the ovarian point first, re-calibrate only if the escalation rule fires).
17. Kappa sweep, 5 seeds per kappa, full battery.
18. Validation analyses (no curated landmarks: niche AUC, Moran, pseudotime; landmarks only if
    a structural cluster can be identified from markers).

## D. Baselines (not planned by the author; listed because reviewers will ask)

19. Plain conditional VAE (no w, no leakage) and "c straight into the decoder" (NCEM-like) on
    the same readouts.
20. resolVI on the same section, comparing the contamination estimate to the sweep.

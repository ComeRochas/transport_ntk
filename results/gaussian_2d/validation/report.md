# Scientific validation of the Wasserstein-projected transport NTK

This study separates a change of latent representation, the geometric correction in the natural benchmark, and the effect of the Gaussian-process selector on test error. All training uses the released 2D Gaussian task. The preceding 30-seed comparison is context for this follow-up. Its seeds overlap with this campaign, so the new active-learning contrasts are sensitivity estimates on the same benchmark. [Paper, Lemma 2 and Appendix B.3.2](https://arxiv.org/pdf/2610.35652) · [Released code](https://github.com/biologicalML/transport_ntk/tree/fcd580d079b99dd3ed97e76b1f2753af3afd3463)

## Construction and audit

For the fitted law $\mathcal N(b,C)$ with $C=MM^\top$, a retained parameter direction $a$ induces $C_a=M_aM^\top+MM_a^\top$. The symmetric matrix $S_a$ solves $S_a C+C S_a=C_a$. The field $b_a+S_a(y-b)$ is the $L^2(\mathcal N(b,C))$ projection onto gradients. Subtracting the residual leaves the continuity equation unchanged since $(M_aM^{-1}-S_a)C$ is skew-symmetric. The Gram features are $(b_a,M_a)$ for tNTK, $(b_a,S_aM)$ for the projected pullback and $(b_a,S_a C^{1/2})$ for the canonical projection. Both projected diagonals equal the trace of the local $W_2$ metric over the retained directions. The off-diagonal Gram depends on the chosen identification across conditions. [Paper, Equations 13 and 16 and Appendix B.3.2](https://arxiv.org/pdf/2610.35652)

The two mean outputs contribute $2(hh'^\top+1)$ to the full kernel. For the shape head, each Cholesky output direction is weighted by its derivative, then its Gram is multiplied elementwise by $gg'^\top+1$. The two $+1$ terms account for last-layer biases. The released **mean-only baseline** uses $2hh'^\top$, with its features from the same jointly Gaussian-KL-trained model. We retain that weights-only baseline and add a separate sensitivity arm that adds $2$ to every Gram entry. [Paper, Appendices D.2–D.3](https://arxiv.org/pdf/2610.35652#page=32) · [Released kernel code](https://github.com/biologicalML/transport_ntk/blob/fcd580d079b99dd3ed97e76b1f2753af3afd3463/train/experiment.py#L307)

| Audit item | Finding |
| --- | --- |
| Implementation correction | The analytic kernels now use the exact derivative of default PyTorch softplus, with slope 1 above its threshold of 20. The sigmoid approximation in the earlier implementation agrees in the float32 linear-branch range because it rounds to 1. A float64 autodiff test covers the boundary and the linear branch. |
| Finite differences | Centered differences along a nonlinear head-parameter path, evaluated in float64 with an independent Procrustes formula, converge at second order. The maximum relative error across both projected variants falls from $4.85\times10^{-5}$ at step $10^{-2}$ to $4.83\times10^{-11}$ at $10^{-5}$. The tests also verify the affine Gaussian projection, an independent 2D vertical-energy formula and explicit head Jacobians. |
| Benchmark fidelity | The released target mean differs from Eq. (151). The historical selector conditions on acquired labels with $\lambda=0.001$ and evaluates hypothetical updates with noise 1. We use both released choices for the historical arm. The new coherent setting uses $0.001$ in both updates. Direct Schur-complement reconditioning verifies both selectors. |
| Evaluation | The released float32 Bures evaluator can give values of order $10^{-6}$ below zero on identical individual Gaussians. On the replayed seed-0 tNTK trajectory, its test-set means agree with double-precision Procrustes to $8.2\times10^{-8}$. The benchmark retains the released evaluator for numerical comparability. |
| Pairing | All arms share the test grid, five initial labels and round-wise starting weights for each seed. The full run exactly reproduces acquisitions and loss components in all 480 matched rows of the preceding CPU campaign. It reuses fits only for identical ordered labels and round-wise RNG states. |

## A. Condition-dependent rotation of the latent coordinates

**Hypothesis.** Under a fixed orthogonal map $Q(x)$ at each condition, the induced law and projected pointwise metric are invariant. The canonical Gram and acquisitions should be invariant, too. With the same latent shared across conditions, tNTK and the projected pullback can change off diagonal.

Twelve seed-0-to-11 models were independently trained at round 0 with the released configuration. For each, we replaced $M$ by $MQ$ and the true derivative $M_a$ by $M_aQ$, since $Q$ is fixed with respect to parameters. The transformed directions enter all three Gram calculations. An autodiff test differentiates the full transformed model. The controls introduce a global angle $\pi/3$, and local angles $\alpha(x_1-x_2+x_1x_2)$ with $\alpha\in\{\pi/4,\pi,2\pi\}$. Kernels and selection scores are evaluated in float64.

| Local angle amplitude $\pi$ | Mean relative Frobenius change | Mean first-batch overlap |
| --- | ---: | ---: |
| tNTK | 37.41% | 0% |
| Projected pullback | 29.09% | 4.17% |
| Projected canonical | maximum $1.49\times10^{-16}$ | 100% |

Across all rotations, the largest relative covariance change is $1.38\times10^{-16}$. The largest relative pointwise Gram-diagonal change is $1.56\times10^{-15}$. All 12 canonical batches match in order at every amplitude. Under the global rotation, all three Grams and batches are invariant at numerical precision. At local amplitude $2\pi$, overlap is zero for both tNTK and projected pullback. These results isolate sensitivity to latent alignment across inputs at fixed laws and diagonal scale. [Figure](experiment_a/gauge_invariance.png) · [Per-seed metrics](experiment_a/per_seed.csv) · [Full summary](experiment_a/summary.json)

## B. Non-tangent energy in the natural benchmark

**Hypothesis.** If acquisition disagreement is driven by high pointwise energy removed by projection, raw-only candidates should tend to have larger $r(x)=1-\operatorname{Tr}(\widetilde G_x)/\operatorname{Tr}(G_x)$ than projected-only candidates at an identical checkpoint. Restoring the raw diagonal to projected kernels should then restore some acquisition agreement, conditional on the projected correlation matrix.

We compute $r$ on the complete 196-point pool for every fitted checkpoint of the original tNTK trajectory. We also report the covariance-only fraction and ratios of summed energies. For each checkpoint the original selector compares alternative kernels using the same labels and eligible candidates. The three original histories are saved separately. The primary descriptive trajectory is the original tNTK history, over acquisition rounds 0 through 4. The round-5 selection is a hypothetical query after the final evaluation. Spatial adjustments and the projected histories are exploratory. As a control we replace $K_{\rm proj}$ by $DK_{\rm proj}D$, where $D_{ii}=(K_{{\rm raw},ii}/K_{{\rm proj},ii})^{1/2}$. This preserves projected correlations and matches the raw diagonal.

The mean full-pool removal is **14.41%** (95% CI [13.86%, 15.03%]) across rounds 0–5. The average checkpoint P10/P50/P90 is 1.23% / 14.50% / 27.19%. The covariance-only removal averages **31.84%** (95% CI [31.11%, 32.58%]). The ratio of summed energies is 10.87%, so the arithmetic mean of the ratios gives a somewhat larger correction. The full-pool mean goes from 10.08% initially to 14.34% at the last checkpoint, an increase of 4.26 percentage points [2.85, 5.69] with a peak of 18.37% at round 1.

| Same-checkpoint comparison over rounds 0–4 | Pullback | Canonical |
| --- | ---: | ---: |
| Batch overlap with tNTK | 8.75% | 9.38% |
| Overlap after matching the tNTK diagonal | 8.13% | 7.50% |
| Removed-energy excess, tNTK-only minus projected-only, percentage points (95% CI) | +2.54 [1.35, 3.92] | +3.01 [1.72, 4.32] |
| Same excess after additive $x_1,x_2$ adjustment | +0.76 [−0.21, 1.73] | +1.22 [0.25, 2.19] |

The association weakens after spatial adjustment and varies with the fitted history. On tNTK checkpoints, tNTK-only queries have mean $x_2$ lower by 0.137 and 0.088 relative to pullback-only and canonical-only queries, respectively. The raw-diagonal control does little to restore overlap. The two projected kernels already share the same diagonal and overlap with each other on only 33.13% of acquisitions (95% CI [26.25%, 40.00%]) on the tNTK history. Pointwise energy removal is substantial. Cross-condition correlations also matter for which points the selector queries. These checkpoint comparisons address acquisition choices. The end-to-end loss comparison is below. [Energy by round](experiment_b/energy_dynamics.png) · [Acquisition contrasts](experiment_b/energy_contrasts.png) · [Spatial maps](experiment_b/energy_acquisitions.png) · [Checkpoint CSV](experiment_b/checkpoint_metrics.csv) · [Raw pool data](experiment_b/checkpoints.csv.gz) · [Summary](experiment_b/summary.json)

## C. Scale and GP regularization

**Primary contrast fixed before inspecting the new AL outcomes.** Final test $W_2^2$ for the projected pullback minus tNTK under the released protocol. Secondary planned contrasts examine how that difference changes under (i) normalization by the mean diagonal, (ii) coherent noise of $0.001$, and (iii) both interventions. The canonical kernel, mean-head bias and component or trajectory endpoints are exploratory. A negative difference favors projection. We interpret a gain as uncertain when the paired interval includes zero. A robust advantage across protocols would motivate evaluation on new seeds and then in $d=5$.

The campaign uses seeds 0–15, 196 candidate conditions, 100 test conditions, five fixed initial labels and five batches of two. Each fit uses the released two width-64 heads and 2,000 Adam steps. The four main kernel arms are rerun under all four protocols. At each checkpoint, normalization uses the mean diagonal over all 196 points. The bias sensitivity arm uses the released selector and the random trajectory is shared. We report final $W_2^2$ and its mean/covariance components, mean test loss over acquisition rounds 1–5, overlap of actual acquisitions and their locations. Percentile 95% intervals use 10,000 paired seed bootstrap draws and are unadjusted. The experimental unit is the seed, with one fixed target field and split.

The one-step trace reduction from querying $j$ is $\|K_{\rm post}[:,j]\|_2^2/(K_{{\rm post},jj}+\lambda_{\rm hypothetical})$. Scaling the kernel by $c$ at fixed noise has the same rankings and updates, up to the common scale, as dividing both noise terms by $c$ for the unscaled kernel. This is why we cross the two interventions. [Paper, Equations 17–18](https://arxiv.org/pdf/2610.35652#page=6)

| GP protocol | Final tNTK $W_2^2$ | Pullback minus tNTK (95% CI) | Canonical minus tNTK (95% CI) |
| --- | ---: | ---: | ---: |
| Original | 0.13874 | +0.00115 [−0.01838, +0.02425] | −0.00791 [−0.02556, +0.01063] |
| Unit mean diagonal | 0.20770 | +0.02357 [−0.00945, +0.05564] | −0.00246 [−0.04407, +0.04306] |
| Coherent noise | 0.10322 | +0.03170 [+0.00595, +0.06017] | +0.04547 [+0.01395, +0.09314] |
| Both | 0.09556 | +0.01505 [+0.00549, +0.02493] | +0.01135 [+0.00194, +0.02215] |

| Original selector | Final total | Final mean component | Final covariance component | Mean test loss, rounds 1–5 |
| --- | ---: | ---: | ---: | ---: |
| tNTK | 0.13874 | 0.01519 | 0.12355 | 0.19109 |
| Pullback | 0.13988 | 0.01454 | 0.12535 | 0.19610 |
| Canonical | 0.13082 | 0.01483 | 0.11599 | 0.18851 |
| Mean NTK | 0.22328 | 0.02091 | 0.20237 | 0.26607 |
| Mean NTK with bias | 0.21763 | 0.01835 | 0.19928 | 0.27410 |
| Random | 0.15420 | 0.01876 | 0.13544 | 0.26054 |

The primary interval includes zero. With coherent noise, both projections have higher final and mean-across-round error than tNTK, with unadjusted paired intervals above zero in this sample. The exploratory canonical contrast shifts by +0.05338 [0.01860, 0.10141] when switching from the original selector to coherent noise at the original scale. The corresponding pullback shift is +0.03055 [−0.00259, 0.06694]. Normalizing alone raises the observed mean-across-round loss for both projected variants relative to tNTK, by 0.06336 and 0.05660, respectively. Adding the bias to the mean NTK gives a final contrast of −0.00565 [−0.03578, +0.02092] relative to weights only.

The actual selected sets also change. Under the original selector, pullback and canonical share only 13.13% and 14.38% of the ten acquired points with tNTK, respectively. The share of queries in the right third of the pool ($x_1\ge2/3$) is 43.13% for tNTK and 38.75% for each projection. Changing the tNTK selector to coherent noise retains 9.38% of its acquired points. The full data include the coordinates and component losses for every run. The 16-seed 2D campaign took 330 s on an arm64 CPU with seven workers and one PyTorch thread per worker, using 1,347 distinct fits. [Paired contrasts](active_learning/paired_contrasts.png) · [Learning curves](active_learning/learning_curves.png) · [Acquisition maps](active_learning/spatial_acquisitions.png) · [Seed data](active_learning/active_learning.csv) · [Pair overlap](active_learning/paired_acquisitions.csv) · [Full summary](active_learning/gp_summary.json) · [Manifest](active_learning/manifest.json)

## Interpretation and priorities

The canonical construction exactly addresses the latent-rotation ambiguity tested in A. B shows that the natural 2D task contains material non-tangent energy and that matching local energies leaves substantial acquisition disagreement. C places the accuracy claim in context. The observed advantage of the canonical kernel with the original selector is uncertain, and the two coherent selectors favor tNTK on these seeds. A local metric correction and a full-pool GP acquisition model impose different requirements on the kernel.

The study uses a fixed field, pool and initial labels, and the 16 seeds overlap with the previous comparison. Paired intervals condition on that setup. The regularization grid is a targeted ablation, with no tuning or multiplicity adjustment. A uses prescribed rotations, and the matched-diagonal intervention in B changes the diagonal at fixed correlation. Both isolate aspects of acquisition without measuring a causal path from the energy ratio to test error. All metrics concern the last layer and an exact-label synthetic task.

The next priority is to choose noise levels on separate design runs and freeze them for confirmation on new seeds and independently generated covariance fields. We would then check whether a projection gain persists across tasks or stems from the choice of cross-condition correlations. The present results do not trigger expansion to the more expensive $d=5$ benchmark.

## Reproduce

From the repository root, use a CPU environment with PyTorch, NumPy, Matplotlib, PyYAML and tqdm. The scripts print their output path or checkpoint counts.

~~~bash
python -m unittest train.test_gaussian_projection train.test_gaussian_scientific_validation eval.test_gaussian_gauge_experiment
python -m eval.gaussian_gauge_experiment --seeds 12 --output-dir ../validation_results/experiment_a
python -m train.gaussian_validation --seeds 16 --jobs 7 --output-dir ../validation_run
python eval/gaussian_energy_analysis.py \
  --input ../validation_run/energy_pool.csv.gz --output-dir ../validation_results/experiment_b
python eval/gaussian_gp_analysis.py \
  --input-dir ../validation_run --output-dir ../validation_results/active_learning
~~~

# What Wasserstein projection changes in Gaussian active learning

The projection gives the exact local Wasserstein metric over the parameter directions used here. The canonical construction also gives a Gram invariant to a condition-dependent rotation of the latent Gaussian. This study distinguishes those geometric properties from performance under a scalar GP selector. It also corrects a material numerical issue in the historical float32 implementation.

## 1. Why projection is well motivated

For a fixed input, Lemma 2 counts the change of the transport at a shared latent sample. A change in how samples are coupled can move those samples without changing their law. Appendix B.3.2 separates the transport velocity into the Wasserstein tangent field and a residual with zero density-weighted divergence. Projection removes the latter. The diagonal of the projected Gram then traces the local squared Wasserstein speed. The original Cholesky construction has its own exact interpretation in the Knothe–Rosenblatt metric. [Paper, Lemma 2 and Appendices B.3.2 and C](https://arxiv.org/pdf/2610.35652)

An acquisition rule can benefit if the removed motion would otherwise draw queries toward changes that observations of the law cannot identify. The diagonal alone leaves cross-condition covariances undetermined. The one-step GP trace reduction uses all of the posterior column $P_{:i}$ through $\lVert P_{:i}\rVert^2/(P_{ii}+\lambda_{\rm h})$. The posterior $P$ also depends on the kernel through every already observed condition. Exact local geometry therefore leaves two empirical questions about the choice of cross-condition covariance and how the GP models observations. [Paper, Equations 17–18](https://arxiv.org/pdf/2610.35652#page=6)

## 2. How to compute the projection

Take $T_\theta(z)=b+Mz$ for $z\sim\mathcal N(0,I)$ and $C=MM^\top\succ0$. For each parameter direction $a$, calculate $C_a=M_aM^\top+MM_a^\top$ and solve $S_a C+CS_a=C_a$ for symmetric $S_a$. In an eigenbasis of $C$, divide the $ij$ entry of the transformed $C_a$ by $\lambda_i+\lambda_j$. The field $b_a+S_a(y-b)$ is a gradient and carries the same probability derivative as the original field, since $(M_aM^{-1}-S_a)C$ is skew-symmetric. Its squared norm is $\lVert b_a\rVert^2+\lVert S_aM\rVert_F^2$. [Wasserstein geometry of Gaussian covariances](https://arxiv.org/abs/1801.09269)

For the pullback Gram, use $(b_a,S_aM)$ as features at each condition and sum their inner products over $a$. For the canonical Gram, use $(b_a,S_aC^{1/2})$. Under $M\mapsto M Q(x)$ with $Q$ orthogonal, the covariance and $S_a$ are invariant. The pullback feature rotates by $Q(x)$ and the canonical feature stays fixed. Tests verify the last-layer biases and exact softplus slopes against autodiff. A centered finite-difference check against double-precision Gaussian $W_2^2$ converges at second order and reaches relative error $4.83\times10^{-11}$. [Detailed validation and rotation experiment](https://github.com/ComeRochas/transport_ntk/blob/4d72632/results/gaussian_2d/validation/report.md)

## 3. What the experiments established

### Numerical reliability

The previous archive saved ordered labels and loss values. We reconstructed and saved the fitted weights by deterministically replaying the recorded histories. All 864 archived loss components and 240 ordered batches match exactly across 16 seeds. The precision audit uses 208 active checkpoints, comprising round zero shared across the three histories and rounds 1–4 on each. On every checkpoint it recomputes four full 196-by-196 Grams, for tNTK, the two projections, and the weights-only mean NTK, in both precisions. Each of the four GP protocols contributes 832 two-query batches. Rank comparisons put each method on the same query prefix. Batch comparisons let both greedy selectors follow their own picks. [Replay provenance](replay_manifest.json) · [Precision statistics](precision/summary.json)

| GP protocol | Float32 first-query changes | Float32 ordered-batch changes | Mean same-prefix Spearman | Worst posterior minimum eigenvalue, float32 / float64 |
| --- | ---: | ---: | ---: | ---: |
| Original | 59 / 832 | 88 / 832 | 0.88436 | −507.68 / −7.48e−13 |
| Unit mean diagonal | 6 / 832 | 16 / 832 | 0.99392 | −0.05417 / −7.33e−15 |
| Coherent | 216 / 832 | 246 / 832 | 0.85883 | −44,617.72 / −7.58e−13 |
| Unit mean diagonal and coherent | 43 / 832 | 88 / 832 | 0.98436 | −0.18599 / −7.24e−15 |

The mean NTK is especially affected. Under unnormalized coherent noise, the changed-batch counts are 55 for tNTK, 12 for the pullback, 14 for the canonical projection, and 165 for the mean NTK, each out of 208. A near tie is defined descriptively as a relative score gap of at most $10^{-4}$. In that protocol 424 of 430 changed same-prefix picks lose more than $10^{-4}$ of the reference winner's score. For example, on one tNTK checkpoint the float32 first query gives up 98.9% of the best float64 trace reduction, despite a 1.32% reference gap between first and second place. The large negative eigenvalues are numerical, and can already occur at the first conditioning step.

The maximum relative Frobenius error from recomputing the Grams in float32 is below $6.8\times10^{-7}$. Keeping those rounded Grams and solving in float64 leaves only 6 of 3,328 ordered batches different from the all-float64 reference. The historical inverse/update formula in native float64 agrees with the reference on every ordered batch. This localizes the main failure to low-precision GP arithmetic around ill-conditioned blocks. The corrected runner computes the Gram natively in float64 and refactors the original labeled block by Cholesky at each greedy step. It uses a triangular solve, explicitly symmetrizes the result, and leaves the two specified noise values unchanged. [Per-step diagnostics](precision/precision_steps.csv.gz)

### Hypothetical noise and kernel structure

We held the fitted models, labels and observed noise $0.001$ fixed. The separate float64 sweep uses nine log-spaced hypothetical noises from $10^{-4}$ to $1$ on 96 shared/tNTK-history checkpoints, including the final checkpoint as a hypothetical query. The sweep reads no test losses. Switching hypothetical noise from $1$ to $0.001$ retains **15.10%, 18.23%, and 17.71%** of raw-kernel batch members for tNTK, pullback and canonical, respectively. First-score Spearman correlations are 0.870, 0.875 and 0.894. The large ranking change at the top remains after normalization, with batch overlaps of 21.35%, 10.94% and 8.85%.

The mean raw diagonals are 95.74, 85.27 and 85.27. Entropy effective ranks of the raw Grams are 1.41, 1.47 and 1.61, and those of the conditioned posterior are 6.45, 4.13 and 3.41. The hypothetical noise enters the candidate-specific denominator. As it grows, scores approach rankings by column energy. As it shrinks, the posterior variance at each candidate carries more weight. For the raw Grams, noise $1$ exceeds the eligible posterior variance for 92.4%, 58.1% and 46.8% of candidates, respectively. For the unit-diagonal-scale Grams this figure is 100% for all three kernels. Rescaling a Gram and both noise values by the same positive constant multiplies all scores by that constant, preserving rankings. All 288 scaling controls preserve the ordered batches. Normalizing the Gram at fixed noise therefore changes both effective noises. It also leaves the correlation matrices different. The pullback and canonical diagonals agree to rounding, and their correlation Grams differ by 8.32% in relative Frobenius norm. Spectra describe how concentrated the posterior is. Candidate rankings also depend on eigenvectors and where variance remains after conditioning. [Noise sweep](noise/noise_sensitivity.png) · [Spectra](noise/gram_spectra.png) · [Cross-kernel agreement](noise/kernel_agreement.png) · [Statistics](noise/summary.json)

### Geometry, training and active-learning performance

The earlier rotation control isolated cross-condition alignment by holding every law and every diagonal fixed. On the corrected original-protocol tNTK history, projection removes 14.41% of total energy and 31.83% of covariance energy on average. At the same corrected checkpoint, tNTK batch overlap with pullback is 8.75% and with canonical is 8.13%. Matching the tNTK diagonal while keeping projected correlations changes these to 7.50% and 6.88%. The two projections themselves share a diagonal and overlap on only 33.75% of queries. Thus removing considerable pointwise energy and restoring the original diagonal both leave room for large correlation-driven changes in acquisition. The rotation control establishes this independently of GP scaling or fit history. [Mechanistic evidence and earlier benchmark](https://github.com/ComeRochas/transport_ntk/blob/4d72632/results/gaussian_2d/validation/report.md) · [Corrected energy diagnostics](corrected_energy/summary.json)

**Corrected 16-seed campaign.** We reran the same four protocols and all existing arms with the fixed selector, on the same target, labels at round zero, training code and seeds. This is a numerical correction of the existing sensitivity analysis. The sweep and Fisher diagnostic did not use these losses or select an acquisition setting. Differences are projected minus tNTK in final test $W_2^2$, with unadjusted paired bootstrap 95% intervals.

| GP protocol | tNTK final | Pullback minus tNTK | Canonical minus tNTK |
| --- | ---: | ---: | ---: |
| Original | 0.13872 | +0.00116 [−0.01838, +0.02425] | −0.00790 [−0.02552, +0.01065] |
| Unit mean diagonal | 0.20770 | +0.02357 [−0.00945, +0.05564] | −0.00246 [−0.04407, +0.04306] |
| Coherent | 0.08959 | +0.02592 [+0.01060, +0.04131] | +0.03603 [+0.01906, +0.05212] |
| Unit mean diagonal and coherent | 0.09617 | +0.01542 [+0.00507, +0.02554] | +0.01169 [+0.00205, +0.02256] |

The prespecified original-protocol pullback contrast still includes zero. Both coherent protocols favor tNTK at the final round and on average across rounds. The fix changes at least one batch in 14 of 16 tNTK trajectories under unnormalized coherent noise, and reduces their mean final loss by 0.01363 [0.00638, 0.02270]. For the weights-only mean NTK, even the original protocol changes 15 of 16 trajectories and reduces mean final loss from 0.22328 to 0.18579. The corrected outcome table supersedes the legacy numerical comparison for these seeds. The intervals describe repeated initializations on one fixed field, with the original target generator from the released code and its fixed split. The data support no robust projection advantage on that field. [Corrected results by seed](corrected/active_learning.csv) · [Paired contrasts](corrected/paired_contrasts.png) · [Numerical effect on complete trajectories](corrected/precision_impact.json)

**Training-alignment hypothesis.** The benchmark fits a Cholesky model by Gaussian KL with Adam. At a matched law, the local KL curvature is the Gaussian Fisher form $b_a^\top C^{-1} b_a+\tfrac12\operatorname{tr}(C^{-1}C_aC^{-1}C_a)$. This differs from both raw Cholesky energy and projected Wasserstein energy. For example, at $C=\lambda I$, the Fisher shape metric is $2/\lambda$ times the projected shape metric. On the 96 shared/tNTK checkpoints, the mean Frobenius cosine between the coordinate shape metric and Fisher is 0.92826 for the raw metric and 0.91964 after projection. The paired difference is +0.00862, 95% interval [0.00749, 0.00974]. Near isotropy the sign reverses. This is narrow, coordinate-dependent evidence about local shape curvature. It leaves the explanation involving actual retraining open, since the benchmark updates the entire network from scratch and the scalar GP uses off-diagonal covariances and a simplified observation model. [KL training, Appendix D.3](https://arxiv.org/pdf/2610.35652#page=33) · [Local alignment statistics](alignment/summary.json)

### Proposed independent check

Freeze one comparison of tNTK and canonical projection on **12 independently drawn 2D Gaussian fields**, with two fresh paired model seeds per field. Fix the 14-by-14 candidate grid, the offset 10-by-10 test grid, five initial conditions and five lots of two, and the released architecture and KL optimizer. Use native float64 GP acquisition, unit mean Gram diagonal and **both noises fixed at 0.01**. The choice represents noise of 1% of the mean prior variance and is outside the existing four-protocol comparison. The canonical arm is chosen for its proven invariance.

Generate five independent smooth Fourier channels per field. Use frequencies $(1,0),(0,1),(1,1),(1,-1),(2,0),(0,2)$ with independent standard normal amplitudes and uniform phases, scaled by $\sqrt{2/6}$. Set mean components to 0.4 times the first two channels, standard deviations to $0.2+0.7\sigma(f_3)$ and $0.2+0.7\sigma(f_4)$, and the covariance rotation angle to $\pi\sigma(f_5)$. Freeze field IDs 4001–4012 and model IDs 5001–5024 before running. Keep training and test points identical within each pair.

The sole primary endpoint is the difference in mean test $W_2^2$ over acquisition rounds 1–5, canonical minus tNTK. Average the two paired starts within each field and bootstrap the 12 fields. Report individual field differences and the two-sided 95% interval. Final and covariance errors are secondary. Freeze the generator, seeds, analysis and selector before inspecting losses, then run all pairs once. This small pilot measures transfer to fresh laws and retains inconclusive results. The campaign has been proposed only.

## Reproduce

Run from the repository root with PyTorch, NumPy, SciPy, Matplotlib, PyYAML and tqdm. The Gaussian command-line runners now default to the stable selector. The previous report records the legacy flags for exact historical reproduction.

~~~bash
python -m eval.gaussian_checkpoint_replay --jobs 6 --output-dir ../checkpoints
python -m eval.gaussian_gp_precision --checkpoint-dir ../checkpoints --output-dir ../precision
python -m eval.gaussian_gp_noise_diagnostics --checkpoint-dir ../checkpoints --output-dir ../noise
python -m eval.gaussian_metric_alignment --checkpoints ../checkpoints --output-dir ../alignment
python -m train.gaussian_validation --seeds 16 --jobs 7 --stable-gp --output-dir ../corrected_run
python eval/gaussian_gp_analysis.py --input-dir ../corrected_run \
  --legacy-input-dir results/gaussian_2d/validation/active_learning --output-dir ../corrected
python eval/gaussian_energy_analysis.py \
  --input ../corrected_run/energy_pool.csv.gz --output-dir ../corrected_energy
~~~

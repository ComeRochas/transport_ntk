# Wasserstein-projected tNTK on the 2D Gaussian benchmark

Appendix B.3.2 decomposes the transport velocity into the Wasserstein tangent velocity and a residual with zero weighted divergence. Lemma 2 gives a Gram of mean and factor derivatives. In a direction `dM`, the raw factor derivative can include a component that only changes how the latent samples are coupled. Appendix C identifies the raw Cholesky quadratic form with the local Knothe–Rosenblatt metric. This experiment compares both kernels on the benchmark's $W_2^2$ loss. [Paper](https://arxiv.org/pdf/2610.35652)

For a Gaussian with covariance `C = M Mᵀ`, a factor direction has covariance derivative `dC = dM Mᵀ + M dMᵀ`. The projected field is `db + A(y − b)`, where the symmetric matrix `A` solves `A C + C A = dC`. The residual induces zero covariance derivative and has zero weighted divergence. In an eigenbasis of `C`, the solver divides each entry of `dC` by the sum of the corresponding eigenvalues.

- `projected` forms a Gram of `(db, A M)`, using the shared latent sample as in the original tNTK.
- `projected_canonical` forms a Gram of `(db, A C^(1/2))`, using the symmetric positive-definite square root. This choice is invariant to right orthogonal rotations of `M` at each condition, including parameter-dependent rotations. Both projected variants have the same pointwise metric and can give different acquisitions.

The implementation solves one direction per Cholesky output and uses the shared last-layer embeddings to account for weights and biases. It also retains the mean term of the original tNTK. The diagonal gives the trace of the local Wasserstein metric, while the scalar GP acquisition uses the entire Gram. A local contraction therefore supplies a hypothesis about acquisition performance.

## Reproduce the comparison

The upstream configuration uses 196 candidate conditions, 100 held-out conditions, five fixed starting conditions and five acquisitions of two conditions each. Each round fits the same pair of width-64 ReLU networks from scratch with 2,000 Adam steps and Gaussian KL. All arms share the seed and starting conditions. The primary contrast is final test $W_2^2$ for the original projected pullback minus tNTK. The mean change from round zero over rounds 1–5 matches the definition in Table S9. The canonical pullback is an exploratory comparison.

```bash
python -m pip install torch numpy matplotlib pyyaml tqdm
python -m unittest train.test_gaussian_projection
python train/al_2d_gaussian.py --configs both_complex_b2 --seeds 30 \
  --strategies transport projected projected_canonical mean random \
  --device cpu --threads 1 --jobs 4 --output-dir ../gaussian_projection_run
python eval/gaussian_projection_summary.py \
  --results ../gaussian_projection_run/both_complex_b2/results.pt \
  --output-dir ../gaussian_projection_run/summary
```

The reproduction targets the implementation of Figure 2. The released generator uses $2\exp(-6x_1)[x_2,1-x_2]$ for the mean, while Equation (151) places the $x_2$-dependent factors inside the exponentials. The released selector uses `lam = 0.001` in conditioning on labeled points and an additive `1` in hypothetical within-batch updates. Both choices are shared across the kernel-based arms. [Upstream benchmark](https://github.com/biologicalML/transport_ntk/blob/fcd580d079b99dd3ed97e76b1f2753af3afd3463/train/experiment.py)

## Follow-up experiments

1. **Rotation sensitivity.** Introduce a condition-dependent right rotation of the factor with a separate parameter and a predeclared amplitude sweep. The output law is identical for every amplitude. Check projected Gram and first-batch invariance, then measure additional label cost or loss for the original kernel as the nuisance gradient grows. A one-query version is covered by the tests.
2. **Acquisition scale and Gaussian dimension.** Match mean kernel diagonals and use one common regularization for observed and hypothetical updates as separate ablations. Compare paired final $W_2^2$, Bures covariance loss and runtime. Then reconstruct the appendix E.2 experiment at $d=5$, with 15 initial conditions and five batches of four, before increasing dimension. Record how much local metric energy the projection removes and whether that predicts a gain.
3. **Non-affine transport.** On the synthetic radial-flow experiment, approximate the $L^2(\mu_x)$ gradient projection by regressing endpoint transport Jacobians onto gradients of a fixed potential basis. Validate the approximation on known affine fields first. Use the paper's Sinkhorn metric and report fitting and acquisition costs. [Paper, appendices B.3.2, D.5 and E.2](https://arxiv.org/pdf/2610.35652)

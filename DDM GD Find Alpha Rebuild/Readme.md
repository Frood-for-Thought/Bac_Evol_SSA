# D-DEME: Dynamic Data Evolving Mean Estimator

*A readme for the codebase and algorithmic logic*

## What D-DEME does (in one breath)

D-DEME tunes a scalar control parameter, α, so that the **ensemble mean** of a noisy black-box output \$v\_j(α)\$ matches a desired target \$v\_d\$. It never needs the explicit form of \$v(α)\$. Instead, at each α it draws a large batch of samples, treats that batch as a **macroscopic** object (by the CLT), logs the **mean** and **variance**, estimates local **slopes** of the mean from data, and takes guarded gradient-like steps on a simple loss.

---

## The loss and the (data-driven) gradient

D-DEME optimizes the CLT-friendly loss

$$
L(\alpha) = \big(\mu(\alpha) - v_d\big)^2 + \lambda\, s^2(\alpha),
$$

where \$\mu(α)\$ and \$s^2(α)\$ are the **batch** mean and variance at α. The optional variance penalty \$λ≥0\$ trades off accuracy vs. stability. The code computes

$$
\frac{dL}{d\alpha}
= 2\bigl(\mu(\alpha)-v_d\bigr)\frac{d\mu}{d\alpha}
\+\ \lambda\frac{d s^2}{d\alpha}
$$

using finite differences for the derivatives.

Why this loss? With large i.i.d. batches, \$\hat\mu(α)\$ concentrates around the population mean and the MSE of \$\hat\mu\$ decays like \$σ^2/n\$; the document “Loss Function Derivative” develops this CLT-level view and shows when variance terms help or hinder learning, and how local linearity enables stable slope use. You don’t need every derivative from that note, but its key identities justify the form above and the second-order “variance-aware” tweaks.

---

## How the modules fit together

```
sample_v_func_NU()  →  MacroStats.macro_observations()  →  FiniteDifferenceTracker
         │                       │                                   │
         └─────────────── run_sampling_pipeline.main() orchestrates ──┴──→ LossEvaluator
```

* **`run_sampling_pipeline.py`** orchestrates everything:

  1. sweeps an α-grid to log macro stats; 2) computes finite differences;
  2. finds **sign-change brackets** where \$\mu(α)-v\_d\$ crosses zero;
  3. iterates updates of α using guarded step sizes. It also normalizes step-size by the local slope and applies a **γ-cap** derived from linear stability analysis (details below).

* **`macro_stats.py`** ingests a batch of samples at a given α and stores a record with:
  $\mu$, $s^2$, $s$, `stderr`, `stderr_ci`, $n$. It also proposes a **small, safe probe step** \$h\$ when needed (e.g., to escape a false plateau).

* **`finite_difference_tracker.py`** computes forward differences for $\mu$, $s^2$, and (optionally) a centered Polyak–Ruppert (PR) slope $m_k$ of $\mu$ vs. $\alpha$ over a window $[\alpha_{\min}\, \alpha_{\max}]$. It keeps a “linear-slope-ready” flag once $m_k$ stabilizes and exposes a centered-residual gate to sanity-check local linearity: 
$r_{\text{cent}}(\alpha) = \bigl(\mu(\alpha) - \bar{\mu}\bigr) - m_k\,\bigl(\alpha - \bar{\alpha}\bigr)$ 
should be no larger than a stderr-based bound.

* **`landscape_analysis.py`** scans the recorded $\alpha,\, \mu(\alpha)$ to detect sign transitions of $\mu(\alpha)-v_d$; these yield a bracket $[\alpha_{\min}\, \alpha_{\max}]$ around a root of $\mu(\alpha)-v_d=0$, which initializes the optimizer near a valid solution.

* **`loss_evaluator.py`** is the “optimizer brain.” It:

  * evaluates \$L(α)\$ and \$dL/dα\$ from the records and finite differences;
  * **decides the next α** with several safeguards:

    * **slope-normalized step:** use a γ scaled by \$1/|\mu'(α)|\$ so the raw update magnitude doesn’t explode on steep/flat patches;
    * **γ-cap using \$m\_k\$:** if PR slope is ready, enforce \$|1-2γm\_k^2|<1\$ with the best non-oscillatory choice near \$γ≈1/(2m\_k^2)\$;
    * **bracket direction override:** keep steps moving toward/within $[\alpha_{\min}, \alpha_{\max}]$;
    * **probe fallback:** if curvature/slope diagnostics suggest a false plateau, take a small \$h\$ suggested by `MacroStats`.

---

## The training loop, step-by-step

1. **Seed the landscape.** Sample on a coarse α-grid; for each α, call `macro_observations()` to log \$\mu(α), s^2(α)\$ (large n to reduce noise). Then compute all forward differences.

2. **Bracket a solution.** Run `detect_sign_transitions(stats, v_d)` to find the first interval where \$\mu(α)-v\_d\$ changes sign; set $[\alpha_{\min}, \alpha_{\max}]$.

3. **Start PR slope estimation.** Enable `run_linear_estimation(stats, window=[alpha_min, alpha_max])`. Over the window $[\alpha_{\min}, \alpha_{\max}]$ the tracker maintains a centered linear fit $m_k$ of $\mu$ vs. $\alpha$, and stores $\bar{\alpha}$ and $\bar{\mu}$ for the centered-residual test.

5. **Iterate updates.** At each \$α\_k\$:

   * Ensure a fresh macro record exists (and refresh finite differences).
   * Optionally log the **centered-residual gate** status to check local linearity.
   * Update PR slope (no reset), and read `linear_slope_ready`.
   * Compute \$d\mu/dα\$ from finite differences; set the **scale** \$γ = 1/|\mu'(α\_k)|\$.
   * If PR slope is ready, **cap γ** with \$γ \le (1-\eta)/(2m\_k^2)\$ for a small safety margin \$\eta\$.
   * Ask `LossEvaluator.decide_next_alpha(...)` for the next α (it evaluates the gradient step, may apply a direction override to respect the bracket, may probe forward if a plateau is detected, etc.).

---

## The math that motivates the safeguards

### 1) CLT macro-view

At each α we aggregate \$n\$ independent samples; \$\hat\mu(α)\$ is approximately normal with variance \$σ^2(α)/n\$. Using batches means we can **discard** microstates and keep only the macroscopic record \$(\alpha,\mu,s^2)\$ while still getting an accurate signal for optimization.

### 2) Local linear recursion and the γ-cap

Near a solution \$α\_\*\$ where \$\mu(α\_\*)=v\_d\$, write \$\mu(α)\approx \mu(α\_\*) + m\_k(α-α\_\*)\$ with slope \$m\_k\$ from the PR fit. A vanilla gradient step with step-size γ gives the **error recursion**

$$
e_{k+1} \;=\; (1-2γm_k^2)\,e_k,\quad e_k:=α_k-α_\*.
$$

Thus \$|1-2γm\_k^2|<1\$ is the stability condition; the **best non-oscillatory contraction** is at \$γ≈1/(2m\_k^2)\$. The pipeline computes/updates \$m\_k\$ and enforces a small-margin cap on γ accordingly.

### 3) Centered PR slope and linearity gate

To avoid intercept bias in residual tests, the slope is fit **centered**:

$$
m_k=\frac{\sum(\alpha_i-\bar\alpha)(\mu_i-\bar\mu)}{\sum(\alpha_i-\bar\alpha)^2+\varepsilon}.
$$

We then monitor \$|r\_{\text{cent}}(α)|=\big|(\mu(α)-\bar\mu)-m\_k(\alpha-\bar\alpha)\big|\$ and compare it against a stderr-based bound; small centered residuals indicate the local **linearity** required by the γ-cap analysis.

---

## What each file contains (quick reference)

* `run_sampling_pipeline.py`
  *Sampling function*, initial α-sweep, FD computation, sign-change bracketing, PR slope windowing, slope-normalized γ, γ-cap, and the training loop.

* `macro_stats.py`
  Records $\alpha$, $\mu$, $s^2$, $s$, $\mathrm{stderr}$, $n$, and confidence bounds; suggests cautious probe sizes to escape plateaus.

* `finite_difference_tracker.py`
  Forward differences for \$\mu\$ and \$s^2\$; centered PR slope \$m\_k\$ with stabilization checks; centered-residual gate and window means \$(\bar\alpha,\bar\mu)\$.

* `landscape_analysis.py`
  Utility to detect intervals where \$\mu(α)-v\_d\$ changes sign; provides a bracket to start iterations safely.

* `loss_evaluator.py`
  Implements \$L(α)\$, \$dL/dα\$, and **`decide_next_alpha`** which combines gradient steps, bracket-direction overrides, γ-cap logic (via tracker), and probe fallbacks using `MacroStats`.

* Notes / manuscript
  Conceptual framing of D-DEME as CLT-driven, memoryless batch optimization; links to Robbins–Monro lineage while enabling slope-aware step-sizes and variance-aware penalties.

---

## Why this may be new (and useful)

* **Macro first, micro never:** It explicitly leans on CLT so we can **throw away** raw samples each iteration while still controlling noise and taking principled steps on \$L(α)\$.
* **Slope-normalized stepping + γ-cap:** Normalizing by \$|\mu'(α)|\$ keeps step magnitudes sane on both steep and flat regions; the PR-slope γ-cap yields the same linear-systems stability guarantees practitioners expect, but **from data** rather than model gradients.
* **Variance-aware objective:** Penalizing \$s^2(α)\$ directly trades off accuracy vs. robustness, which is natural when the black-box’s variability itself depends on α.
* **Bracketed search on learned landscapes:** Sign-change detection on \$\mu(α)-v\_d\$ gives practical, data-driven initialization near a feasible root, even when \$v(α)\$ is non-monotone.

---

## Minimal usage sketch

```bash
# inside this repo
python run_sampling_pipeline.py
```

What happens:

1. batches are drawn from `sample_v_func_NU`;
2. `MacroStats` logs \$\mu,s^2\$ per α;
3. `FiniteDifferenceTracker` builds finite differences and (optionally) a PR slope over a bracket;
4. `LossEvaluator` steps α using the data-driven gradient with γ-normalization, γ-cap, and bracket safeguards.

---

## Implementation details worth noting

* **Forward finite differences** for both \$\mu\$ and \$s^2\$ are stored as records alongside α; `estimate_derivative_at` looks them up for the loss gradient.
* The **centered residual bound** scales like \$2,s(α)/\sqrt{n}\$ (stderr-style) so the gate tightens automatically with larger batches.
* When the gradient step **worsens** the loss but slope hasn’t collapsed, `LossEvaluator` falls back to a **small forward probe** \$h\$ from `MacroStats.suggest_step_size`.
* The **direction override** keeps updates moving toward/within the bracket $\[α\_\ell,α\_r]\$ to avoid wandering when noise is high.

---

## Extending or swapping pieces

* Replace the sampler with your domain function (e.g., a simulator) as long as it returns a batch of \$v\_j(α)\$.
* Keep `MacroStats` and `FiniteDifferenceTracker` intact to preserve the CLT assumptions and the PR-slope stability logic.
* Adjust \$λ\$ for your accuracy–stability preference; if your noise is α-independent, you can set \$λ=0\$.

---

## TL;DR

D-DEME is a **batch-first**, **slope-aware**, **variance-aware**, and **bracket-safe** way to fit α using only samples of \$v\_j(α)\$. It trades microstate bookkeeping for CLT-level macrostats and guards its steps with data-estimated slopes and simple stability math—so it stays practical even when the function is unknown, noisy, and non-monotone.

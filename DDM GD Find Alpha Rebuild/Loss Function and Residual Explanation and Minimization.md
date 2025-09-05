# Intrinsic Error, Centered Residual, and the Stderr-Based Gate

## 1. Intrinsic sampling error

Let \(\{v_j(\alpha)\}_{j=1}^n\) be i.i.d. samples drawn at parameter \(\alpha\) from a distribution with **population mean** \(\bar{v}(\alpha)\) and **population variance** \(\sigma^2(\alpha)\). The **sample mean** is
\[
\mu(\alpha) := \frac{1}{n}\sum_{j=1}^n v_j(\alpha).
\]

Define the **intrinsic sampling error** (a random variable) by
\[
\delta_n(\alpha) := \mu(\alpha) - \bar{v}(\alpha).
\]

Under standard conditions (finite variance, i.i.d. sampling), the Central Limit Theorem (CLT) implies
\[
\delta_n(\alpha) = \mathcal{O}_p\!\left(\frac{1}{\sqrt{n}}\right),
\qquad
\mathrm{Var}[\delta_n(\alpha)] \approx \frac{\sigma^2(\alpha)}{n}.
\]
Thus, even when \(\alpha\) is tuned so that \(\bar{v}(\alpha)\) is exactly on target \(v_d\), the observable \(\mu(\alpha)\) fluctuates around \(v_d\) with standard deviation of order \(\sigma(\alpha)/\sqrt{n}\). This is the **irreducible sampling noise** at a given batch size \(n\).

## 2. Centered residual \(r_{\mathrm{cent}}\)

We estimate a **local linear model** for the expectation,
\[
\bar{v}(\alpha) \approx m_k\,\alpha + b
\]
over a bracket/window \(\alpha \in [\alpha_{\min},\alpha_{\max}]\). A direct test of \(|\mu(\alpha) - m_k \alpha|\) is biased by the (unknown) intercept \(b\). To remove this bias, we use the **centered residual**
\[
r_{\mathrm{cent}}(\alpha) := \big(\mu(\alpha) - \bar{\mu}\big) \;-\; m_k \big(\alpha - \bar{\alpha}\big),
\]
where \(\bar{\mu}\) and \(\bar{\alpha}\) are the means of the observed \(\mu\) and \(\alpha\) within the window. If the local model is linear, \(\mu(\alpha) \approx m_k \alpha + b + \delta_n(\alpha)\), then
\[
r_{\mathrm{cent}}(\alpha) \;\approx\; \delta_n(\alpha) - \overline{\delta_n},
\]
i.e., an intercept-free fluctuation with mean zero. Consequently, \(r_{\mathrm{cent}}\) behaves like **pure sampling noise**, making it the right quantity to compare to a stderr-type threshold.

## 3. Stderr-based bound (“gate”)

Let \(s^2(\alpha)\) be the **sample variance** at \(\alpha\). The standard error of the sample mean is \(s(\alpha)/\sqrt{n}\). A simple two-sided “\(\approx\)95%” width is \(2\,s(\alpha)/\sqrt{n}\). Using a tunable multiplier \(\texttt{stderr\_tol} > 0\), we declare **linear-agreement OK** if
\[
\big|\, r_{\mathrm{cent}}(\alpha) \,\big| \;\le\; \texttt{stderr\_tol}\,\frac{2\,s(\alpha)}{\sqrt{n}}.
\]
Interpretation: within this probabilistic tolerance band, the observed mean \(\mu(\alpha)\) is consistent with a locally linear model of slope \(m_k\).

## 4. Intrinsic error in the gradient update

Consider the **variance-penalized loss**
\[
L(\alpha) \;=\; \big(\mu(\alpha) - v_d\big)^2 \;+\; \lambda\cdot \frac{n}{n-1}\, s^2(\alpha).
\]
Its derivative is
\[
\frac{dL}{d\alpha} \;=\; 2\big(\mu(\alpha) - v_d\big)\,\mu'(\alpha) \;+\; \lambda\cdot \frac{n}{n-1}\,\frac{d s^2}{d\alpha},
\]
where \(\mu'(\alpha) = d\mu/d\alpha\). Decomposing the sample mean into population mean plus sampling error,
\[
\mu(\alpha) = \bar{v}(\alpha) + \delta_n(\alpha),
\qquad
\mu'(\alpha) = \bar{v}'(\alpha) + \delta_n'(\alpha),
\]
yields
\[
\frac{dL}{d\alpha} \;=\; 2\big(\bar{v}(\alpha) - v_d\big)\,\bar{v}'(\alpha)
\;+\; \underbrace{2\big(\bar{v}(\alpha) - v_d\big)\,\delta_n'(\alpha) + 2\,\delta_n(\alpha)\,\bar{v}'(\alpha) + 2\,\delta_n(\alpha)\,\delta_n'(\alpha)}_{\text{intrinsic sampling-error terms}}
\;+\; \lambda\cdot \frac{n}{n-1}\,\frac{d s^2}{d\alpha}.
\]
The bracketed terms are **stochastic**, centered around zero with magnitude controlled by \(n\). As \(n\) grows, \(\delta_n = \mathcal{O}_p(n^{-1/2})\) and, under mild smoothness, \(\delta_n' = \mathcal{O}_p(n^{-1/2})\) as well, so these contributions diminish at rate \(n^{-1/2}\).

In a locally linear regime where \(\bar{v}(\alpha) \approx m_k \alpha + b\), we have \(\bar{v}'(\alpha) \approx m_k\). If the **centered residual gate** (Sec. 3) holds, it is also reasonable to approximate \(\mu'(\alpha)\) by a stable finite-difference estimate \(\widehat{\mu'}\) or by \(m_k\) itself in the tight-linear limit.

## 5. Minimizing intrinsic error: learning-rate and batch-size schedules

Even with ideal \(\alpha\), the update is driven by a noisy gradient because of \(\delta_n\). Near the target (where \(\bar{v}(\alpha) - v_d \approx 0\)), the deterministic component shrinks while the stochastic component is still of order \(n^{-1/2}\). To stabilize convergence in this **noise-dominated** regime, we employ two complementary schedules:

**(a) Learning-rate decay.**  
Use an **epoch-wise** or **iteration-block** decay, e.g. after every fixed block of iterations \(i\),
\[
\gamma_{i+1}' \;=\; \frac{\gamma_i'}{i+1},
\]
consistent with classical Robbins–Monro step conditions. This gradually reduces the influence of stochastic perturbations in the update \( \alpha_{k+1} = \alpha_k - \gamma'\, \frac{dL}{d\alpha} \).

**(b) Increasing batch size.**  
Within the same blocks, **increase \(n\)** to reduce \(\mathrm{Std}[\delta_n] \sim \sigma(\alpha)/\sqrt{n}\). For example, doubling \(n\) reduces the standard error by a factor \(1/\sqrt{2}\). This directly tightens the gate in Sec. 3 and shrinks the stochastic terms in \(dL/d\alpha\).

In combination, learning-rate decay and batch-size growth **lower the effective noise floor** while preserving progress toward the optimum.

## 6. Practical computation in our pipeline

1. **Slope tracking (Polyak–Ruppert, centered):** compute \(m_k\) via centered least squares on \((\alpha_i,\mu_i)\) within the working window. Track \(\Delta m = |m_k - m_{k-1}|\) and set `linear_slope_ready = True` when \(\Delta m < \texttt{slope\_tol}\).  
2. **Centered residual:** at each visited \(\alpha\), compute
   \[
   r_{\mathrm{cent}}(\alpha) = \big(\mu(\alpha) - \bar{\mu}\big) - m_k \big(\alpha - \bar{\alpha}\big).
   \]
3. **Gate:** compute the bound \( \text{bound} = \texttt{stderr\_tol} \cdot 2\, s(\alpha)/\sqrt{n} \) and check \(|r_{\mathrm{cent}}(\alpha)| \le \text{bound}\). Log `(resid_ok, r_cent, bound)`; this is a diagnostic that the local linear model is statistically plausible at the current \(n\).  
4. **Gradient step with stability cap:** when `linear_slope_ready` is true, cap the effective step-size using the linear model: require \(|1 - 2\,\gamma\,m_k^2| < 1\), e.g. set \(\gamma \le (1-\eta)/(2 m_k^2)\) with a small safety margin \(\eta>0\).  
5. **Schedules:** apply a learning-rate scheduler (e.g., PyTorch cosine/step) and a batch-size schedule (e.g., double \(n\) every fixed block). This mirrors the document’s prescription and mitigates \(\mathcal{O}(n^{-1/2})\) fluctuations as we approach the target.

## 7. Interpretation

- \(r_{\mathrm{cent}}\) is an **intercept-free residual** that should look like pure noise when the local linear model is valid.  
- The **stderr-based gate** quantifies when observed deviations are statistically compatible with that linear model at the current \(n\).  
- The gradient’s **intrinsic error terms** scale like \(n^{-1/2}\) and do not vanish unless we either increase \(n\) or reduce the learning rate sufficiently.  
- The **stability cap** on \(\gamma\) based on \(m_k\) prevents oscillations/instability even when \(m_k\) is large in magnitude.

## 8. Limitations and caveats

- The gate uses a **heuristic 95%-ish width** \(2\,s/\sqrt{n}\) multiplied by \(\texttt{stderr\_tol}\); it is not a formal hypothesis test.  
- Nonlinearities within the window, heteroskedasticity, or dependence between samples can weaken the CLT intuition. In such cases, tightening the window, increasing \(n\), or using robust regression for \(m_k\) can help.

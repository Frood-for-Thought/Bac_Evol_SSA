Here you go — the exact write-up in plain text so you can paste it into your own `.md` file.

---

# Intrinsic Error, Centered Residual, and Step-Size/Batch-Size Control

## 1. Setup and notation

Let $v_j(\alpha)$ be stochastic observations generated at parameter value $\alpha$, and let

$$
\mu(\alpha)=\frac{1}{n}\sum_{j=1}^n v_j(\alpha)
$$

be the sample mean from $n$ draws. The (unknown) population mean is $\bar v(\alpha) = \mathbb{E}[v_j(\alpha)]$, and the sampling error of the mean is

$$
\delta_n(\alpha)\;:=\;\mu(\alpha)-\bar v(\alpha).
$$

Under standard regularity, $\delta_n(\alpha)=\mathcal{O}_p(n^{-1/2})$. The sample variance and standard deviation are

$$
s^2(\alpha)=\frac{1}{n-1}\sum_{j=1}^n\bigl(v_j(\alpha)-\mu(\alpha)\bigr)^2,
\qquad
s(\alpha)=\sqrt{s^2(\alpha)}.
$$

We also maintain a *local linear* approximation of the mean in a bracket $[\alpha_{\min},\alpha_{\max}]$ via a centered least-squares (Polyak–Ruppert style) slope $m_k$ computed on the window:

$$
m_k \;=\;\frac{\sum_i\bigl(\alpha_i-\bar\alpha\bigr)\bigl(\mu(\alpha_i)-\bar\mu\bigr)}{\sum_i\bigl(\alpha_i-\bar\alpha\bigr)^2+\varepsilon},
\qquad
\bar\alpha:=\frac{1}{K}\sum_i \alpha_i,\quad \bar\mu:=\frac{1}{K}\sum_i \mu(\alpha_i).
$$

This *centered* regression removes intercept bias and makes $m_k$ a consistent estimator of the local slope $d\bar v/d\alpha$ when the window is locally linear.

---

## 2. Intrinsic error of the sample mean

The sampling error $\delta_n(\alpha)=\mu(\alpha)-\bar v(\alpha)$ satisfies

$$
\mathbb{E}[\delta_n(\alpha)]=0,
\qquad 
\mathrm{Var}\bigl(\delta_n(\alpha)\bigr)=\frac{\sigma^2(\alpha)}{n},
$$

with $\sigma^2(\alpha)=\mathrm{Var}(v_j(\alpha))$. By the CLT, $\delta_n(\alpha)$ is approximately normal with standard deviation $\sigma(\alpha)/\sqrt{n}$. In practice we estimate $\sigma(\alpha)$ by $s(\alpha)$, so a two-sided $\approx95\%$ noise band for the *mean* is $\pm 2\,s(\alpha)/\sqrt{n}$. This is the intrinsic uncertainty that persists even when the optimization is near the target $\bar v(\alpha)\approx v_d$.

---

## 3. Centered residual and its bound

A naive residual $|\mu(\alpha)-m_k\alpha|$ is biased when the local line has a nonzero intercept. The correct (intercept-free) diagnostic is the *centered residual*

$$
r_{\mathrm{cent}}(\alpha)\;:=\;\bigl(\mu(\alpha)-\bar\mu\bigr)\;-\;m_k\,\bigl(\alpha-\bar\alpha\bigr).
$$

If $\mu(\alpha)\approx m_k\,\alpha+b$ on the window, then $r_{\mathrm{cent}}(\alpha)\approx$ (sampling noise only). Hence it is appropriate to compare $|r_{\mathrm{cent}}(\alpha)|$ to a standard-error-type bound. We declare “linear-agreement OK” when

$$
\boxed{\;\;|r_{\mathrm{cent}}(\alpha)|\;\le\;\texttt{stderr\_tol}\;\cdot\;\frac{2\,s(\alpha)}{\sqrt{n}}\;\;}
$$

where $\texttt{stderr\_tol}$ is a small multiplier (e.g. $2$). This gate is used diagnostically to confirm that the local window behaves linearly and that the PR slope $m_k$ is meaningful for step-size control.

---

## 4. Linearized gradient and the role of $m_k$

For the variance-penalized loss $L(\alpha)=(\mu(\alpha)-v_d)^2+\lambda \, s^2(\alpha)$, the sample gradient is

$$
\frac{dL}{d\alpha}
=2\bigl(\mu(\alpha)-v_d\bigr)\,\mu'(\alpha)+\lambda\,\frac{d s^2(\alpha)}{d\alpha}.
$$

Decomposing the mean and its derivative,

$$
\mu(\alpha)=\bar v(\alpha)+\delta_n(\alpha),
\qquad
\mu'(\alpha)=\bar v'(\alpha)+\delta'_n(\alpha),
$$

we obtain

$$
\frac{dL}{d\alpha}
=2\bigl(\bar v(\alpha)-v_d\bigr)\bar v'(\alpha)+\lambda\,\frac{d s^2}{d\alpha}
\;+\;2\,\delta_n(\alpha)\,\bar v'(\alpha)
\;+\;2\bigl(\bar v(\alpha)-v_d\bigr)\delta'_n(\alpha)
\;+\;2\,\delta_n(\alpha)\,\delta'_n(\alpha).
$$

The last three terms are *intrinsic sampling terms*. When the *linear gate* holds and the PR slope has stabilized ($|m_k-m_{k-1}|<\texttt{slope\_tol}$), we approximate $\bar v'(\alpha)\approx m_k$ and $\bar v(\alpha)\approx m_k\,\alpha+b$. The intrinsic terms then behave like mean-zero noise with scale $\mathcal{O}_p(n^{-1/2})$, so the dominant (deterministic) part of the gradient near the target is

$$
\frac{dL}{d\alpha}
\;\approx\;2\bigl(\mu(\alpha)-v_d\bigr)\,m_k
\;\;+\;\;\lambda\,\frac{d s^2}{d\alpha},
$$

and when $\lambda=0$ this further reduces to $2(\mu-v_d)\,m_k$ up to sampling noise.

---

## 5. Why oscillations can persist near the target

Near the solution, $\bar v(\alpha_\star)=v_d$, the deterministic part of the gradient vanishes, but the intrinsic terms remain of order $\mathcal{O}_p(n^{-1/2})$. Consequently a standard gradient step

$$
\alpha_{k+1}
= \alpha_k \;-\; \gamma\,\frac{dL}{d\alpha}(\alpha_k)
$$

continues to move by a noise-driven amount of size $\gamma\,\mathcal{O}_p(n^{-1/2})$. This is the *irreducible stochastic jitter* around the optimum that causes small oscillations in $\alpha_k$ even after the slope $m_k$ is stable and the linear gate passes comfortably.

---

## 6. Minimizing intrinsic error in practice

To reduce this jitter, we control *both* the step size and the sampling noise, in line with classical stochastic approximation:

1. **Learning-rate decay (Robbins–Monro style).**
   Use a decaying schedule $\gamma_i'=\gamma_0/(i+1)$ (or similar) so that updates shrink over time. In the linearized neighborhood where $\bar v'(\alpha)\approx m_k$, the error recursion is

   $$
   e_{k+1} \approx \bigl(1-2\,\gamma_k\,m_k^2\bigr)\,e_k \;+\; \underbrace{\gamma_k\cdot \mathcal{O}_p(n^{-1/2})}_{\text{intrinsic noise}},
   \quad e_k:=\alpha_k-\alpha_\star.
   $$

   As $\gamma_k\downarrow 0$, the noise term is damped and oscillations diminish.

2. **Batch-size (epoch) growth.**
   Increase $n$ over training (e.g., double every fixed number of iterations). Since $\mathrm{sd}(\delta_n)=\sigma/\sqrt{n}$, the noise in the gradient estimate contracts as $n^{-1/2}$:

   $$
   \text{noise scale} \;\sim\; \frac{1}{\sqrt{n}}.
   $$

   Larger $n$ shrinks the bound $2\,s(\alpha)/\sqrt{n}$ used by the linear gate and reduces the jitter in $\alpha_{k+1}$.

3. **Safety cap from the PR slope.**
   When the linear gate is satisfied and $m_k$ has stabilized, cap the *effective* step size using the linear error recursion. If the loss is locally dominated by $2(\mu-v_d)m_k$, then

   $$
   e_{k+1} \approx \bigl(1-2\,\gamma\,m_k^2\bigr)e_k.
   $$

   Ensuring $|1-2\gamma m_k^2|<1$ (e.g., $\gamma \le (1-\eta)/(2m_k^2)$ with a small margin $\eta>0$) yields contraction without large oscillations. This cap uses $m_k^2$ (magnitude only), so it works regardless of the sign of the slope.

---

## 7. How the diagnostics map to the code

* **Intrinsic error.**
  The *intrinsic* sampling variability at $\alpha$ is quantified by the empirical bound

  $$
  \text{bound}(\alpha)\;=\;\texttt{stderr\_tol}\cdot\frac{2\,s(\alpha)}{\sqrt{n}},
  $$

  where $s(\alpha)=\sqrt{s^2(\alpha)}$ is computed from the current batch at $\alpha$.

* **Centered residual.**
  We compute

  $$
  r_{\mathrm{cent}}(\alpha) = \bigl(\mu(\alpha)-\bar\mu\bigr) - m_k\,\bigl(\alpha-\bar\alpha\bigr),
  $$

  using $\bar\alpha,\bar\mu$ from the current window and the latest $m_k$. Passing the linear gate

  $$
  |r_{\mathrm{cent}}(\alpha)| \le \text{bound}(\alpha)
  $$

  indicates that residuals are consistent with *pure sampling noise* around the local line. This is the right condition to (i) trust $m_k$ as a local slope proxy and (ii) apply the $m_k$-based stability cap to $\gamma$.

* **Minimizing the error.**
  We simultaneously:

  * **decay** the learning rate (e.g., PyTorch schedulers) to suppress noise-driven motion,
  * **increase** the batch size $n$ to reduce the scale of $\delta_n$ and shrink the bound,
  * **cap** $\gamma$ when the linear gate is satisfied, to keep the linearized recursion contracting.

Together, these measures reduce oscillations induced by intrinsic error while preserving fast progress when the local mean is well approximated by the stabilized PR slope.

---

## 8. Summary in one line

* The intrinsic error is $\delta_n(\alpha)=\mu(\alpha)-\bar v(\alpha)=\mathcal{O}_p(n^{-1/2})$.
* We *detect* linear, intercept-free behavior via the centered residual gate $|r_{\mathrm{cent}}|\le 2\,s/\sqrt{n}$ (up to a tolerance multiplier).
* Once the gate passes and $m_k$ stabilizes, we (i) cap $\gamma$ using $m_k$ for stability, (ii) decay $\gamma$ and (iii) grow $n$ to suppress the $\mathcal{O}_p(n^{-1/2})$ jitter that otherwise causes small oscillations near the optimum.

---

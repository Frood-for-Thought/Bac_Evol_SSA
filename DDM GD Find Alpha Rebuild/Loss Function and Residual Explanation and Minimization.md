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

**Polyak–Ruppert Slope Estimation**

A *local linear* approximation of the mean is also maintained in a window $[\alpha_{\min},\alpha_{\max}]$ via a centered least-squares Polyak–Ruppert (PR) style slope $m_k$ computed:

$$
m_k \;=\;\frac{\sum_i\bigl(\alpha_i-\bar\alpha\bigr)\bigl(\mu(\alpha_i)-\bar\mu\bigr)}{\sum_i\bigl(\alpha_i-\bar\alpha\bigr)^2+\varepsilon},
\qquad
\bar\alpha:=\frac{1}{K}\sum_i \alpha_i,\quad \bar\mu:=\frac{1}{K}\sum_i \mu(\alpha_i).
$$

This *centered* regression removes intercept bias and makes $m_k$ a consistent estimator of the local slope $d\bar v/d\alpha$ when the window is locally linear. 
The **PR slope estimator** provides a smoothed, sample-based estimate of the local expectation slope:

$$
m_k \approx \frac{d\mu(α)}{dα}
$$

using repeated stochastic observations of ( μ(α) ) over a local window of α-values. When the stochastic field is **heteroskedastic** (variance differs between α points), or when
the **sample standard deviation exceeds one (std > 1)**, unweighted regression fails because large-variance samples dominate the slope.
To correct this, the PR slope is computed using a **Weighted Least Squares (WLS)** formulation whereby each point in the local window contributes according to its weight:

$$
w_i = \frac{n_i}{s_i^2},
$$

* $( n_i )$ — number of samples collected at αᵢ
* $( s_i^2 )$ — unbiased sample variance at αᵢ

Even when heteroskydastic spread is large, weighting by $( 1/s_i^2 )$ ensures that high-variance noisy region contributions are supressed and low-variance α-points have a higher influence on the slope.
This is independent of the variance limit (i.e., λ = 0) in the loss function. The slope is then now calculated as:

$$
m_k = \frac{\sum_i w_i (α_i - \bar{α}_w)(μ_i - \bar{μ}_w)}{\sum_i w_i (α_i - \bar{α}_w)^2 + ε},
$$

with:

$$
\bar{α}_w = \frac{\sum_i w_i α_i}{\sum_i w_i}, \quad
\bar{μ}_w = \frac{\sum_i w_i μ_i}{\sum_i w_i}.
$$


This **centered regression** removes intercept bias and allows a residual gate to test for true linearity using centered residuals, as discussed in section 3.
As the number of samples $( n_i )$ grows, the standard error of μ(α) shrinks like $( 1/\sqrt{n_i} )$, stabilizing both the regression and the residual gate.

---

## 2. Intrinsic error of the sample mean

The sampling error $\delta_n(\alpha)=\mu(\alpha)-\bar v(\alpha)$ satisfies

$$
\mathbb{E}[\delta_n(\alpha)]=0,
\qquad 
\mathrm{Var}\bigl(\delta_n(\alpha)\bigr)=\frac{\sigma^2(\alpha)}{n},
$$

with $\sigma^2(\alpha)=\mathrm{Var}(v_j(\alpha))$. By the CLT, $\delta_n(\alpha)$ is approximately normal with standard deviation $\sigma(\alpha)/\sqrt{n}$. In practice we estimate the population variance, $\sigma(\alpha)$, by the sample variance, $s(\alpha)$, so a two-sided $\approx95\%$ noise band for the *mean* is $\pm 2s(\alpha)/\sqrt{n}$. This is the intrinsic uncertainty that persists even when the optimization is near the target $\bar v(\alpha)\approx v_d$.

---

### 3. Centered residual and its bound

A naive residual $\lvert \mu(\alpha) - m_k\alpha \rvert$ is biased when the local line has a nonzero intercept.
The correct (intercept-free) diagnostic is the centered residual:

$$
r_{\mathrm{cent}}(\alpha)
\=\
\bigl(\mu(\alpha) - \bar{\mu}\bigr)
\-\
m_k\bigl(\alpha - \bar{\alpha}\bigr)
$$

With the **estimated** slope $m_k$,

$$
\underbrace{(\mu-\bar\mu) - m_k(\alpha-\bar\alpha)}_{r_{\text{cent}}}
= \underbrace{(\mu-\bar\mu) - m(\alpha-\bar\alpha)}_{\text{noise}}
\;+\; \underbrace{(m - m_k)}_{\text{slope error}}\cdot(\alpha-\bar\alpha).
$$

So $r_{\text{cent}}$ decomposes into:

* a **mean-zero noise** part (what the stderr bound is designed to control), and
* a **leakage term** from slope estimation error: $(m-m_k)(\alpha-\bar\alpha)$.

**Stderr gate.**: If $\mu(\alpha) \approx m_k\alpha + b$ on the window, then $r_{\mathrm{cent}}(\alpha)$ is approximately sampling noise only. Hence it is appropriate to compare $\lvert r_{\mathrm{cent}}(\alpha) \rvert$ to a standard-error bound. We declare “linear-agreement OK” when:

$$
\boxed{\lvert r_{\mathrm{cent}}(\alpha)\rvert \le\ c\frac{2s(\alpha)}{\sqrt{n}}}
$$

Here $s(\alpha)$ is the sample standard deviation at $\alpha$, $n$ is the batch size, and $c = \verb|stderr_tol|$ is a small multiplier. This gate is used diagnostically to confirm that the local window behaves linearly and that the PR slope $m_k$ is meaningful for step-size control.

**When the PR slope has stabilized**:

$$
\lvert m_k - m_{k-1}\rvert \< \verb|slope_tol|
$$

we expect $m_k$ to be close to $m$. (We don’t know $m$, but the stabilization of $m_k$ gives a practical proxy.)

---

## 4. Linearized gradient and the role of $m_k$

For the variance-penalized loss $L(\alpha)=(\mu(\alpha)-v_d)^2+\lambda \ s^2(\alpha)$, the sample gradient is

$$
\frac{dL}{d\alpha}
=2\bigl(\mu(\alpha)-v_d\bigr)\mu'(\alpha)+\lambda\frac{d s^2(\alpha)}{d\alpha}.
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
=2\bigl(\bar v(\alpha)-v_d\bigr)\bar v'(\alpha)+\lambda\frac{d s^2}{d\alpha}
\+2\delta_n(\alpha)\bar v'(\alpha)
\+2\bigl(\bar v(\alpha)-v_d\bigr)\delta'_n(\alpha)
\+2\delta_n(\alpha)\delta'_n(\alpha)
$$

The last three terms are intrinsic sampling terms. When the linear gate holds and the PR slope has stabilized,

$$
|m_k - m_{k-1}| < slope_tol.
$$

we approximate $\bar v'(\alpha)\approx m_k$ and $\bar v(\alpha)\approx m_k\alpha+b$. Under these conditions the intrinsic terms behave like mean-zero noise of order $O_p(n^{-1/2})$. Thus, the dominant (deterministic) part of the gradient near the target is

$$
\frac{dL}{d\alpha}
\approx\
2\bigl(\mu(\alpha)-v_d\bigr)m_k
+\
\lambda\frac{ds^2}{d\alpha},
$$

and when $\lambda=0$ this further reduces to

$$
\frac{dL}{d\alpha} \approx\ 2\(\mu - v_d)m_k,
$$

up to sampling noise.

---

## 5. Why oscillations can persist near the target

Near the solution, $\bar v(\alpha_\star)=v_d$, the deterministic part of the gradient vanishes, but the intrinsic terms remain of order $\mathcal{O}_p(n^{-1/2})$. Consequently a standard gradient step

$$
\alpha_{k+1}
= \alpha_k \-\gamma\frac{dL}{d\alpha}(\alpha_k)
$$

continues to move by a noise-driven amount of size $\gamma\,\mathcal{O}_p(n^{-1/2})$. This is the *irreducible stochastic jitter* around the optimum that causes small oscillations in $\alpha_k$ even after the slope $m_k$ is stable and the linear gate passes comfortably.

---

## 6. Minimizing intrinsic error in practice

To reduce this jitter, we control *both* the step size and the sampling noise, in line with classical stochastic approximation:

1. **Learning-rate decay (Robbins–Monro style).**
   Use a decaying schedule $\gamma_i'=\gamma_0/(i+1)$ (or similar) so that updates shrink over time. In the linearized neighborhood where $\bar v'(\alpha)\approx m_k$, the error recursion is

$$
e_{k+1} \approx\ \bigl(1 - 2 \gamma_k m_k^2\bigr)\ e_k 
\+\ \underbrace{\gamma_k \cdot \mathcal{O}_p\!\left(n^{-1/2}\right)}_{\text{intrinsic noise}},
\quad e_k = \alpha_k - \alpha_\star
$$

   As $\gamma_k\downarrow 0$, the noise term is damped and oscillations diminish.

2. **Batch-size (epoch) growth.**
   Increase $n$ over training (e.g., double every fixed number of iterations). Since $\mathrm{sd}(\delta_n)=\sigma/\sqrt{n}$, the noise in the gradient estimate contracts as $n^{-1/2}$:

$$
\text{noise scale} \sim\ \frac{1}{\sqrt{n}}
$$

   Larger $n$ shrinks the bound $2\,s(\alpha)/\sqrt{n}$ used by the linear gate and reduces the jitter in $\alpha_{k+1}$.

3. **Safety cap from the PR slope.**
   When the linear gate is satisfied and $m_k$ has stabilized, cap the *effective* step size using the linear error recursion. If the loss is locally dominated by $2(\mu-v_d)m_k$, then

$$
e_{k+1} \approx \bigl(1-2\gamma m_k^2\bigr)e_k
$$

   Ensuring $|1-2\gamma m_k^2|<1$ (e.g., $\gamma \le (1-\eta)/(2m_k^2)$ with a small margin $\eta>0$) yields contraction without large oscillations. This cap uses $m_k^2$ (magnitude only), so it works regardless of the sign of the slope.

   #### Clarification of the “linear recursion” and the bounds

- Define the tracking error:

$$
e_k := \alpha_k - \alpha_\star
$$

  Using the linear model above, the gradient near $\alpha_\star$ is:

$$
\frac{dL}{d\alpha}\big|_{\alpha_k} \approx\ 2 \ m_k^2 \ e_k
$$

- One GD step:

$$
\alpha_{k+1} = \alpha_k - \gamma \cdot (2 \, m_k^2 \, e_k)
$$

  Subtract $\alpha_\star$ from both sides:

$$
e_{k+1} = (\alpha_{k+1} - \alpha_\star)
       = (\alpha_k - \alpha_\star) - 2 \gamma \ m_k^2 \ e_k
       = (1 - 2 \ \gamma \ m_k^2)\ e_k
$$

  This is the linear recursion with multiplier $q := (1 - 2 \gamma m_k^2)$. This holds true even when the mean relation includes a non-zero intercept:
  
$$
\mu(\alpha) = m_k \alpha + b
$$

where the stationary point (target solution) shifts to:

$$
\alpha_\star = \frac{v_d - b}{m_k}
$$

as opposed to the zero-intercept case:

$$
\alpha_\star = \frac{v_d}{m_k}.
$$

The local stability and γ-cap condition remain identical because the gradient term $(\frac{dL}{d\alpha} = 2m_k^2(\alpha - \alpha_\star))$ is still linear in the tracking error $(e_k = \alpha - \alpha_\star)$.

  Intuition: the recursion says each new error $e_{k+1}$ is just the old error $e_k$ multiplied by a constant factor $q$.  
  So the whole behavior depends on $|q|$:  
  - if $|q| < 1$ → errors shrink  
  - if $|q| = 1$ → errors persist  
  - if $|q| > 1$ → errors grow  

- What “convergence” means here:  
  We need $|e_{k+1}| < |e_k|$ for errors to shrink.  
  That requires:

$$
|q| < 1 \quad \Longleftrightarrow \quad |1 - 2 \gamma m_k^2| < 1
$$

  Solving gives:

$$
0 < \gamma < \frac{1}{m_k^2}
$$

  Detailed steps:  
  - Start: $|1 - 2\gamma m_k^2| < 1$  
  - Equivalent to: $-1 < 1 - 2\gamma m_k^2 < 1$  
  - Left inequality: $-1 < 1 - 2\gamma m_k^2 \;\;\Rightarrow\;\; -2 < -2\gamma m_k^2 \;\;\Rightarrow\;\; \gamma < 1/m_k^2$  
  - Right inequality: $1 - 2\gamma m_k^2 < 1 \;\;\Rightarrow\;\; -2\gamma m_k^2 < 0 \;\;\Rightarrow\;\; \gamma > 0$  
  - Combined: $0 < \gamma < 1/m_k^2$  

- Behavior by $\gamma$ range (assume $m_k^2 > 0$; the square handles $m_k < 0$ too):  
  - $0 < \gamma < \tfrac{1}{2 m_k^2}$: $q \in (0, 1)$ → monotone convergence (no sign flips).  
  - $\gamma = \tfrac{1}{2 m_k^2}$: $q = 0$ → one-step to $\alpha_\star$ in ideal linear/noiseless case.  
  - $\tfrac{1}{2 m_k^2} < \gamma < \tfrac{1}{m_k^2}$: $q \in (-1, 0)$ → convergent but oscillatory ($e_k$ flips sign each step).  
  - $\gamma = \tfrac{1}{m_k^2}$: $q = -1$ → no contraction; persistent large oscillation.  
  - $\gamma > \tfrac{1}{m_k^2}$: $|q| > 1$ → divergence (errors grow).  

- Why this matches the intuition:  
  - When $\gamma$ is “too big” relative to the local curvature scale $m_k^2$, the step overshoots, flips the sign, and if $|q| \ge 1$ the amplitude does not decay (oscillates or explodes).  
  - Setting $\gamma \sim 1$ normalizes by nothing; if $m_k \sim 1$ (common in linear patches), then

$$
q \sim 1 - 2 \cdot 1 \cdot 1 = -1,
$$

    i.e., the problematic oscillation factor.
    
  - The square $m_k^2$ is why the condition depends only on the magnitude of slope, not its sign — negative slopes behave the same. The **conditional γ-cap rule** is the key addition that transforms the method into a **self-regularizing stochastic process**.
It ensures that the effective learning rate $( \gamma )$ behaves differently depending on whether the system is still in the *signal-dominant* phase (where μ(α) is far from $v_d$) or in the *noise-limited* phase (where μ(α) is statistically indistinguishable from $v_d$):

Formally, the cap on the step size is defined as:

$$
\gamma_{cap} =
\frac{1 - \eta}{2 m_k^2}, \quad |μ(α_k) - v_d| > \frac{s(α_k)}{\sqrt{n}}
$$

$$
\gamma_{cap} =
\frac{lr_{scale}*(1 - \eta) }{2 m_k^2}, \quad |μ(α_k) - v_d| \le \frac{s(α_k)}{\sqrt{n}}
$$

where $lr_{scale}$ is the scheduler-controlled adaptive learning rate decay factor from the learning-rate scheduler to reduce $\gamma_{cap}$ to arrive closer to the local minimum for:

$$
e_{k+1} \approx \bigl(1 - 2 \gamma_{cap} \ m_k^2)\ e_k \+\ \gamma_{cap} \cdot \text{noise}
$$

- Effect of the variance term ($\lambda > 0$):  
  - The gradient gains $+ \lambda d(s^2)/d\alpha$.  
  - If that term is small near the target or comparatively flat, the $m_k^2$-driven analysis dominates.  
  - If it is not small, it perturbs $q$ slightly; the same form still holds locally with $m_k$ replaced by the effective local slope factor of the full gradient.

---

## 7. How the diagnostics map to the code

* **Intrinsic error.**
  The *intrinsic* sampling variability at $\alpha$ is quantified by the empirical bound

$$
\text{bound}(\alpha) \=\ \verb|stderr_tol|\cdot\frac{2\ s(\alpha)}{\sqrt{n}}
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

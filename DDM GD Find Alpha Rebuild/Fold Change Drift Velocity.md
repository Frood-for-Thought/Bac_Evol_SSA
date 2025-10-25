## **The α-Based Temporal–Stimulation Model: From Receptor Dynamics to Drift Velocity**

### **1. Physical and Biological Meaning of α**

Each receptor can bind or unbind attractant molecules, producing a **fractional occupancy**:

$$
P_b = \frac{N_{\text{bound}}}{N_{\text{total}}}
$$

where $(N_{\text{bound}})$ is the number of ligand-bound receptors and $(N_{\text{total}})$ is the total number of receptor sites.
The time derivative $(dP_b/dt)$ measures how rapidly receptors detect a changing environment.

The instantaneous **run–tumble bias** is defined as:

$$
\eta_\pm(t) = A\exp\big[t(-r_0 \pm \alpha\tfrac{dP_b}{dt})\big]
$$

where $(r_0)$ is the baseline tumble rate and $(A)$ is a scaling constant.
The product $( \alpha  dP_b/dt )$ modulates this baseline rate: as attractant increases $((dP_b/dt > 0))$, runs lengthen; when attractant decreases, tumbling increases.

Dimensional analysis shows α is a parameter that defines *how frequently* the cell integrates receptor information, rather than by measuring methylation feedback.

---

### **2. Fold-Change Detection and the Inverse-Occupancy Scaling**

When receptor occupancy lies between 0 and 1, α scales inversely with the fraction of bound receptors:

$$
\alpha \propto \frac{1}{P_b}
$$

When few receptors are bound $(low (P_b))$, the system is highly responsive (large α), while at saturation $(high (P_b))$ sensitivity declines.
Substituting the product $( \alpha dP_b/dt )$ gives:

$$
\frac{1}{P_b}\frac{dP_b}{dt} = \frac{d(\ln P_b)}{dt}
$$

which is the rate at which receptor occupancy changes relative to its current state, i.e. **Weber-law of sensing** [1] whereby the bacterium responds to **relative** changes. 

The filter α is a placeholder function that describes the dynamic self-regulating feedback from methylation and depends on receptor occupancy. If Methylation (CheR/CheB) works on receptor activity A(t) to bring it back toward an adapted value, α is a nonlinear feedback filter that rescales how strongly that correction is felt by the bacterial system to dial down sensitivity to avoid overreaction. 

At low gradients $((P_b\ll1))$, methylation can adapt to the concentration, and α represents a high-pass filter to amplify low-frequency gradient signals. The filter α gets larger when methylation can easily compensate so the bacterial system can track environmental changes closely.

In steep gradients $((P_b\to1))$, methylation cannot keep up with the rapid changes in concentration, so α represents a low-pass filter to prevent the model from overreacting to noise by filtering out high-frequency gradient signals. It automatically gets smaller when methylation falls behind and receptors saturate so cells can’t “see” further up the gradient. 

Instead of filtering out methylation adaptation frequency signals, this instead is a dynamic, nonlinear **amplitude** filter to contol the magnitude of the signal based on receptor occupancy $(P_b)$. 
Previous models fail in steep gradients because methylation sites saturate and can no longer bring receptor activity to baseline, causing adaptation to fail and chemosensing to collapse creating nonstop runs.
In the α-model, because tumble rate depends on $(d(\ln P_b)/dt)$, the feedback gain from methylation logarithmically scales with how occupied the receptors are to detect **relative** changes in concentration, and scale the sensitivity preventing adaptation from failing.
Therefore, drift velocity continues smoothly in steep gradients instead of collapsing to replicate the theoretical drift velocity in experiemnts.

---

### **3. Drift Velocity Derived from Receptor Dynamics**

![Alpha and Drift Velocity vs Position](Alpha%20and%20Drift%20Velocity%20vs%20Position.jpg)

![Alpha with mk and Rtroc vs Position](Alpha%20with%20mk%20and%20Rtroc%20vs%20Position.jpg)

The data collected shows the relation between mean drift velocity μ and the receptor response is:

$$
m_k = \frac{d\mu}{d\alpha} = k\frac{dP_b}{dt}
$$

Here (k) translates molecular receptor changes (s⁻¹) into macroscopic velocity changes (m·s⁻¹).
Solving the differential equation gives:

$$
\mu(\alpha) = k\alpha\frac{dP_b}{dt} + \mu_0(\alpha)
$$

μ₀(α) represents the baseline mean drift velocity for an initial α.
Because α is proportional to (1/P_b), the equation is then:

$$
\mu(\alpha) = k'\frac{1}{P_b}\frac{dP_b}{dt} + \mu_0(\alpha)
$$

Simplifying the equation gives:

$$
\boxed{\mu(\alpha) = k'\frac{d(\ln P_b)}{dt} + \mu_0(\alpha)}
$$

---

### Non-linear extension

At high receptor occupancy $(P_b \to 1)$ or large $(\alpha dP_b/dt)$,
the physical response of the flagellar motor saturates and the linear proportionality between μ and α breaks down.
Therefore the linear effect of α on μ decreases because the drift velocity approaches its maximum $( \mu_{\max} )$.
The relation can be expressed as:

$$
\boxed{
\mu(\alpha) = \mu_{\max}\left[1 - e^{-\frac{k\alpha dP_b/dt}{\mu_{\max}}}\right] + \mu_0(\alpha)
}
$$

which reduces to the linear form:

$$
\mu(\alpha) \approx k\alpha\frac{dP_b}{dt} + \mu_0(\alpha)
$$

when $(\alpha dP_b/dt)$ is small, but saturates as $(\alpha dP_b/dt)$ becomes large.

This represents the **non-linear α–μ relation** observed in steep gradients at very high drift velocity, the tumble rate difference from receptor-motor coupling remains monotonic but flattens as the maximum drift velocity is approached. 

In the linear region, $\mu = m_k \alpha + b$ and $\frac{d\mu}{d\alpha} = m_k$, and the derivative $m_k$ is the true slope. 
In the non-linear regions, if $\mu(\alpha)$ has a quadratic term:

$$
\mu(\alpha) = b + m\alpha + c\alpha^2
$$

Where *m* is the true (physical) slope in the underlying equation, *mₖ* is the fitted slope across the α-window, and *b* is the true intercept.
Therefore, $(m_k = d\mu/d\alpha)$, measured across the α window becomes a **secant** of the non-linear equation, rather than it replicating an α–μ equation with a constant linear slope. The larger the alpha window the more the slope inherits curvature bias $(c\Delta\alpha_{\text{window}})$:

$$
\frac{d\mu}{d\alpha} = m + 2c\alpha = m_k \approx m + c\Delta\alpha_{\text{window}}
$$

Although, when the mean drift velocity equation within the alpha window is linear enough for the residual check of the secant to still pass as linear, $(\boxed{\lvert r_{\mathrm{cent}}(\alpha)\rvert \le\ c\frac{2s(\alpha)}{\sqrt{n}}})$, the ML model will still converge.

---

### **4. Biological Meaning**

The slope observed within the linear α–μ relation, $(m_k = d\mu/d\alpha)$, is isometric to the receptor rate $(dP_b/dt)$, which describes how the behavioral output for drift velocity changes within alpha space (α-domain) is proportional to how the receptor signal changes as the environment evolves in real time (t-domain).
α acts as a temporal control variable filtering adaptation, while the mobility constant $(k')$ gives a proportion to how **relative** receptor changes describe drift velocity.

Fold-change detection (FCD) by Goentoro and Alon [1] was made as a general sensory law for biological systems and experimentally verified in E. coli chemotaxis by Tu *et al.* [2], showed that *E. coli* chemotaxis depends on the **relative rate** of receptor stimulation through biochemical methylation feedback.  This model arrives at **the same FCD principle**, but instead of requiring methylation integrators, it reproduces the logarithmic sensing behavior through a temporal control variable α acting as a frequency filter. The α filter is reproduced through stochastic learning by D-DEME alongside the gradient of the drift velocity within a deme's alpha space, which mirrors the chemotactic FCD receptor response to environmental change seen in nature. Through the statistical aggregate data of the ML model, each deme recreates the drift velocity as the evolving mean $(d\mu/d\alpha)$ acts as a macroscopic analogue of how receptor feedback converts the time rate of change of protein bound $(dP_b/dt)$ into directed motion.

---

### **References**

1. L. Goentoro, O. Shoval, M. Kirkegaard, Y. Hart, E. Mayo, and U. Alon,
   “The incoherent feedforward loop can provide fold-change detection in gene regulation,”
   *Molecular Cell*, vol. 36, no. 6, pp. 894–899, Dec. 2009.
   doi: 10.1016/j.molcel.2009.11.018.

2. Y. V. Kalinin, L. Jiang, Y. Tu, and M. Wu,
   “Logarithmic sensing in *Escherichia coli* bacterial chemotaxis,”
   *Biophysical Journal*, vol. 96, no. 6, pp. 2439–2448, Mar. 2009.
   doi: 10.1016/j.bpj.2008.10.027.

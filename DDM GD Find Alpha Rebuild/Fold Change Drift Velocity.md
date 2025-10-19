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

which is the rate at which receptor occupancy changes relative to its current state, i.e. **Weber-law of sensing** whereby the bacterium responds to **relative** changes.
In shallow gradients,  In steep gradients, methylation cannot keep up with the rapid changes in concentration, so α represents a low-pass filter to prevent the model from overreacting to noise by filtering out high-frequency gradient signals. 
At low concentrations $((P_b\ll1))$, methylation can adapt to the concentration, and α represents a high-pass filter to amplify low-frequency gradient signals.
In steep gradients $((P_b\to1))$, methylation cannot keep up with the rapid changes in concentration, so α represents a low-pass filter to prevent the model from overreacting to noise by filtering out high-frequency gradient signals.

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

μ₀(α) represents the baseline mean drift velocity for an initial α. Because α is proportional to $(1/P_b)$, the equation is then:

$$
\mu(\alpha) = k'\frac{1}{P_b}\frac{dP_b}{dt} + \mu_0(\alpha)
$$

Simplifying the equation gives:

$$
\boxed{\mu(\alpha) = k'\frac{d(\ln P_b)}{dt} + \mu_0(\alpha)}
$$

---

### **4. Biological Meaning**

Classical methylation models fail in steep gradients because finite methylation sites saturate with receptor activity freezing while adaptation stops, producing nonstop runs limited only by rotational diffusion.
In the α-model, because the drift law depends solely on $(d(\ln P_b)/dt)$, receptor sites no longer saturate, the logarithmic equations shows receptors detect relative changes and drift velocity decreases smoothly rather than collapsing.
This mechanism removes the boundary from methylation-limited simulations and preserves continuous drift over all concentration ranges.

The slope $(m_k = d\mu/d\alpha)$ is isometric to the receptor rate $(dP_b/dt)$, which describes how the behavioral output for drift velocity changes within alpha space (α-domain) is proportional to how the receptor signal changes as the environment evolves in real time (t-domain).
α acts as a temporal control variable filtering adaptation, while the mobility constant $(k')$ gives a proportion to how how **relative** receptor changes describe drift velocity. 

Tu et al's fold-change detection (FCD) showed that *E. coli* chemotaxis depends on the **relative rate** of receptor stimulation through biochemical methylation feedback.  This model arrives at **the same FCD principle**, but instead of requiring methylation integrators, it reproduces the logarithmic sensing behavior through a temporal control variable α acting as a frequency filter. The filter is reproduced through stochastic learning by D-DEME within a deme's alpha space that mirrors the chemotactic receptor response to environmental change seen in nature.

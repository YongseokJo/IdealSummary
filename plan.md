

## NOTES

### Four-step story (paper outline)
1. Show that galaxy catalogs carry more information than SMF (SMF is a lossy summary) using standard ML models (semi-interpretable).
2. Analyze which parts of the learned machine summary constrain which physical parameters (extend to multiple galaxy properties, not only stellar mass).
3. Replace the learned set encoder with **fully symbolic, permutation-invariant summary statistics** that are human-interpretable and observation-reproducible.
4. Apply the same symbolic summaries to **masked / incomplete catalogs** (realistic observational setting), using masking inside the symbolic expressions.

---

## Symbolic permutation-invariant summary statistics (fully symbolic Set-DSR)
**Goal:** Learn summary statistic(s) directly from a variable-length set of galaxy properties
\[
X = \{x_i\}_{i=1}^{N}, \quad x_i \in \mathbb{R}^D
\]
with mask/weights \(m_i\) (binary or completeness weights). Output a small vector of summary statistics
\[
s(X, m) \in \mathbb{R}^K
\]
that is **permutation invariant** and **symbolic**.

### Key requirement
- No external pooling “module” like DeepSets. Instead, *pooling/reduction is part of the symbolic language* (grammar).
- The symbolic program must be defined using **Reduce operators** over the galaxy index \(i\).

### Inputs (scalable beyond stellar mass)
- Currently: \(x_i = [M_{\star,i}]\) (or \(\log M_{\star,i}\)).
- Later: \(x_i = [\log M_{\star,i},\ \mathrm{color}_i,\ \mathrm{SFR}_i,\ R_i,\ ...]\).
- Keep implementation generic: reducers operate on any scalar expression \(f(x_i)\) built from available features.

### Masking (must be inside the equation)
Define:
- \(N_{\mathrm{eff}} = \mathrm{SUM}_i(m_i)\)
- \(\mathrm{masked\_sum}(f) = \mathrm{SUM}_i\big(m_i \cdot f(x_i)\big)\)
- \(\mathrm{masked\_mean}(f) = \mathrm{masked\_sum}(f) / (N_{\mathrm{eff}} + \epsilon)\)

All “means”/normalizations must be represented this way so the same expression works for incomplete catalogs.

### Symbolic grammar (Set-DSR)
**Reduce operators (permutation invariant):**
- `SUM( expr_i )`
- `LOGSUMEXP( expr_i )` (smooth max-like; optional)
- `MAX( expr_i )` (optional; may be non-smooth)

**Scalar operators (inside/outside Reduce):**
- `+`, `-`, `*`, `safe_div(a,b)=a/(|b|+eps)`
- `square`, `abs`, `tanh`
- `log(|x|+eps)`, `sqrt(|x|+eps)`
- (optional later) `exp(clamp(x))`

**Terminals:**
- per-galaxy features: components of \(x_i\) (start with stellar mass only)
- mask/weight: \(m_i\)
- constants (ephemeral constants)

### Model usage
- Learn \(K\) symbolic summaries \(s_k(X,m)\) (small K).
- Predict physical parameters with a simple head:
  \[
  \hat{\theta} = \mathrm{MLP}(s)
  \]
  Keep MLP small so interpretability is carried by \(s\), not the head.

### Training / interpretability constraints
- Penalize expression complexity (program length / operator count).
- Operator curriculum:
  1) start with `SUM` + `{+, *, square}`
  2) add `safe_div`, `log`, `sqrt`
  3) optionally add `LOGSUMEXP` / `MAX`
- After training: prune small terms and refit constants to obtain clean final expressions.

---

## Practical note
- Before final symbolic comparisons, apply consistent catalog cuts (mass threshold / completeness) and document them.



# IdealSummary — Nodes (working plan)

## Current status
- Models implemented/used so far:
  - MLP on SMF
  - DeepSet on galaxy catalog
  - SlotSetPool on galaxy catalog
- SlotSetPool currently looks best, but needs a fair comparison via Optuna.

## Near-term TODO
- Run Optuna for the three models (MLP/DeepSet/SlotSetPool).
- Finalize evaluation protocol (same data splits, same metrics, same cuts).
- Decide/implement the symbolic summary-statistic encoder (leaning Set-DSR).

## 4-step story (paper outline)
1. Show galaxy catalogs contain more information than SMF (SMF is a lossy summary), using standard ML models (semi-interpretable).
2. Diagnose which parts of the learned “machine summary” constrain which physical parameters; extend beyond stellar mass to more galaxy properties (color, SFR, etc.).
3. Replace the learned set encoder with **human-interpretable, scientifically meaningful, observation-reproducible** symbolic summary statistics learned from catalogs.
4. Apply the same approach to **masked / incomplete catalogs** (observational realism).

---

## Symbolic permutation-invariant summary statistics (primary direction: Set-DSR)
### Goal
Learn a **K-dimensional** summary
- input: variable-length set of galaxies per simulation: \(X=\{x_i\}_{i=1}^N\), with mask/weights \(m_i\)
- output: \(s(X,m)\in\mathbb{R}^K\)
- then: \(\hat{\theta}=\mathrm{MLP}(s)\)

### Key requirement (no external pooling module)
Permutation invariance must be guaranteed by **Reduce operators inside the symbolic program/grammar**, e.g.
- `SUM_i(·)`, `MEAN_i(·)`, `MAX_i(·)`, `LOGSUMEXP_i(·)` (and/or soft variants)

### Masking (must be inside the equation)
- \(N_\mathrm{eff}=\mathrm{SUM}_i(m_i)\)
- `masked_sum(f) = SUM_i(m_i * f(x_i))`
- `masked_mean(f) = masked_sum(f) / (N_eff + eps)`
This should work identically for complete and masked catalogs.

### Symbolic regression approach
- Use **DSR-style** program search over a typed grammar with:
  - Reduce operators (SUM/MEAN/LOGSUMEXP/…)
  - safe scalar ops: `+ - * safe_div log(|x|+eps) sqrt(|x|+eps) square abs tanh`
  - terminals: per-galaxy features `x_i` (currently stellar mass; later add color/SFR/…), mask `m_i`, constants
- Learn \(K\) programs \(s_1,\dots,s_K\) (small K), then predict parameters with a small MLP head.

### Regularization / interpretability
- Penalize expression complexity (program length / operator count).
- Operator curriculum:
  1) start with `{+, *, square}` (+ `SUM`)
  2) add `{safe_div, log, sqrt}`
  3) optionally add `{LOGSUMEXP, MAX}`
- After training: prune tiny terms and refit constants for clean closed-form expressions.

## Practical note
- Need consistent catalog cuts (mass threshold / completeness) before final comparisons.
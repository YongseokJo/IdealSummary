## PROGRESS
- I have three models: MLP for SMF, DeepSet and SlotSetPool for galaxy catalog.
- Currently, SlotSetPool seems outperforming other combination, but for the precise comparison, I have to run Optuna for each model.


## TO-DO
- Run Optuna for three models and get a more stabilized and finalized comparison result.
- Decide the target “symbolic summary statistic” interface (inputs/outputs, masking, operator set) and implement Set-DSR.
- Add support for additional per-galaxy properties beyond stellar mass (color, SFR, size, etc.).



## URGET
-- Make each slot different/diverse/apart from each other
-- Yet, if they can't find a good distict slot and inactivate the slots.
-- I should have applied some cuts to the catalog




### Four-step story (paper outline)
1. Show that galaxy catalogs carry more information than SMF (SMF is a lossy summary) using standard ML models (semi-interpretable).
2. Analyze which parts of the learned machine summary constrain which physical parameters (extend to multiple galaxy properties, not only stellar mass).
3. Replace the learned set encoder with **fully symbolic, permutation-invariant summary statistics** that are human-interpretable and observation-reproducible.
4. Apply the same symbolic summaries to **masked / incomplete catalogs** (realistic observational setting), using masking inside the symbolic expressions.

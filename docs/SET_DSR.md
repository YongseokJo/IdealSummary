# Set-DSR: Deep Symbolic Regression for Permutation-Invariant Summary Statistics

## Overview

Set-DSR learns **symbolic, interpretable expressions** that summarize variable-length sets of data (e.g., galaxy catalogs) into fixed-length feature vectors. The learned expressions are permutation-invariant by construction.

**Key Features:**
- Grammar-based symbolic program generation
- Permutation-invariant via reduce operations (SUM, MEAN, WSUM, WMEAN, etc.)
- Two-phase training: structure search + parameter fine-tuning
- Curriculum learning for progressive operator unlocking
- Supports GP-based evolution or RL-based search

---

## Architecture

### Grammar System

The grammar enforces a typed structure that guarantees permutation invariance:

```
SCALAR → MLP(REDUCE, REDUCE, ...)
REDUCE → SUM(PER_ELEMENT) | MEAN(PER_ELEMENT) | WSUM(PER_ELEMENT) | WMEAN(PER_ELEMENT) | ...
PER_ELEMENT → x_i | ADD(PER_ELEMENT, PER_ELEMENT) | MUL(...) | LOG(...) | ...
```

**Type Hierarchy:**
1. **PER_ELEMENT**: Operations on individual set elements (e.g., `x₁ + x₂`, `log(x₁)`)
2. **REDUCE**: Aggregation over the set (e.g., `SUM(...)`, `MEAN(...)`)
3. **SCALAR**: Final output combining multiple reduce operations via MLP

### Operator Scopes

Control available operators via `--operator-scope`:

| Scope | Operators |
|-------|-----------|
| `simple` | `+`, `-`, `*`, `/`, `x_i`, `const`, `SUM`, `MEAN`, `WSUM`, `WMEAN` |
| `intermediate` | + `LOG`, `EXP`, `SIN`, `COS`, `SQRT`, `ABS`, `POW` |
| `full` | + `LOGSUMEXP`, `MIN`, `MAX`, `VAR`, `STD` |

### Weighted Reductions

`WSUM` and `WMEAN` learn element-wise weights:
- `WSUM(f(x))` = Σᵢ wᵢ · f(xᵢ), where wᵢ = softmax(MLP(xᵢ))
- `WMEAN(f(x))` = WSUM / Σwᵢ

---

## Training Phases

### Phase 1: Structure Search

Searches for optimal symbolic program structures using either:

**A) GP-Based Evolution (default)**
- Tournament selection
- Crossover between elite individuals
- Point mutation and subtree mutation
- Fitness = R² - λ·complexity

**B) RL-Based Search (`--use-rl-search`)**
- Policy network generates programs token-by-token
- REINFORCE with baseline for gradient estimation
- Entropy bonus for exploration

### Phase 2: Parameter Fine-Tuning

After structure search, fine-tune:
1. **MLP weights**: Map K symbolic features → output parameters
2. **Learned constants**: Via `ParameterizedSetDSR` wrapper (if constants exist in expressions)

---

## Fitness Evaluation

### Quick Linear Fit (Every Evaluation)
Uses `torch.linalg.lstsq` to fit a linear layer on program outputs:
```
min_W ||Y - X·W||²
```
- O(K²) per individual
- Provides meaningful fitness without full MLP training

### Periodic MLP Retrain (Every N Generations)
Every `--mlp-retrain-interval` generations:
1. Take the current best individual
2. Retrain full MLP for `--mlp-retrain-epochs` epochs
3. Captures non-linear feature interactions

---

## Curriculum Learning

Progressive unlocking of operators over generations:

| Level | Generations | Operators Added |
|-------|-------------|-----------------|
| 1 | 0 - 33% | Basic: `+`, `-`, `*`, `/`, `SUM`, `MEAN` |
| 2 | 33% - 66% | + `LOG`, `EXP`, `SQRT`, `POW` |
| 3 | 66% - 100% | + Full scope (trig, advanced reductions) |

Enable with `--use-curriculum`.

---

## CLI Reference

### Core Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--n-generations` | 100 | Number of evolution generations |
| `--population-size` | 50 | Population size for GP |
| `--n-programs` | 8 | Number of symbolic programs (K) |
| `--complexity-weight` | 0.01 | Penalty for program complexity |

### Operator & Grammar

| Argument | Default | Description |
|----------|---------|-------------|
| `--operator-scope` | `simple` | Operator set: `simple`, `intermediate`, `full` |
| `--use-curriculum` | False | Enable curriculum learning |
| `--max-depth` | 4 | Maximum expression tree depth |

### Evolution Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-subtree-mutation` | False | Enable subtree mutation |
| `--use-constant-optimization` | False | Optimize constants during evolution |
| `--const-opt-steps` | 10 | Steps for constant optimization |
| `--const-opt-lr` | 0.01 | Learning rate for constant optimization |

### MLP Retraining

| Argument | Default | Description |
|----------|---------|-------------|
| `--mlp-retrain-interval` | 10 | Retrain MLP every N generations |
| `--mlp-retrain-epochs` | 20 | Epochs per MLP retrain |

### RL Search (`--use-rl-search`)

| Argument | Default | Description |
|----------|---------|-------------|
| `--rl-lr` | 1e-3 | Policy network learning rate |
| `--rl-entropy-weight` | 0.01 | Entropy bonus weight |
| `--rl-baseline-decay` | 0.99 | Exponential moving average decay |

### Post-Processing

| Argument | Default | Description |
|----------|---------|-------------|
| `--simplify-expressions` | False | Simplify with SymPy after training |
| `--phase2-epochs` | 0 | Epochs for Phase 2 fine-tuning |
| `--phase2-lr` | 1e-4 | Learning rate for Phase 2 |

---

## Output Files

Training saves to `--save-dir` (default: `data/models/set_dsr/`):

| File | Description |
|------|-------------|
| `set_dsr_TIMESTAMP.pt` | Model checkpoint |
| `expressions_TIMESTAMP.json` | Learned symbolic expressions |
| `history_TIMESTAMP.json` | Training metrics history |

---

## Weights & Biases Logging

Metrics logged to wandb:

| Metric | Description |
|--------|-------------|
| `evo/best_fitness` | Best fitness in population |
| `evo/mean_fitness` | Population mean fitness |
| `evo/best_r2` | R² of best individual |
| `evo/mean_complexity` | Mean program complexity |
| `evo/r2_param_{i}` | Per-parameter R² scores |
| `phase2/train_loss` | Phase 2 training loss |
| `phase2/val_r2` | Phase 2 validation R² |

---

## Example Usage

### Basic GP Evolution
```bash
python train_dsr.py \
    --data-path ../data/camels_LH.hdf5 \
    --n-generations 100 \
    --population-size 50 \
    --n-programs 8 \
    --operator-scope intermediate \
    --use-subtree-mutation \
    --mlp-retrain-interval 10
```

### With Curriculum Learning
```bash
python train_dsr.py \
    --data-path ../data/camels_LH.hdf5 \
    --n-generations 150 \
    --operator-scope full \
    --use-curriculum \
    --use-constant-optimization \
    --complexity-weight 0.005
```

### RL-Based Search
```bash
python train_dsr.py \
    --data-path ../data/camels_LH.hdf5 \
    --use-rl-search \
    --rl-lr 1e-3 \
    --rl-entropy-weight 0.02 \
    --n-generations 200
```

### With Phase 2 Fine-Tuning
```bash
python train_dsr.py \
    --data-path ../data/camels_LH.hdf5 \
    --n-generations 100 \
    --phase2-epochs 50 \
    --phase2-lr 1e-4 \
    --simplify-expressions
```

---

## Model Components

### `SetDSR` (src/set_dsr.py)
Core model class containing:
- `Grammar`: Defines typed operators and production rules
- `programs`: List of K symbolic expression trees
- `mlp`: Neural network mapping K features → output dimension
- `execute_programs()`: Evaluates expressions on input sets

### `SetDSREvolver`
GP-based evolution with:
- Tournament selection
- Single-point crossover
- Point mutation
- Quick linear fit for fitness
- Periodic MLP retraining

### `AdvancedSetDSREvolver`
Extends base evolver with:
- Subtree mutation (replace entire subtrees)
- Constant optimization (gradient-based tuning of constants)

### `RLProgramSearch`
Alternative to GP:
- LSTM-based policy network
- Generates programs autoregressively
- REINFORCE training with entropy regularization

### `ParameterizedSetDSR`
Phase 2 wrapper:
- Extracts constants from expressions as `nn.Parameter`
- Enables end-to-end backprop through expressions

---

## Tips & Best Practices

1. **Start simple**: Use `--operator-scope simple` first to find basic patterns
2. **Control complexity**: Increase `--complexity-weight` if expressions are too large
3. **Use curriculum**: `--use-curriculum` helps avoid local optima in complex operator spaces
4. **Enable constant optimization**: `--use-constant-optimization` refines numerical constants
5. **Phase 2 fine-tuning**: Add `--phase2-epochs 50` for final polish after GP search
6. **Monitor per-param R²**: Check `evo/r2_param_*` in wandb to identify hard parameters

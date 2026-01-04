"""
Set-DSR: Deep Symbolic Regression for Permutation-Invariant Summary Statistics

This module implements symbolic regression over variable-length sets (galaxy catalogs)
with reduce operators (SUM, MEAN, LOGSUMEXP, MAX) as part of the symbolic grammar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import random


# =============================================================================
# Type system for the grammar
# =============================================================================

class ExprType(Enum):
    """Types in our typed grammar."""
    SCALAR = "scalar"          # single value per simulation (after reduction)
    PER_ELEMENT = "per_elem"   # one value per galaxy (before reduction)


# =============================================================================
# Operators / Tokens
# =============================================================================

@dataclass
class Operator:
    """Represents an operator in the symbolic grammar."""
    name: str
    arity: int
    input_types: Tuple[ExprType, ...]
    output_type: ExprType
    func: Callable
    complexity: float = 1.0


def safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return a / (b.abs() + eps)


def safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x.abs() + eps)


def safe_sqrt(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(x.abs() + eps)


def clamped_exp(x: torch.Tensor, max_val: float = 10.0) -> torch.Tensor:
    return torch.exp(torch.clamp(x, -max_val, max_val))


# =============================================================================
# Reduce operators (permutation invariant)
# =============================================================================

def masked_sum(expr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    expr: (batch, max_galaxies) or (batch, max_galaxies, dim)
    mask: (batch, max_galaxies)
    returns: (batch,) or (batch, dim)
    """
    if expr.dim() == 2:
        return (expr * mask).sum(dim=1)
    else:
        return (expr * mask.unsqueeze(-1)).sum(dim=1)


def masked_mean(expr: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Masked mean: sum / N_eff."""
    n_eff = mask.sum(dim=1, keepdim=True) + eps
    if expr.dim() == 2:
        return (expr * mask).sum(dim=1) / n_eff.squeeze(-1)
    else:
        return (expr * mask.unsqueeze(-1)).sum(dim=1) / n_eff


def masked_logsumexp(expr: torch.Tensor, mask: torch.Tensor, neg_inf: float = -1e9) -> torch.Tensor:
    """Masked logsumexp (smooth max)."""
    masked_expr = expr.clone()
    if expr.dim() == 2:
        masked_expr[mask == 0] = neg_inf
        return torch.logsumexp(masked_expr, dim=1)
    else:
        masked_expr[mask.unsqueeze(-1).expand_as(expr) == 0] = neg_inf
        return torch.logsumexp(masked_expr, dim=1)


def masked_max(expr: torch.Tensor, mask: torch.Tensor, neg_inf: float = -1e9) -> torch.Tensor:
    """Masked max."""
    masked_expr = expr.clone()
    if expr.dim() == 2:
        masked_expr[mask == 0] = neg_inf
        return masked_expr.max(dim=1).values


def masked_weighted_sum(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mask-aware weighted sum over set dimension.

    values: (B, N) or (B, N, D)
    weights: (B, N) or (B, N, D)
    mask: (B, N)
    returns: (B,) or (B, D)
    """
    if values.dim() not in (2, 3):
        raise ValueError(f"values must be 2D or 3D, got {values.shape}")
    if weights.dim() not in (2, 3):
        raise ValueError(f"weights must be 2D or 3D, got {weights.shape}")

    if values.dim() == 2:
        m = mask
    else:
        m = mask.unsqueeze(-1)

    w = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    w = w * m
    v = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return (v * w).sum(dim=1)


def masked_weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mask-aware weighted mean over set dimension."""
    if values.dim() == 2:
        denom = (torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0) * mask).sum(dim=1) + eps
    else:
        denom = (torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0) * mask.unsqueeze(-1)).sum(dim=1) + eps
    return masked_weighted_sum(values, weights, mask, eps=eps) / denom


# =============================================================================
# Grammar definition
# =============================================================================

class Grammar:
    """
    Typed grammar for Set-DSR.
    
    Operators are split into:
    - per-element ops: operate on per-galaxy values
    - reduce ops: reduce per-galaxy values to per-simulation scalars
    - scalar ops: operate on scalars (after reduction)
    
    operator_scope controls complexity:
    - "simple": Only arithmetic (+, -, *, /, square, abs) + basic reduce (SUM, MEAN)
    - "intermediate": + log, sqrt, exp, tanh, sin, cos + WSUM, WMEAN
    - "full": + LOGSUMEXP, MAX, more exotic ops
    
    curriculum_level controls when operators become available during training:
    - Level 1: Basic ops only
    - Level 2: Intermediate ops
    - Level 3: All ops (within the scope)
    """
    
    def __init__(
        self,
        n_features: int = 1,
        feature_names: Optional[List[str]] = None,
        curriculum_level: int = 1,
        operator_scope: str = "full",  # "simple", "intermediate", "full"
    ):
        self.n_features = n_features
        self.feature_names = feature_names or [f"x_{i}" for i in range(n_features)]
        self.curriculum_level = curriculum_level
        self.operator_scope = operator_scope
        
        self._build_operators()
        self._build_terminals()
    
    def _build_operators(self):
        """Build operator library based on operator_scope and curriculum_level."""
        
        scope = self.operator_scope
        level = self.curriculum_level
        
        # =====================================================================
        # Per-element unary operators
        # =====================================================================
        self.per_elem_unary = [
            Operator("identity", 1, (ExprType.PER_ELEMENT,), ExprType.PER_ELEMENT, lambda x: x, 0.0),
            Operator("square", 1, (ExprType.PER_ELEMENT,), ExprType.PER_ELEMENT, lambda x: x ** 2, 1.0),
            Operator("abs", 1, (ExprType.PER_ELEMENT,), ExprType.PER_ELEMENT, torch.abs, 1.0),
        ]
        
        # Intermediate scope: add log, sqrt, tanh, sin, cos
        if scope in ("intermediate", "full") and level >= 2:
            self.per_elem_unary.extend([
                Operator("safe_log", 1, (ExprType.PER_ELEMENT,), ExprType.PER_ELEMENT, safe_log, 2.0),
                Operator("safe_sqrt", 1, (ExprType.PER_ELEMENT,), ExprType.PER_ELEMENT, safe_sqrt, 2.0),
                Operator("tanh", 1, (ExprType.PER_ELEMENT,), ExprType.PER_ELEMENT, torch.tanh, 1.5),
                Operator("sin", 1, (ExprType.PER_ELEMENT,), ExprType.PER_ELEMENT, torch.sin, 2.0),
                Operator("cos", 1, (ExprType.PER_ELEMENT,), ExprType.PER_ELEMENT, torch.cos, 2.0),
            ])
        
        # Full scope: add exp
        if scope == "full" and level >= 3:
            self.per_elem_unary.extend([
                Operator("exp", 1, (ExprType.PER_ELEMENT,), ExprType.PER_ELEMENT, clamped_exp, 3.0),
            ])
        
        # =====================================================================
        # Per-element binary operators
        # =====================================================================
        self.per_elem_binary = [
            Operator("add", 2, (ExprType.PER_ELEMENT, ExprType.PER_ELEMENT), ExprType.PER_ELEMENT, torch.add, 1.0),
            Operator("sub", 2, (ExprType.PER_ELEMENT, ExprType.PER_ELEMENT), ExprType.PER_ELEMENT, torch.sub, 1.0),
            Operator("mul", 2, (ExprType.PER_ELEMENT, ExprType.PER_ELEMENT), ExprType.PER_ELEMENT, torch.mul, 1.0),
        ]
        
        # Division available at level 2 for all scopes
        if level >= 2:
            self.per_elem_binary.extend([
                Operator("safe_div", 2, (ExprType.PER_ELEMENT, ExprType.PER_ELEMENT), ExprType.PER_ELEMENT, safe_div, 2.0),
            ])
        
        # =====================================================================
        # Reduce operators (per-element -> scalar)
        # =====================================================================
        self.reduce_ops = [
            Operator("SUM", 1, (ExprType.PER_ELEMENT,), ExprType.SCALAR, 
                     lambda e, m: masked_sum(e, m), 1.0),
            Operator("MEAN", 1, (ExprType.PER_ELEMENT,), ExprType.SCALAR,
                     lambda e, m: masked_mean(e, m), 1.0),
            # Weighted reductions available in all scopes
            Operator("WSUM", 2, (ExprType.PER_ELEMENT, ExprType.PER_ELEMENT), ExprType.SCALAR,
                     lambda v, w, m: masked_weighted_sum(v, w, m), 2.0),
            Operator("WMEAN", 2, (ExprType.PER_ELEMENT, ExprType.PER_ELEMENT), ExprType.SCALAR,
                     lambda v, w, m: masked_weighted_mean(v, w, m), 2.0),
        ]
        
        # Full scope: add LOGSUMEXP, MAX at level 3
        if scope == "full" and level >= 3:
            self.reduce_ops.extend([
                Operator("LOGSUMEXP", 1, (ExprType.PER_ELEMENT,), ExprType.SCALAR,
                         lambda e, m: masked_logsumexp(e, m), 2.0),
                Operator("MAX", 1, (ExprType.PER_ELEMENT,), ExprType.SCALAR,
                         lambda e, m: masked_max(e, m), 2.0),
            ])
        
        # =====================================================================
        # Scalar binary operators (scalar, scalar -> scalar)
        # =====================================================================
        self.scalar_binary = [
            Operator("add_s", 2, (ExprType.SCALAR, ExprType.SCALAR), ExprType.SCALAR, torch.add, 1.0),
            Operator("sub_s", 2, (ExprType.SCALAR, ExprType.SCALAR), ExprType.SCALAR, torch.sub, 1.0),
            Operator("mul_s", 2, (ExprType.SCALAR, ExprType.SCALAR), ExprType.SCALAR, torch.mul, 1.0),
        ]
        
        if level >= 2:
            self.scalar_binary.extend([
                Operator("safe_div_s", 2, (ExprType.SCALAR, ExprType.SCALAR), ExprType.SCALAR, safe_div, 2.0),
            ])
        
        # =====================================================================
        # Scalar unary operators
        # =====================================================================
        self.scalar_unary = [
            Operator("identity_s", 1, (ExprType.SCALAR,), ExprType.SCALAR, lambda x: x, 0.0),
            Operator("square_s", 1, (ExprType.SCALAR,), ExprType.SCALAR, lambda x: x ** 2, 1.0),
        ]
        
        if scope in ("intermediate", "full") and level >= 2:
            self.scalar_unary.extend([
                Operator("safe_log_s", 1, (ExprType.SCALAR,), ExprType.SCALAR, safe_log, 2.0),
                Operator("safe_sqrt_s", 1, (ExprType.SCALAR,), ExprType.SCALAR, safe_sqrt, 2.0),
            ])
    
    def _build_terminals(self):
        """Build terminal symbols (per-galaxy features + constants)."""
        self.terminals = []
        for i, name in enumerate(self.feature_names):
            self.terminals.append({
                "name": name,
                "type": ExprType.PER_ELEMENT,
                "index": i,
            })
        # mask is always available
        self.terminals.append({
            "name": "m",
            "type": ExprType.PER_ELEMENT,
            "index": -1,  # special: mask
        })
    
    def get_all_operators(self) -> List[Operator]:
        return (
            self.per_elem_unary + self.per_elem_binary +
            self.reduce_ops +
            self.scalar_unary + self.scalar_binary
        )


# =============================================================================
# Expression tree
# =============================================================================

@dataclass
class ExprNode:
    """A node in the expression tree."""
    op: Optional[Operator] = None
    terminal: Optional[Dict] = None
    constant: Optional[float] = None
    const_type: Optional[ExprType] = None
    children: List["ExprNode"] = field(default_factory=list)
    
    @property
    def is_terminal(self) -> bool:
        return self.terminal is not None or self.constant is not None
    
    @property
    def output_type(self) -> ExprType:
        if self.terminal is not None:
            return self.terminal["type"]
        if self.constant is not None:
            return self.const_type or ExprType.SCALAR
        return self.op.output_type
    
    def evaluate(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the expression tree.
        
        X: (batch, max_galaxies, n_features)
        mask: (batch, max_galaxies)
        returns: (batch,) for SCALAR type, (batch, max_galaxies) for PER_ELEMENT type
        """
        if self.constant is not None:
            batch_size = X.shape[0]
            ctype = self.const_type or ExprType.SCALAR
            if ctype == ExprType.PER_ELEMENT:
                n = X.shape[1]
                return torch.full((batch_size, n), self.constant, device=X.device, dtype=X.dtype)
            return torch.full((batch_size,), self.constant, device=X.device, dtype=X.dtype)
        
        if self.terminal is not None:
            idx = self.terminal["index"]
            if idx == -1:
                return mask  # mask terminal
            else:
                return X[:, :, idx]  # feature terminal
        
        # Operator node
        child_values = [c.evaluate(X, mask) for c in self.children]
        
        if self.op.name in ["SUM", "MEAN", "LOGSUMEXP", "MAX", "WSUM", "WMEAN"]:
            # Reduce operators need mask
            if self.op.arity == 1:
                return self.op.func(child_values[0], mask)
            if self.op.arity == 2:
                return self.op.func(child_values[0], child_values[1], mask)
            raise ValueError(f"Unsupported reduce arity: {self.op.arity} for {self.op.name}")
        else:
            return self.op.func(*child_values)
    
    def to_string(self) -> str:
        """Convert expression tree to string representation."""
        if self.constant is not None:
            return f"{self.constant:.4f}"
        if self.terminal is not None:
            return self.terminal["name"]
        
        child_strs = [c.to_string() for c in self.children]
        if self.op.arity == 1:
            return f"{self.op.name}({child_strs[0]})"
        elif self.op.arity == 2:
            if self.op.name in ["add", "add_s"]:
                return f"({child_strs[0]} + {child_strs[1]})"
            elif self.op.name in ["sub", "sub_s"]:
                return f"({child_strs[0]} - {child_strs[1]})"
            elif self.op.name in ["mul", "mul_s"]:
                return f"({child_strs[0]} * {child_strs[1]})"
            elif self.op.name in ["safe_div", "safe_div_s"]:
                return f"({child_strs[0]} / {child_strs[1]})"
            else:
                return f"{self.op.name}({', '.join(child_strs)})"
        return f"{self.op.name}({', '.join(child_strs)})"
    
    def complexity(self) -> float:
        """Compute total complexity of the expression."""
        if self.is_terminal:
            return 0.0 if self.terminal else 0.5  # constants have small cost
        return self.op.complexity + sum(c.complexity() for c in self.children)
    
    def depth(self) -> int:
        """Compute depth of the expression tree."""
        if self.is_terminal:
            return 1
        return 1 + max(c.depth() for c in self.children)


# =============================================================================
# Program sampler (for DSR-style search)
# =============================================================================

class ProgramSampler:
    """
    Samples valid expression trees from the grammar.
    Used for initialization and mutation in the search.
    """
    
    def __init__(self, grammar: Grammar, max_depth: int = 5):
        self.grammar = grammar
        self.max_depth = max_depth
    
    def sample_program(self, target_type: ExprType = ExprType.SCALAR) -> ExprNode:
        """Sample a random valid program that outputs the target type."""
        return self._sample_node(target_type, depth=0)
    
    def _sample_node(self, target_type: ExprType, depth: int) -> ExprNode:
        """Recursively sample a node of the given type."""
        
        # If at max depth, must return a terminal or reduce
        if depth >= self.max_depth:
            if target_type == ExprType.SCALAR:
                # Must use a reduce operator or constant
                if random.random() < 0.3:
                    return ExprNode(constant=random.uniform(-2, 2), const_type=ExprType.SCALAR)
                # Sample a reduce op with a simple per-element child
                reduce_op = random.choice(self.grammar.reduce_ops)
                child = self._sample_terminal(ExprType.PER_ELEMENT)
                return ExprNode(op=reduce_op, children=[child])
            else:
                return self._sample_terminal(target_type)
        
        # Decide: terminal, constant, or operator
        choice = random.random()
        
        if target_type == ExprType.SCALAR:
            if choice < 0.1:
                # Constant
                return ExprNode(constant=random.uniform(-2, 2), const_type=ExprType.SCALAR)
            elif choice < 0.4:
                # Reduce operator (per_elem -> scalar)
                reduce_op = random.choice(self.grammar.reduce_ops)
                child = self._sample_node(ExprType.PER_ELEMENT, depth + 1)
                return ExprNode(op=reduce_op, children=[child])
            elif choice < 0.7:
                # Scalar binary
                op = random.choice(self.grammar.scalar_binary)
                children = [self._sample_node(ExprType.SCALAR, depth + 1) for _ in range(2)]
                return ExprNode(op=op, children=children)
            else:
                # Scalar unary
                op = random.choice(self.grammar.scalar_unary)
                child = self._sample_node(ExprType.SCALAR, depth + 1)
                return ExprNode(op=op, children=[child])
        
        else:  # PER_ELEMENT
            if choice < 0.3:
                return self._sample_terminal(target_type)
            elif choice < 0.4:
                # Per-element constant (broadcasted)
                return ExprNode(constant=random.uniform(-2, 2), const_type=ExprType.PER_ELEMENT)
            elif choice < 0.6:
                # Per-element unary
                op = random.choice(self.grammar.per_elem_unary)
                child = self._sample_node(ExprType.PER_ELEMENT, depth + 1)
                return ExprNode(op=op, children=[child])
            else:
                # Per-element binary
                op = random.choice(self.grammar.per_elem_binary)
                children = [self._sample_node(ExprType.PER_ELEMENT, depth + 1) for _ in range(2)]
                return ExprNode(op=op, children=children)
    
    def _sample_terminal(self, target_type: ExprType) -> ExprNode:
        """Sample a terminal of the given type."""
        valid_terminals = [t for t in self.grammar.terminals if t["type"] == target_type]
        if not valid_terminals:
            # Fallback: return a constant
            return ExprNode(constant=random.uniform(-2, 2), const_type=target_type)
        terminal = random.choice(valid_terminals)
        return ExprNode(terminal=terminal)


# =============================================================================
# Set-DSR Model (K symbolic summaries + MLP head)
# =============================================================================

class SetDSR(nn.Module):
    """
    Set-DSR model for learning K symbolic summary statistics.
    
    Architecture:
    1. K symbolic programs (expression trees) that map (X, mask) -> scalar
    2. Small MLP that maps [s_1, ..., s_K] -> parameters
    """
    
    def __init__(
        self,
        n_features: int = 1,
        feature_names: Optional[List[str]] = None,
        n_summaries: int = 8,
        n_params: int = 5,
        mlp_hidden: List[int] = [64, 64],
        curriculum_level: int = 1,
        max_depth: int = 5,
        operator_scope: str = "full",  # "simple", "intermediate", "full"
    ):
        super().__init__()
        
        self.n_features = n_features
        self.n_summaries = n_summaries
        self.n_params = n_params
        self.curriculum_level = curriculum_level
        self.operator_scope = operator_scope
        
        # Grammar
        self.grammar = Grammar(
            n_features=n_features,
            feature_names=feature_names,
            curriculum_level=curriculum_level,
            operator_scope=operator_scope,
        )
        
        # Program sampler
        self.sampler = ProgramSampler(self.grammar, max_depth=max_depth)
        
        # Initialize K programs (expression trees)
        self.programs: List[ExprNode] = [
            self.sampler.sample_program(ExprType.SCALAR)
            for _ in range(n_summaries)
        ]
        
        # MLP head: summaries -> parameters
        layers = []
        in_dim = n_summaries
        for hidden_dim in mlp_hidden:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_params))
        self.mlp_head = nn.Sequential(*layers)
    
    def compute_summaries(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute K symbolic summary statistics.
        
        X: (batch, max_galaxies, n_features)
        mask: (batch, max_galaxies)
        returns: (batch, K)
        """
        summaries = []
        for prog in self.programs:
            try:
                s = prog.evaluate(X, mask)
                # Ensure scalar output
                if s.dim() > 1:
                    s = s.mean(dim=-1)
                summaries.append(s)
            except Exception:
                # Fallback: return zeros
                summaries.append(torch.zeros(X.shape[0], device=X.device))
        
        return torch.stack(summaries, dim=1)
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        X: (batch, max_galaxies, n_features)
        mask: (batch, max_galaxies)
        returns: (batch, n_params)
        """
        summaries = self.compute_summaries(X, mask)
        # Handle NaN/Inf
        summaries = torch.nan_to_num(summaries, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.mlp_head(summaries)
    
    def get_program_strings(self) -> List[str]:
        """Return string representations of all programs."""
        return [p.to_string() for p in self.programs]
    
    def total_complexity(self) -> float:
        """Return total complexity of all programs."""
        return sum(p.complexity() for p in self.programs)
    
    def set_curriculum_level(self, level: int):
        """Update curriculum level (rebuilds grammar)."""
        self.curriculum_level = level
        self.grammar = Grammar(
            n_features=self.n_features,
            feature_names=self.grammar.feature_names,
            curriculum_level=level,
            operator_scope=self.operator_scope,
        )
        self.sampler = ProgramSampler(self.grammar)
    
    def parameterize_constants(self) -> "ParameterizedSetDSR":
        """
        Convert this model to a ParameterizedSetDSR where all constants
        become trainable nn.Parameters for gradient-based optimization.
        """
        return ParameterizedSetDSR(self)


# =============================================================================
# Parameterized Set-DSR (learnable constants via backprop)
# =============================================================================

class ParameterizedSetDSR(nn.Module):
    """
    Wrapper that makes all constants in the symbolic programs trainable.
    
    After GP/RL finds good program structures, this wrapper:
    1. Extracts all constants from the expression trees
    2. Converts them to nn.Parameters
    3. Evaluates programs using these parameters (so gradients flow)
    
    This enables expressions like WSUM(logM, a*logM + b) where a,b are trained.
    """
    
    def __init__(self, base_model: SetDSR):
        super().__init__()
        self.base_model = base_model
        self.n_summaries = base_model.n_summaries
        self.n_params = base_model.n_params
        
        # Copy the MLP head
        self.mlp_head = base_model.mlp_head
        
        # Extract and register constants as parameters
        self.program_params = nn.ParameterList()
        self.param_info = []  # List of (program_idx, node_path, const_type)
        
        for prog_idx, prog in enumerate(base_model.programs):
            self._extract_constants(prog, prog_idx, path=[])
    
    def _extract_constants(self, node: ExprNode, prog_idx: int, path: List[int]):
        """Recursively extract constants and register as parameters."""
        if node.constant is not None:
            # Register as parameter
            param = nn.Parameter(torch.tensor(node.constant, dtype=torch.float32))
            self.program_params.append(param)
            self.param_info.append({
                "prog_idx": prog_idx,
                "path": path.copy(),
                "const_type": node.const_type or ExprType.SCALAR,
            })
        
        for i, child in enumerate(node.children):
            self._extract_constants(child, prog_idx, path + [i])
    
    def _get_node_at_path(self, prog: ExprNode, path: List[int]) -> ExprNode:
        """Navigate to node at given path."""
        node = prog
        for idx in path:
            node = node.children[idx]
        return node
    
    def compute_summaries(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute summaries using parameterized constants."""
        summaries = []
        
        for prog_idx, prog in enumerate(self.base_model.programs):
            try:
                s = self._evaluate_parameterized(prog, prog_idx, X, mask)
                if s.dim() > 1:
                    s = s.mean(dim=-1)
                summaries.append(s)
            except Exception:
                summaries.append(torch.zeros(X.shape[0], device=X.device))
        
        return torch.stack(summaries, dim=1)
    
    def _evaluate_parameterized(
        self, node: ExprNode, prog_idx: int, X: torch.Tensor, mask: torch.Tensor,
        path: List[int] = None
    ) -> torch.Tensor:
        """Evaluate with parameterized constants."""
        if path is None:
            path = []
        
        # Check if this node is a parameterized constant
        if node.constant is not None:
            # Find the corresponding parameter
            for i, info in enumerate(self.param_info):
                if info["prog_idx"] == prog_idx and info["path"] == path:
                    param_val = self.program_params[i]
                    batch_size = X.shape[0]
                    ctype = info["const_type"]
                    if ctype == ExprType.PER_ELEMENT:
                        n = X.shape[1]
                        return param_val.expand(batch_size, n)
                    return param_val.expand(batch_size)
            # Fallback: use original constant
            batch_size = X.shape[0]
            ctype = node.const_type or ExprType.SCALAR
            if ctype == ExprType.PER_ELEMENT:
                n = X.shape[1]
                return torch.full((batch_size, n), node.constant, device=X.device, dtype=X.dtype)
            return torch.full((batch_size,), node.constant, device=X.device, dtype=X.dtype)
        
        if node.terminal is not None:
            idx = node.terminal["index"]
            if idx == -1:
                return mask
            return X[:, :, idx]
        
        # Operator node
        child_values = [
            self._evaluate_parameterized(c, prog_idx, X, mask, path + [i])
            for i, c in enumerate(node.children)
        ]
        
        if node.op.name in ["SUM", "MEAN", "LOGSUMEXP", "MAX", "WSUM", "WMEAN"]:
            if node.op.arity == 1:
                return node.op.func(child_values[0], mask)
            if node.op.arity == 2:
                return node.op.func(child_values[0], child_values[1], mask)
            raise ValueError(f"Unsupported reduce arity: {node.op.arity}")
        else:
            return node.op.func(*child_values)
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        summaries = self.compute_summaries(X, mask)
        summaries = torch.nan_to_num(summaries, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.mlp_head(summaries)
    
    def get_program_strings(self) -> List[str]:
        """Return program strings with current parameter values."""
        # Update constants in base model with current param values
        for i, info in enumerate(self.param_info):
            prog = self.base_model.programs[info["prog_idx"]]
            node = self._get_node_at_path(prog, info["path"])
            node.constant = self.program_params[i].item()
        return self.base_model.get_program_strings()
    
    def total_complexity(self) -> float:
        return self.base_model.total_complexity()
    
    def num_learnable_constants(self) -> int:
        return len(self.program_params)


# =============================================================================
# Genetic Programming Search for Set-DSR
# =============================================================================

class SetDSREvolver:
    """
    Evolutionary search for Set-DSR programs.
    
    Uses genetic programming to search for K symbolic summary statistics
    that minimize prediction loss + complexity penalty.
    
    Fitness evaluation uses quick linear fit (least-squares) for speed.
    MLP is retrained periodically for better accuracy.
    """
    
    def __init__(
        self,
        model: SetDSR,
        population_size: int = 100,
        elite_size: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        mlp_retrain_interval: int = 10,  # Retrain MLP every N generations
        mlp_retrain_epochs: int = 20,    # Epochs per retrain
        mlp_retrain_lr: float = 1e-3,
    ):
        self.model = model
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mlp_retrain_interval = mlp_retrain_interval
        self.mlp_retrain_epochs = mlp_retrain_epochs
        self.mlp_retrain_lr = mlp_retrain_lr
        self.generation_count = 0
        
        # Population: list of program lists (each individual has K programs)
        self.population: List[List[ExprNode]] = []
        self._init_population()
    
    def _init_population(self):
        """Initialize population with random programs."""
        self.population = []
        for _ in range(self.population_size):
            individual = [
                self.model.sampler.sample_program(ExprType.SCALAR)
                for _ in range(self.model.n_summaries)
            ]
            self.population.append(individual)
    
    def mutate(self, programs: List[ExprNode]) -> List[ExprNode]:
        """Mutate one or more programs in the individual."""
        new_programs = programs.copy()
        for i in range(len(new_programs)):
            if random.random() < self.mutation_rate:
                # Replace with new random program
                new_programs[i] = self.model.sampler.sample_program(ExprType.SCALAR)
        return new_programs
    
    def crossover(self, parent1: List[ExprNode], parent2: List[ExprNode]) -> List[ExprNode]:
        """Crossover: swap some programs between parents."""
        child = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child
    
    def evaluate_fitness(
        self,
        individual: List[ExprNode],
        X: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        complexity_weight: float = 0.01,
    ) -> float:
        """
        Evaluate fitness of an individual using quick linear fit.
        
        Uses least-squares to fit summaries to targets (fast O(K²) solve).
        Lower is better.
        """
        # Temporarily set programs
        old_programs = self.model.programs
        self.model.programs = individual
        
        try:
            with torch.no_grad():
                # Compute summaries only (skip MLP)
                S = self.model.compute_summaries(X, mask)  # [B, K]
                S = torch.nan_to_num(S, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Quick linear fit: solve S @ W = y in least-squares sense
                # Add bias term
                ones = torch.ones(S.shape[0], 1, device=S.device, dtype=S.dtype)
                S_aug = torch.cat([S, ones], dim=1)  # [B, K+1]
                
                # Solve via pseudo-inverse: W = (S^T S)^{-1} S^T y
                try:
                    # Use lstsq for numerical stability
                    W = torch.linalg.lstsq(S_aug, y).solution  # [K+1, P]
                    pred = S_aug @ W  # [B, P]
                    mse = F.mse_loss(pred, y).item()
                except Exception:
                    # Fallback: use MLP
                    pred = self.model.mlp_head(S)
                    mse = F.mse_loss(pred, y).item()
            
            # Complexity penalty
            complexity = sum(p.complexity() for p in individual)
            fitness = mse + complexity_weight * complexity
            
        except Exception:
            fitness = float("inf")
        
        self.model.programs = old_programs
        return fitness
    
    def retrain_mlp(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
    ):
        """Retrain MLP head with current best programs."""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.mlp_head.parameters(), lr=self.mlp_retrain_lr)
        criterion = nn.MSELoss()
        
        # Precompute summaries
        with torch.no_grad():
            S = self.model.compute_summaries(X, mask)
            S = torch.nan_to_num(S, nan=0.0, posinf=1e6, neginf=-1e6)
        
        for _ in range(self.mlp_retrain_epochs):
            optimizer.zero_grad()
            pred = self.model.mlp_head(S)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        
        self.model.eval()
    
    def evolve_generation(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        complexity_weight: float = 0.01,
    ) -> Tuple[List[ExprNode], float]:
        """
        Run one generation of evolution.
        
        Returns: (best individual, best fitness)
        """
        # Evaluate all individuals
        fitness_scores = [
            self.evaluate_fitness(ind, X, mask, y, complexity_weight)
            for ind in self.population
        ]
        
        # Sort by fitness (lower is better)
        sorted_indices = np.argsort(fitness_scores)
        sorted_pop = [self.population[i] for i in sorted_indices]
        sorted_fitness = [fitness_scores[i] for i in sorted_indices]
        
        # Keep elites
        new_population = sorted_pop[:self.elite_size]
        
        # Fill rest with offspring
        while len(new_population) < self.population_size:
            # Tournament selection
            p1 = self._tournament_select(sorted_pop, sorted_fitness)
            p2 = self._tournament_select(sorted_pop, sorted_fitness)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self.crossover(p1, p2)
            else:
                child = p1.copy()
            
            # Mutation
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        
        # Periodic MLP retraining with best individual
        self.generation_count += 1
        if self.mlp_retrain_interval > 0 and self.generation_count % self.mlp_retrain_interval == 0:
            # Use best individual for MLP retrain
            self.model.programs = sorted_pop[0]
            self.retrain_mlp(X, mask, y)
        
        return sorted_pop[0], sorted_fitness[0]
    
    def _tournament_select(
        self,
        population: List[List[ExprNode]],
        fitness: List[float],
        k: int = 3,
    ) -> List[ExprNode]:
        """Tournament selection."""
        indices = random.sample(range(len(population)), min(k, len(population)))
        best_idx = min(indices, key=lambda i: fitness[i])
        return population[best_idx]


# =============================================================================
# Utility: extract and simplify final expressions
# =============================================================================

def extract_expressions(model: SetDSR) -> Dict[str, str]:
    """Extract symbolic expressions from the model."""
    return {
        f"s_{i}": prog.to_string()
        for i, prog in enumerate(model.programs)
    }


def print_model_summary(model: SetDSR):
    """Print a summary of the model's symbolic programs."""
    print("=" * 60)
    print("Set-DSR Model Summary")
    print("=" * 60)
    print(f"Number of summaries (K): {model.n_summaries}")
    print(f"Total complexity: {model.total_complexity():.2f}")
    print("-" * 60)
    for i, prog in enumerate(model.programs):
        print(f"s_{i}: {prog.to_string()}")
        print(f"     complexity: {prog.complexity():.2f}, depth: {prog.depth()}")
    print("=" * 60)


# =============================================================================
# SymPy Simplification (optional)
# =============================================================================

def simplify_expression_sympy(expr_str: str) -> str:
    """
    Simplify an expression string using SymPy.
    
    Requires sympy to be installed. Returns original if simplification fails.
    """
    try:
        import sympy as sp
        
        # Replace our function names with sympy-compatible ones
        expr_str_clean = expr_str
        
        # Handle SUM, MEAN as symbols (they can't be simplified further)
        # We'll treat them as function calls
        replacements = {
            'SUM(': 'Function("SUM")(',
            'MEAN(': 'Function("MEAN")(',
            'LOGSUMEXP(': 'Function("LOGSUMEXP")(',
            'MAX(': 'Function("MAX_")(',  # MAX is reserved in sympy
            'safe_log(': 'log(',
            'safe_sqrt(': 'sqrt(',
            'safe_div(': 'Mul(',  # approximate
            'square(': 'Pow(',
            'abs(': 'Abs(',
        }
        
        for old, new in replacements.items():
            expr_str_clean = expr_str_clean.replace(old, new)
        
        # Try to parse and simplify
        try:
            # Define symbols that might appear
            symbols_dict = {}
            for name in ['logM', 'm', 'x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'N']:
                symbols_dict[name] = sp.Symbol(name)
            
            expr = sp.sympify(expr_str_clean, locals=symbols_dict)
            simplified = sp.simplify(expr)
            return str(simplified)
        except Exception:
            return expr_str
            
    except ImportError:
        # SymPy not installed
        return expr_str
    except Exception:
        return expr_str


def simplify_all_expressions(expressions: Dict[str, str]) -> Dict[str, str]:
    """Simplify all expressions in a dictionary."""
    return {
        name: simplify_expression_sympy(expr)
        for name, expr in expressions.items()
    }


def expressions_to_latex(expressions: Dict[str, str]) -> Dict[str, str]:
    """Convert expressions to LaTeX format using SymPy."""
    try:
        import sympy as sp
        
        latex_exprs = {}
        for name, expr_str in expressions.items():
            try:
                # Basic replacements for LaTeX
                latex = expr_str
                latex = latex.replace('SUM(', r'\sum_{i} \left(')
                latex = latex.replace('MEAN(', r'\frac{1}{N}\sum_{i} \left(')
                latex = latex.replace(')', r'\right)')
                latex = latex.replace('*', r' \cdot ')
                latex = latex.replace('logM', r'\log M_{\star,i}')
                latex = latex.replace('safe_log', r'\log')
                latex = latex.replace('safe_sqrt', r'\sqrt')
                latex = latex.replace('square', r'^2')
                latex_exprs[name] = latex
            except Exception:
                latex_exprs[name] = expr_str
        
        return latex_exprs
    except Exception:
        return expressions


# =============================================================================
# Tree utilities for subtree mutation
# =============================================================================

def get_all_nodes(node: ExprNode) -> List[Tuple[ExprNode, ExprNode, int]]:
    """
    Get all nodes in the tree with their parent and child index.
    Returns list of (node, parent, child_idx). Root has parent=None.
    """
    nodes = [(node, None, -1)]
    
    def _collect(n, parent, idx):
        for i, child in enumerate(n.children):
            nodes.append((child, n, i))
            _collect(child, n, i)
    
    _collect(node, None, -1)
    return nodes


def copy_tree(node: ExprNode) -> ExprNode:
    """Deep copy an expression tree."""
    if node.constant is not None:
        return ExprNode(constant=node.constant)
    if node.terminal is not None:
        return ExprNode(terminal=node.terminal.copy())
    
    new_children = [copy_tree(c) for c in node.children]
    return ExprNode(op=node.op, children=new_children)


def collect_constants(node: ExprNode) -> List[Tuple[ExprNode, ExprNode, int]]:
    """Collect all constant nodes with their parent and index."""
    constants = []
    
    def _collect(n, parent, idx):
        if n.constant is not None:
            constants.append((n, parent, idx))
        for i, child in enumerate(n.children):
            _collect(child, n, i)
    
    _collect(node, None, -1)
    return constants


# =============================================================================
# Advanced Evolver with subtree mutation and constant optimization
# =============================================================================

class AdvancedSetDSREvolver(SetDSREvolver):
    """
    Enhanced evolutionary search with:
    - Subtree mutation (preserves good partial solutions)
    - Constant optimization (gradient-based tuning of constants)
    - Quick linear fit for fitness evaluation
    - Periodic MLP retraining
    """
    
    def __init__(
        self,
        model: SetDSR,
        population_size: int = 100,
        elite_size: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        use_subtree_mutation: bool = True,
        use_constant_optimization: bool = True,
        const_opt_steps: int = 10,
        const_opt_lr: float = 0.1,
        mlp_retrain_interval: int = 10,
        mlp_retrain_epochs: int = 20,
        mlp_retrain_lr: float = 1e-3,
    ):
        super().__init__(
            model, population_size, elite_size, mutation_rate, crossover_rate,
            mlp_retrain_interval, mlp_retrain_epochs, mlp_retrain_lr
        )
        self.use_subtree_mutation = use_subtree_mutation
        self.use_constant_optimization = use_constant_optimization
        self.const_opt_steps = const_opt_steps
        self.const_opt_lr = const_opt_lr
    
    def subtree_mutate(self, program: ExprNode) -> ExprNode:
        """
        Subtree mutation: replace a random subtree with a new random subtree.
        Preserves the overall structure while exploring variations.
        """
        # Deep copy the program
        new_prog = copy_tree(program)
        
        # Get all nodes
        nodes = get_all_nodes(new_prog)
        if len(nodes) <= 1:
            # Only root, do full replacement
            return self.model.sampler.sample_program(ExprType.SCALAR)
        
        # Pick a random non-root node to replace
        candidates = [(n, p, i) for n, p, i in nodes if p is not None]
        if not candidates:
            return new_prog
        
        node, parent, child_idx = random.choice(candidates)
        
        # Generate a new subtree of the same output type
        target_type = node.output_type
        # Limit depth based on current depth in tree
        current_depth = self._get_depth_to_node(new_prog, node)
        max_subtree_depth = max(1, self.model.sampler.max_depth - current_depth)
        
        old_max_depth = self.model.sampler.max_depth
        self.model.sampler.max_depth = max_subtree_depth
        
        try:
            new_subtree = self.model.sampler.sample_program(target_type)
        finally:
            self.model.sampler.max_depth = old_max_depth
        
        # Replace the subtree
        parent.children[child_idx] = new_subtree
        
        return new_prog
    
    def _get_depth_to_node(self, root: ExprNode, target: ExprNode) -> int:
        """Get depth from root to target node."""
        def _search(node, depth):
            if node is target:
                return depth
            for child in node.children:
                result = _search(child, depth + 1)
                if result is not None:
                    return result
            return None
        
        result = _search(root, 0)
        return result if result is not None else 0
    
    def optimize_constants(
        self,
        programs: List[ExprNode],
        X: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
    ) -> List[ExprNode]:
        """
        Optimize constants in the programs using perturbation-based gradient estimation.
        """
        # Collect all constants from all programs
        all_constants = []
        for prog_idx, prog in enumerate(programs):
            constants = collect_constants(prog)
            for const_node, parent, child_idx in constants:
                all_constants.append({
                    'prog_idx': prog_idx,
                    'node': const_node,
                    'parent': parent,
                    'child_idx': child_idx,
                    'value': const_node.constant,
                })
        
        if not all_constants:
            return programs
        
        # Perturbation-based optimization (since tree eval isn't differentiable)
        eps = 0.01
        
        for step in range(self.const_opt_steps):
            # Evaluate current loss
            old_programs = self.model.programs
            self.model.programs = programs
            
            try:
                with torch.no_grad():
                    pred = self.model(X, mask)
                    current_loss = F.mse_loss(pred, y).item()
            except Exception:
                current_loss = float('inf')
            finally:
                self.model.programs = old_programs
            
            # Estimate gradient for each constant
            for c in all_constants:
                original_value = c['node'].constant
                
                # Positive perturbation
                c['node'].constant = original_value + eps
                self.model.programs = programs
                try:
                    with torch.no_grad():
                        pred = self.model(X, mask)
                        loss_plus = F.mse_loss(pred, y).item()
                except Exception:
                    loss_plus = float('inf')
                
                # Negative perturbation
                c['node'].constant = original_value - eps
                try:
                    with torch.no_grad():
                        pred = self.model(X, mask)
                        loss_minus = F.mse_loss(pred, y).item()
                except Exception:
                    loss_minus = float('inf')
                
                self.model.programs = old_programs
                
                # Gradient estimate
                grad = (loss_plus - loss_minus) / (2 * eps)
                
                # Update constant
                new_value = original_value - self.const_opt_lr * grad
                new_value = np.clip(new_value, -10, 10)  # Clamp
                c['node'].constant = float(new_value)
        
        return programs
    
    def mutate(self, programs: List[ExprNode]) -> List[ExprNode]:
        """Mutate programs using subtree mutation if enabled."""
        new_programs = []
        for prog in programs:
            if random.random() < self.mutation_rate:
                if self.use_subtree_mutation:
                    new_programs.append(self.subtree_mutate(prog))
                else:
                    # Full replacement (original behavior)
                    new_programs.append(self.model.sampler.sample_program(ExprType.SCALAR))
            else:
                new_programs.append(copy_tree(prog))
        return new_programs
    
    def evolve_generation(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        complexity_weight: float = 0.01,
    ) -> Tuple[List[ExprNode], float]:
        """
        Run one generation with optional constant optimization.
        """
        # Standard evolution
        best_ind, best_fitness = super().evolve_generation(X, mask, y, complexity_weight)
        
        # Optionally optimize constants for top individuals
        if self.use_constant_optimization:
            # Optimize constants for elite individuals
            for i in range(min(self.elite_size, len(self.population))):
                try:
                    self.population[i] = self.optimize_constants(
                        self.population[i], X, mask, y
                    )
                except Exception:
                    pass
            
            # Re-evaluate best individual after constant optimization
            best_ind = self.population[0]
            best_fitness = self.evaluate_fitness(best_ind, X, mask, y, complexity_weight)
        
        return best_ind, best_fitness


# =============================================================================
# RL-based Program Search (Policy Network)
# =============================================================================

class ProgramPolicyNetwork(nn.Module):
    """
    RNN-based policy network for generating symbolic programs.
    
    Generates programs token-by-token, learning which expressions
    tend to produce good results.
    """
    
    def __init__(
        self,
        grammar: Grammar,
        hidden_size: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.grammar = grammar
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Build token vocabulary
        self._build_vocabulary()
        
        # Embedding for tokens
        self.embedding = nn.Embedding(len(self.token_to_idx), hidden_size)
        
        # LSTM for sequential generation
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Output head for token prediction
        self.output_head = nn.Linear(hidden_size, len(self.token_to_idx))
        
        # Start token
        self.start_token_idx = self.token_to_idx['<START>']
        self.end_token_idx = self.token_to_idx['<END>']
    
    def _build_vocabulary(self):
        """Build token vocabulary from grammar."""
        tokens = ['<START>', '<END>', '<CONST>']
        
        # Add operators
        for op in self.grammar.get_all_operators():
            tokens.append(op.name)
        
        # Add terminals
        for term in self.grammar.terminals:
            tokens.append(term['name'])
        
        self.token_to_idx = {t: i for i, t in enumerate(tokens)}
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}
        self.vocab_size = len(tokens)
    
    def forward(self, token_seq: torch.Tensor, hidden=None):
        """
        Forward pass for training.
        
        token_seq: (batch, seq_len) token indices
        returns: (batch, seq_len, vocab_size) logits
        """
        embedded = self.embedding(token_seq)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.output_head(output)
        return logits, hidden
    
    def sample_program(
        self,
        max_length: int = 50,
        temperature: float = 1.0,
        device: torch.device = None,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Sample a program from the policy.
        
        Returns: (token_list, log_prob)
        """
        if device is None:
            device = next(self.parameters()).device
        
        tokens = []
        log_probs = []
        
        # Start with start token
        current_token = torch.tensor([[self.start_token_idx]], device=device)
        hidden = None
        
        for _ in range(max_length):
            logits, hidden = self.forward(current_token, hidden)
            logits = logits[:, -1, :] / temperature
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            sampled_idx = dist.sample()
            log_prob = dist.log_prob(sampled_idx)
            
            token = self.idx_to_token[sampled_idx.item()]
            tokens.append(token)
            log_probs.append(log_prob)
            
            if token == '<END>':
                break
            
            current_token = sampled_idx.unsqueeze(0).unsqueeze(0)
        
        total_log_prob = torch.stack(log_probs).sum()
        return tokens, total_log_prob
    
    def tokens_to_program(self, tokens: List[str]) -> Optional[ExprNode]:
        """
        Convert token sequence to expression tree.
        
        Uses a simple stack-based parser.
        """
        try:
            # Filter out start/end tokens
            tokens = [t for t in tokens if t not in ['<START>', '<END>']]
            
            if not tokens:
                return None
            
            # Simple recursive descent parser
            return self._parse_tokens(tokens, 0)[0]
        except Exception:
            return None
    
    def _parse_tokens(self, tokens: List[str], idx: int) -> Tuple[Optional[ExprNode], int]:
        """Recursive token parser."""
        if idx >= len(tokens):
            return None, idx
        
        token = tokens[idx]
        
        # Check if it's a constant
        if token == '<CONST>':
            return ExprNode(constant=random.uniform(-2, 2)), idx + 1
        
        # Check if it's a terminal
        for term in self.grammar.terminals:
            if term['name'] == token:
                return ExprNode(terminal=term), idx + 1
        
        # Check if it's an operator
        for op in self.grammar.get_all_operators():
            if op.name == token:
                children = []
                current_idx = idx + 1
                for _ in range(op.arity):
                    child, current_idx = self._parse_tokens(tokens, current_idx)
                    if child is None:
                        # Generate random child if parsing fails
                        child = self.grammar.terminals[0] if self.grammar.terminals else ExprNode(constant=1.0)
                        if isinstance(child, dict):
                            child = ExprNode(terminal=child)
                    children.append(child)
                return ExprNode(op=op, children=children), current_idx
        
        # Unknown token, return constant
        return ExprNode(constant=1.0), idx + 1


class RLProgramSearcher:
    """
    RL-based program search using REINFORCE.
    """
    
    def __init__(
        self,
        model: SetDSR,
        policy: ProgramPolicyNetwork,
        lr: float = 1e-3,
        baseline_decay: float = 0.99,
    ):
        self.model = model
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.baseline = 0.0
        self.baseline_decay = baseline_decay
    
    def search_step(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        n_samples: int = 10,
        complexity_weight: float = 0.01,
    ) -> Tuple[List[ExprNode], float]:
        """
        One step of RL-based search.
        
        Samples multiple programs, evaluates them, and updates policy.
        """
        device = X.device
        
        sampled_programs = []
        log_probs = []
        rewards = []
        
        for _ in range(n_samples):
            # Sample K programs for one individual
            individual = []
            individual_log_prob = 0.0
            
            for _ in range(self.model.n_summaries):
                tokens, log_prob = self.policy.sample_program(device=device)
                program = self.policy.tokens_to_program(tokens)
                
                if program is None:
                    # Fallback to random program
                    program = self.model.sampler.sample_program(ExprType.SCALAR)
                
                individual.append(program)
                individual_log_prob = individual_log_prob + log_prob
            
            sampled_programs.append(individual)
            log_probs.append(individual_log_prob)
            
            # Evaluate reward (negative loss)
            old_programs = self.model.programs
            self.model.programs = individual
            
            try:
                with torch.no_grad():
                    pred = self.model(X, mask)
                    mse = F.mse_loss(pred, y).item()
                complexity = sum(p.complexity() for p in individual)
                reward = -(mse + complexity_weight * complexity)
            except Exception:
                reward = -1e6
            finally:
                self.model.programs = old_programs
            
            rewards.append(reward)
        
        # Update baseline
        mean_reward = np.mean(rewards)
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * mean_reward
        
        # Compute policy gradient
        self.optimizer.zero_grad()
        
        policy_loss = 0.0
        for log_prob, reward in zip(log_probs, rewards):
            advantage = reward - self.baseline
            policy_loss = policy_loss - log_prob * advantage
        
        policy_loss = policy_loss / n_samples
        policy_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Return best program from this batch
        best_idx = np.argmax(rewards)
        return sampled_programs[best_idx], -rewards[best_idx]  # Return loss (lower is better)

"""
Simplified Set-DSR: Classical Summary Statistics with Learnable Weights

Pipeline:
1. Per-element symbolic transform g(x) via GP/RL
2. Learnable per-element weights w(x) trained by gradient descent
3. Classical summary statistics (moments, cumulants, quantiles) with optional weighting
4. Top-K feature selection
5. Final prediction head (MLP or linear)

This is a simplified, more interpretable alternative to full Set-DSR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import math
import time

# Import grammar and expression tree from set_dsr for per-element SR
from set_dsr import (
    Grammar, ExprNode, ExprType, Operator,
    safe_div, safe_log, safe_sqrt, clamped_exp,
    masked_sum, masked_mean, masked_weighted_sum, masked_weighted_mean,
)


# =============================================================================
# Classical Summary Statistics (Mask-Aware)
# =============================================================================

class SummaryStatistics(nn.Module):
    """
    Computes classical summary statistics over variable-length sets.
    
    All statistics are mask-aware and support optional per-element weights.
    
    Statistics computed:
    - Mean (weighted)
    - Variance (weighted)
    - Skewness (weighted)
    - Kurtosis (weighted)
    - Raw moments 1-4 (weighted)
    - Central moments 2-4 (weighted)
    - Cumulants 1-4 (weighted)
    - Quantiles (0.1, 0.25, 0.5, 0.75, 0.9) - approximate differentiable
    - Min, Max (soft versions)
    - Count (N_eff)
    """
    
    def __init__(
        self,
        include_moments: bool = True,
        include_cumulants: bool = True,
        include_quantiles: bool = False,
        quantile_probs: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        soft_quantile_temperature: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.include_moments = include_moments
        self.include_cumulants = include_cumulants
        self.include_quantiles = include_quantiles
        self.quantile_probs = quantile_probs
        self.soft_quantile_temperature = soft_quantile_temperature
        self.eps = eps
        
        # Build list of statistic names
        self.stat_names = self._build_stat_names()
    
    def _build_stat_names(self) -> List[str]:
        """Build ordered list of statistic names."""
        names = [
            # Basic stats
            "mean", "var", "std", "skew", "kurtosis",
            # Min/Max (soft)
            "min_soft", "max_soft",
            # Count
            "n_eff",
        ]
        
        if self.include_moments:
            names.extend([
                "moment_1", "moment_2", "moment_3", "moment_4",  # raw moments
                "central_moment_2", "central_moment_3", "central_moment_4",  # central moments
            ])
        
        if self.include_cumulants:
            names.extend([
                "cumulant_1", "cumulant_2", "cumulant_3", "cumulant_4",
            ])
        
        if self.include_quantiles:
            for p in self.quantile_probs:
                names.append(f"quantile_{p:.2f}")
        
        return names
    
    @property
    def n_stats(self) -> int:
        """Number of statistics computed per input feature."""
        return len(self.stat_names)
    
    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute all summary statistics.
        
        Args:
            values: (B, N) or (B, N, C) per-element values
            mask: (B, N) float mask (1 for valid, 0 for padding)
            weights: (B, N) or (B, N, K) optional per-element weights (will be normalized)
        
        Returns:
            stats: (B, C * K * n_stats) all computed statistics
        """
        if values.dim() == 2:
            values = values.unsqueeze(-1)
        B, N, C = values.shape
        device = values.device
        dtype = values.dtype
        eps = self.eps
        mask = mask.to(dtype)
        
        # Effective count
        n_eff = mask.sum(dim=1, keepdim=True) + eps  # (B, 1)
        
        # Normalize weights if provided
        if weights is not None:
            if weights.dim() == 2:
                # Apply mask and normalize
                w = weights * mask
                w_sum = w.sum(dim=1, keepdim=True) + eps
                w_norm = (w / w_sum).unsqueeze(-1)  # (B, N, 1)
            else:
                w = weights * mask.unsqueeze(-1)
                w_sum = w.sum(dim=1, keepdim=True) + eps  # (B, 1, K)
                w_norm = w / w_sum  # (B, N, K)
                w_norm = w_norm.unsqueeze(2)  # (B, N, 1, K)
        else:
            w_norm = (mask / n_eff).unsqueeze(-1)  # (B, N, 1)

        if w_norm.dim() == 3:
            w_norm = w_norm.unsqueeze(-1)  # (B, N, 1, 1)

        values_4d = values.unsqueeze(-1)  # (B, N, C, 1)
        mask_4d = mask.unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        K = w_norm.shape[-1]
        
        # =====================================================================
        # Basic statistics (weighted)
        # =====================================================================
        
        # Mean
        mean = (values_4d * w_norm).sum(dim=1)  # (B, C, K)
        
        # Centered values
        centered = (values_4d - mean.unsqueeze(1)) * mask_4d  # (B, N, C, K)
        
        # Variance
        var = (centered ** 2 * w_norm).sum(dim=1)  # (B, C, K)
        std = torch.sqrt(var + eps)
        
        # Skewness: E[(X - μ)³] / σ³
        m3 = (centered ** 3 * w_norm).sum(dim=1)
        skew = m3 / (std ** 3 + eps)
        
        # Kurtosis: E[(X - μ)⁴] / σ⁴ - 3 (excess kurtosis)
        m4 = (centered ** 4 * w_norm).sum(dim=1)
        kurtosis = m4 / (std ** 4 + eps) - 3.0
        
        # =====================================================================
        # Soft min/max (differentiable)
        # =====================================================================
        
        # Soft max: logsumexp trick
        neg_inf = -1e9
        masked_vals = values.masked_fill(mask.unsqueeze(-1) == 0, neg_inf)
        max_soft = torch.logsumexp(masked_vals / self.soft_quantile_temperature, dim=1) * self.soft_quantile_temperature
        
        # Soft min: -logsumexp(-x)
        masked_vals_neg = (-values).masked_fill(mask.unsqueeze(-1) == 0, neg_inf)
        min_soft = -torch.logsumexp(masked_vals_neg / self.soft_quantile_temperature, dim=1) * self.soft_quantile_temperature

        if K > 1:
            max_soft = max_soft.unsqueeze(-1).expand(-1, -1, K)
            min_soft = min_soft.unsqueeze(-1).expand(-1, -1, K)
        else:
            max_soft = max_soft.unsqueeze(-1)
            min_soft = min_soft.unsqueeze(-1)
        
        # Collect basic stats
        n_eff_b = n_eff.unsqueeze(-1).expand(-1, C, K)
        stats_list = [
            mean,
            var,
            std,
            skew,
            kurtosis,
            min_soft,
            max_soft,
            n_eff_b,
        ]
        
        # =====================================================================
        # Raw and central moments
        # =====================================================================
        
        if self.include_moments:
            # Raw moments E[X^k]
            moment_1 = mean
            moment_2 = (values_4d ** 2 * w_norm).sum(dim=1)
            moment_3 = (values_4d ** 3 * w_norm).sum(dim=1)
            moment_4 = (values_4d ** 4 * w_norm).sum(dim=1)
            
            # Central moments E[(X - μ)^k]
            central_moment_2 = var  # same as variance
            central_moment_3 = m3
            central_moment_4 = m4
            
            stats_list.extend([
                moment_1, moment_2, moment_3, moment_4,
                central_moment_2, central_moment_3, central_moment_4,
            ])
        
        # =====================================================================
        # Cumulants
        # =====================================================================
        
        if self.include_cumulants:
            # κ₁ = μ
            # κ₂ = σ²
            # κ₃ = μ₃ (third central moment)
            # κ₄ = μ₄ - 3μ₂² (fourth cumulant = excess kurtosis * σ⁴)
            kappa_1 = mean
            kappa_2 = var
            kappa_3 = m3
            kappa_4 = m4 - 3 * var ** 2
            
            stats_list.extend([kappa_1, kappa_2, kappa_3, kappa_4])
        
        # =====================================================================
        # Soft quantiles (differentiable approximation)
        # =====================================================================
        
        if self.include_quantiles:
            wq = w_norm.squeeze(2)  # (B, N, K) or (B, N, 1)
            if wq.dim() == 2:
                wq = wq.unsqueeze(-1)
            for p in self.quantile_probs:
                q = torch.empty((B, C, K), device=device, dtype=dtype)
                for c in range(C):
                    vals_c = values[:, :, c]
                    for k in range(K):
                        q[:, c, k] = self._soft_quantile(vals_c, mask, wq[:, :, k], p)
                stats_list.append(q)
        
        # Stack all stats
        stats = torch.stack(stats_list, dim=-1)  # (B, C, K, n_stats)
        
        # Replace NaN/Inf with zeros
        stats = torch.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)

        stats = stats.reshape(B, C * K * len(stats_list))
        
        return stats
    
    def _soft_quantile(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor,
        p: float,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        """
        Differentiable soft quantile approximation.
        
        Uses a soft-sorting approach: for each element, compute a soft indicator
        of whether it's below the quantile, then interpolate.
        
        Memory-efficient: processes pairwise comparisons in chunks to avoid
        O(N²) memory usage.
        
        Args:
            values: (B, N) values
            mask: (B, N) mask
            weights: (B, N) normalized weights
            p: quantile probability (0-1)
            chunk_size: size of chunks for memory-efficient computation
        
        Returns:
            quantile: (B,) approximate p-th quantile
        """
        B, N = values.shape
        temp = self.soft_quantile_temperature
        device = values.device
        dtype = values.dtype
        
        # Compute pairwise comparisons: how many values are <= each value?
        # For soft version, use sigmoid
        # F(x_i) ≈ Σⱼ w_j * σ((x_i - x_j) / temp)
        
        # Memory-efficient: compute CDF in chunks to avoid O(N²) memory
        # For each position i, we compute: cdf[i] = Σⱼ w_j * σ((v_i - v_j) / temp)
        cdf_vals = torch.zeros(B, N, device=device, dtype=dtype)
        
        # Process in chunks over the "i" dimension
        for i_start in range(0, N, chunk_size):
            i_end = min(i_start + chunk_size, N)
            chunk_len = i_end - i_start
            
            # v_i for this chunk: (B, chunk_len, 1)
            v_i_chunk = values[:, i_start:i_end].unsqueeze(2)
            
            # Compute against all j values: v_j is (B, 1, N)
            v_j = values.unsqueeze(1)
            
            # Soft indicator: σ((v_i - v_j) / temp) -> (B, chunk_len, N)
            soft_leq_chunk = torch.sigmoid((v_i_chunk - v_j) / (temp + self.eps))
            
            # Weighted sum over j: (B, chunk_len)
            cdf_chunk = (soft_leq_chunk * weights.unsqueeze(1)).sum(dim=2)
            cdf_vals[:, i_start:i_end] = cdf_chunk
        
        # Now we want the value where CDF = p
        # Use soft argmin on |CDF - p|
        diff = (cdf_vals - p).abs()
        
        # Soft argmin weights
        soft_weights = F.softmax(-diff / temp, dim=1) * mask  # (B, N)
        soft_weights = soft_weights / (soft_weights.sum(dim=1, keepdim=True) + self.eps)
        
        # Weighted average of values at soft argmin
        quantile = (values * soft_weights).sum(dim=1)
        
        return quantile


# =============================================================================
# Learnable Per-Element Weights
# =============================================================================

class LearnableWeights(nn.Module):
    """
    MLP that produces per-element weights w(x) for weighted statistics.
    
    These weights are trained via gradient descent to optimize downstream
    prediction performance.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [32, 16],
        n_kernels: int = 1,
        weight_type: str = "softmax",  # "softmax", "sigmoid", "raw"
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_kernels = n_kernels
        self.weight_type = weight_type
        self.temperature = temperature
        
        # Build MLP
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, n_kernels))  # output weight(s) per element
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute per-element weights.
        
        Args:
            X: (B, N, D) per-element features
            mask: (B, N) mask
        
        Returns:
            weights: (B, N, K) per-element weights (K = n_kernels)
        """
        B, N, D = X.shape
        
        # Compute raw weights
        raw_weights = self.mlp(X)  # (B, N, K)
        
        # Apply masking (set padding to very negative for softmax)
        if self.weight_type == "softmax":
            raw_weights = raw_weights / self.temperature
            neg_inf = torch.finfo(raw_weights.dtype).min
            raw_weights = raw_weights.masked_fill(mask.unsqueeze(-1) == 0, neg_inf)
            weights = F.softmax(raw_weights, dim=1)
        elif self.weight_type == "sigmoid":
            weights = torch.sigmoid(raw_weights) * mask.unsqueeze(-1)
        else:  # raw
            weights = raw_weights * mask.unsqueeze(-1)
        
        return weights


# =============================================================================
# Per-Element Symbolic Transform
# =============================================================================

class PerElementTransform(nn.Module):
    """
    Symbolic per-element transform g(x) using expression trees.
    
    This applies a symbolic expression to each element of the set,
    producing transformed values for downstream summary statistics.
    """
    
    def __init__(
        self,
        n_features: int = 1,
        n_transforms: int = 4,
        grammar: Optional[Grammar] = None,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_transforms = n_transforms
        
        # Initialize grammar for per-element operations only
        if grammar is None:
            self.grammar = Grammar(
                n_features=n_features,
                curriculum_level=2,
                operator_scope="intermediate",
            )
        else:
            self.grammar = grammar
        
        # Expression trees (one per transform)
        self.expressions: List[ExprNode] = []
        self._initialize_expressions()
    
    def _initialize_expressions(self):
        """Initialize with simple identity transforms."""
        for i in range(self.n_transforms):
            # Start with feature terminals
            feat_idx = i % self.n_features
            self.expressions.append(ExprNode(
                terminal=self.grammar.terminals[feat_idx]
            ))
    
    def set_expressions(self, expressions: List[ExprNode]):
        """Set expression trees (from GP/RL search)."""
        self.expressions = expressions
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply per-element transforms.
        
        Args:
            X: (B, N, D) input features
            mask: (B, N) mask
        
        Returns:
            transformed: (B, N, n_transforms) transformed values
        """
        B, N, D = X.shape
        outputs = []
        
        for expr in self.expressions:
            try:
                # Evaluate expression (should produce PER_ELEMENT type)
                out = expr.evaluate(X, mask)  # (B, N) or (B, N, D)
                if out.dim() == 3:
                    out = out.mean(dim=-1)  # reduce to (B, N)
                # Apply mask
                out = out * mask
            except Exception:
                out = torch.zeros(B, N, device=X.device, dtype=X.dtype)
            outputs.append(out)
        
        return torch.stack(outputs, dim=-1)  # (B, N, n_transforms)
    
    def sample_expression(self, max_depth: int = 3) -> ExprNode:
        """Sample a random per-element expression."""
        return self._sample_per_element(max_depth)
    
    def _sample_per_element(self, depth: int) -> ExprNode:
        """Sample a per-element expression tree."""
        if depth <= 1 or random.random() < 0.3:
            # Terminal
            terminal = random.choice(self.grammar.terminals)
            if terminal["type"] == ExprType.PER_ELEMENT:
                return ExprNode(terminal=terminal)
            else:
                # Use a feature terminal
                feat_terminals = [t for t in self.grammar.terminals 
                                  if t["type"] == ExprType.PER_ELEMENT and t["index"] >= 0]
                return ExprNode(terminal=random.choice(feat_terminals))
        
        # Choose operator
        if random.random() < 0.6:
            # Unary
            op = random.choice(self.grammar.per_elem_unary)
            child = self._sample_per_element(depth - 1)
            return ExprNode(op=op, children=[child])
        else:
            # Binary
            op = random.choice(self.grammar.per_elem_binary)
            left = self._sample_per_element(depth - 1)
            right = self._sample_per_element(depth - 1)
            return ExprNode(op=op, children=[left, right])
    
    def get_expression_strings(self) -> List[str]:
        """Get string representations of current expressions."""
        return [self._expr_to_string(e) for e in self.expressions]
    
    def _expr_to_string(self, node: ExprNode) -> str:
        """Convert expression tree to string."""
        if node.constant is not None:
            return f"{node.constant:.4f}"
        if node.terminal is not None:
            return node.terminal["name"]
        
        children_str = [self._expr_to_string(c) for c in node.children]
        if node.op.arity == 1:
            return f"{node.op.name}({children_str[0]})"
        else:
            return f"({children_str[0]} {node.op.name} {children_str[1]})"


# =============================================================================
# Top-K Feature Selection
# =============================================================================

class TopKSelector(nn.Module):
    """
    Select top-K features from summary statistics.
    
    Selection methods:
    - "correlation": Select by absolute correlation with target (requires target)
    - "variance": Select by variance (unsupervised)
    - "learnable": Learnable gating weights
    - "fixed": Fixed first K features
    """
    
    def __init__(
        self,
        n_input: int,
        k: int,
        method: str = "learnable",
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_input = n_input
        self.k = min(k, n_input)
        self.method = method
        self.temperature = temperature
        
        # For learnable selection
        if method == "learnable":
            self.gate_logits = nn.Parameter(torch.zeros(n_input))
        
        # Track selected indices (for non-learnable methods)
        self.register_buffer("selected_indices", torch.arange(self.k))
        self.register_buffer("selection_scores", torch.zeros(n_input))
    
    def compute_selection(
        self,
        features: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        """
        Compute which features to select (call once before training or periodically).
        
        Args:
            features: (N_samples, n_input) feature matrix
            targets: (N_samples, n_targets) target matrix (for correlation method)
        """
        if self.method == "correlation" and targets is not None:
            # Compute absolute correlation with each target, take max
            scores = []
            for i in range(features.shape[1]):
                feat = features[:, i]
                corrs = []
                for j in range(targets.shape[1]):
                    tgt = targets[:, j]
                    corr = torch.corrcoef(torch.stack([feat, tgt]))[0, 1]
                    corrs.append(corr.abs())
                scores.append(max(corrs))
            scores = torch.tensor(scores, device=features.device)
            
        elif self.method == "variance":
            scores = features.var(dim=0)
            
        elif self.method == "learnable":
            # Selection is done via soft gating during forward
            return
            
        else:  # fixed
            scores = torch.arange(self.n_input, 0, -1, device=features.device).float()
        
        # Store scores and get top-k indices
        self.selection_scores = scores
        _, indices = torch.topk(scores, self.k)
        self.selected_indices = indices.sort().values
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-K features.
        
        Args:
            features: (B, n_input) all features
        
        Returns:
            selected: (B, k) selected features
            selection_weights: (B, n_input) or (n_input,) selection weights/mask
        """
        if self.method == "learnable":
            # Soft top-k via Gumbel-softmax-like selection
            # Use straight-through for hard selection with soft gradients
            logits = self.gate_logits / self.temperature
            
            # Get soft weights
            soft_weights = F.softmax(logits, dim=0)
            
            # Hard top-k selection
            _, top_indices = torch.topk(logits, self.k)
            hard_mask = torch.zeros_like(logits)
            hard_mask[top_indices] = 1.0
            
            # Straight-through: hard forward, soft backward
            selection_weights = hard_mask - soft_weights.detach() + soft_weights
            
            # Select features
            selected = features[:, top_indices.sort().values]
            
            return selected, selection_weights
        
        else:
            # Hard selection based on precomputed indices
            selected = features[:, self.selected_indices]
            weights = torch.zeros(self.n_input, device=features.device)
            weights[self.selected_indices] = 1.0
            
            return selected, weights


# =============================================================================
# Final Prediction Head
# =============================================================================

class PredictionHead(nn.Module):
    """
    Final prediction head: either MLP or identity (for SR on top).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = [64, 32],
        use_mlp: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_mlp = use_mlp
        
        if use_mlp:
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, h_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            self.mlp = nn.Sequential(*layers)
        else:
            # Identity / linear only
            self.mlp = nn.Linear(input_dim, output_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict from selected features.
        
        Args:
            features: (B, input_dim) selected summary statistics
        
        Returns:
            predictions: (B, output_dim)
        """
        return self.mlp(features)


# =============================================================================
# Main Model: Simplified Set-DSR
# =============================================================================

class SimplifiedSetDSR(nn.Module):
    """
    Simplified Set-DSR model.
    
    Pipeline:
    1. Per-element symbolic transform g(x): X → g(X) (optional)
    2. Learnable weights w(x) for weighted statistics (optionally multiple kernels)
    3. Classical summary statistics on g(X) or raw features
    4. Top-K feature selection (optional)
    5. Final prediction head (MLP or linear)
    
    The per-element transforms are found via GP/RL search,
    while weights and prediction head are trained by gradient descent.
    """
    
    def __init__(
        self,
        n_features: int = 1,
        n_transforms: int = 4,
        output_dim: int = 6,
        top_k: int = 16,
        include_moments: bool = True,
        include_cumulants: bool = True,
        include_quantiles: bool = False,
        quantile_probs: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        use_learnable_weights: bool = True,
        weight_hidden_dims: List[int] = [32, 16],
        n_weight_kernels: int = 1,
        selection_method: str = "learnable",
        use_mlp_head: bool = True,
        head_hidden_dims: List[int] = [64, 32],
        dropout: float = 0.1,
        soft_quantile_temperature: float = 1.0,
        use_symbolic_transforms: bool = True,
        use_top_k: bool = True,
    ):
        super().__init__()
        
        self.n_features = n_features
        self.n_transforms = n_transforms
        self.output_dim = output_dim
        self.top_k = top_k
        self.use_learnable_weights = use_learnable_weights
        self.use_symbolic_transforms = use_symbolic_transforms
        self.use_top_k = use_top_k
        self.n_weight_kernels = max(1, int(n_weight_kernels))
        
        # 1. Per-element transform (optional)
        if use_symbolic_transforms:
            self.transform = PerElementTransform(
                n_features=n_features,
                n_transforms=n_transforms,
            )
            n_channels = n_transforms
        else:
            self.transform = None
            n_channels = n_features  # Use raw features directly
        
        # 2. Summary statistics
        self.summary_stats = SummaryStatistics(
            include_moments=include_moments,
            include_cumulants=include_cumulants,
            include_quantiles=include_quantiles,
            quantile_probs=quantile_probs,
            soft_quantile_temperature=soft_quantile_temperature,
        )
        
        # Total number of summary features = n_channels * n_stats * n_weight_kernels (if weights enabled)
        kernel_mult = self.n_weight_kernels if use_learnable_weights else 1
        self.n_summary_features = n_channels * self.summary_stats.n_stats * kernel_mult
        
        # 3. Learnable weights (optional)
        if use_learnable_weights:
            self.weight_net = LearnableWeights(
                input_dim=n_features,
                hidden_dims=weight_hidden_dims,
                n_kernels=self.n_weight_kernels,
            )
        else:
            self.weight_net = None
        
        # 4. Top-K selector (optional)
        if use_top_k:
            self.selector = TopKSelector(
                n_input=self.n_summary_features,
                k=top_k,
                method=selection_method,
            )
            head_input_dim = min(top_k, self.n_summary_features)
        else:
            self.selector = None
            head_input_dim = self.n_summary_features
        
        # 5. Prediction head
        self.head = PredictionHead(
            input_dim=head_input_dim,
            output_dim=output_dim,
            hidden_dims=head_hidden_dims,
            use_mlp=use_mlp_head,
            dropout=dropout,
        )
        
        # Store feature names for interpretability
        self._build_feature_names()
        self.profile_steps = False
        self.profile_stats = {}

    def enable_step_profiling(self, enabled: bool = True) -> None:
        self.profile_steps = enabled
        if enabled:
            self.reset_profile_stats()

    def reset_profile_stats(self) -> None:
        self.profile_stats = {
            "weights": 0.0,
            "transform": 0.0,
            "stats": 0.0,
            "selector": 0.0,
            "head": 0.0,
            "summary_total": 0.0,
            "batches": 0,
        }
    
    def _build_feature_names(self):
        """Build names for all summary features."""
        self.feature_names = []
        include_kernels = self.use_learnable_weights and self.n_weight_kernels > 1
        if self.use_symbolic_transforms:
            for t in range(self.n_transforms):
                if include_kernels:
                    for k in range(self.n_weight_kernels):
                        for stat_name in self.summary_stats.stat_names:
                            self.feature_names.append(f"g{t}_k{k}_{stat_name}")
                else:
                    for stat_name in self.summary_stats.stat_names:
                        self.feature_names.append(f"g{t}_{stat_name}")
        else:
            for f in range(self.n_features):
                if include_kernels:
                    for k in range(self.n_weight_kernels):
                        for stat_name in self.summary_stats.stat_names:
                            self.feature_names.append(f"f{f}_k{k}_{stat_name}")
                else:
                    for stat_name in self.summary_stats.stat_names:
                        self.feature_names.append(f"f{f}_{stat_name}")
    
    def compute_summary_features(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute all summary features from input.
        
        Args:
            X: (B, N, D) input features
            mask: (B, N) mask
        
        Returns:
            features: (B, n_summary_features) all summary statistics
            weights: (B, N) or (B, N, K) per-element weights (if learnable)
        """
        B, N, D = X.shape
        
        t0 = time.perf_counter() if self.profile_steps else None

        # Compute per-element weights
        if self.weight_net is not None:
            if self.profile_steps and X.is_cuda:
                torch.cuda.synchronize()
            t_w = time.perf_counter() if self.profile_steps else None
            weights = self.weight_net(X, mask)  # (B, N, K)
            if t_w is not None:
                if X.is_cuda:
                    torch.cuda.synchronize()
                self.profile_stats["weights"] += time.perf_counter() - t_w
        else:
            weights = None
        
        # Apply per-element transforms or use raw features
        if self.use_symbolic_transforms:
            if self.profile_steps and X.is_cuda:
                torch.cuda.synchronize()
            t_t = time.perf_counter() if self.profile_steps else None
            transformed = self.transform(X, mask)  # (B, N, n_transforms)
            if t_t is not None:
                if X.is_cuda:
                    torch.cuda.synchronize()
                self.profile_stats["transform"] += time.perf_counter() - t_t
            n_channels = self.n_transforms
        else:
            # Use raw features directly
            transformed = X  # (B, N, D)
            n_channels = D
        
        # Compute summary statistics for all channels/kernels
        if self.profile_steps and X.is_cuda:
            torch.cuda.synchronize()
        t_s = time.perf_counter() if self.profile_steps else None
        features = self.summary_stats(transformed, mask, weights)
        if t_s is not None:
            if X.is_cuda:
                torch.cuda.synchronize()
            self.profile_stats["stats"] += time.perf_counter() - t_s

        if t0 is not None:
            if X.is_cuda:
                torch.cuda.synchronize()
            self.profile_stats["summary_total"] += time.perf_counter() - t0
            self.profile_stats["batches"] += 1
        
        return features, weights
    
    def forward(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass.
        
        Args:
            X: (B, N, D) input features
            mask: (B, N) mask
        
        Returns:
            predictions: (B, output_dim)
        """
        # Compute all summary features
        features, _ = self.compute_summary_features(X, mask)
        
        # Select top-K features (if enabled)
        if self.use_top_k:
            if self.profile_steps and X.is_cuda:
                torch.cuda.synchronize()
            t_sel = time.perf_counter() if self.profile_steps else None
            selected, _ = self.selector(features)
            if t_sel is not None:
                if X.is_cuda:
                    torch.cuda.synchronize()
                self.profile_stats["selector"] += time.perf_counter() - t_sel
        else:
            selected = features
        
        # Predict
        if self.profile_steps and X.is_cuda:
            torch.cuda.synchronize()
        t_head = time.perf_counter() if self.profile_steps else None
        out = self.head(selected)
        if t_head is not None:
            if X.is_cuda:
                torch.cuda.synchronize()
            self.profile_stats["head"] += time.perf_counter() - t_head
        return out
    
    def get_selected_feature_names(self) -> List[str]:
        """Get names of selected features."""
        if not self.use_top_k:
            return self.feature_names
        
        if self.selector.method == "learnable":
            _, top_indices = torch.topk(self.selector.gate_logits, self.top_k)
            indices = top_indices.sort().values.cpu().numpy()
        else:
            indices = self.selector.selected_indices.cpu().numpy()
        
        return [self.feature_names[i] for i in indices]
    
    def get_transform_expressions(self) -> List[str]:
        """Get string representations of per-element transforms."""
        if not self.use_symbolic_transforms:
            return [f"x_{i}" for i in range(self.n_features)]
        return self.transform.get_expression_strings()


# =============================================================================
# Evolver for Per-Element Transforms
# =============================================================================

class TransformEvolver:
    """
    Evolutionary search for per-element transforms.
    
    Uses genetic programming to find good symbolic transforms g(x),
    while weights and prediction head are trained by gradient descent.
    """
    
    def __init__(
        self,
        model: SimplifiedSetDSR,
        population_size: int = 50,
        elite_size: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        max_depth: int = 4,
    ):
        self.model = model
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_depth = max_depth
        
        # Population: list of expression lists (one per individual)
        self.population: List[List[ExprNode]] = []
        self.fitness_scores: List[float] = []
        
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize random population."""
        self.population = []
        for _ in range(self.population_size):
            expressions = []
            for _ in range(self.model.n_transforms):
                expr = self.model.transform.sample_expression(self.max_depth)
                expressions.append(expr)
            self.population.append(expressions)
        self.fitness_scores = [0.0] * self.population_size
    
    def evaluate_fitness(
        self,
        X: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor,
        complexity_weight: float = 0.01,
    ) -> List[float]:
        """
        Evaluate fitness of all individuals.
        
        Uses linear regression (lstsq) for fast evaluation.
        """
        device = X.device
        B = X.shape[0]
        
        fitness_scores = []
        
        for expressions in self.population:
            try:
                # Set expressions
                self.model.transform.expressions = expressions
                
                # Compute summary features
                with torch.no_grad():
                    features, _ = self.model.compute_summary_features(X, mask)
                
                # Quick linear fit
                F_mat = features  # (B, n_features)
                
                # Add bias
                ones = torch.ones(B, 1, device=device)
                F_aug = torch.cat([F_mat, ones], dim=1)
                
                # Solve least squares
                solution = torch.linalg.lstsq(F_aug, y).solution
                
                # Compute R²
                pred = F_aug @ solution
                ss_res = ((y - pred) ** 2).sum()
                ss_tot = ((y - y.mean(dim=0)) ** 2).sum() + 1e-8
                r2 = 1 - ss_res / ss_tot
                
                # Complexity penalty
                complexity = sum(self._expr_complexity(e) for e in expressions)
                
                fitness = r2.item() - complexity_weight * complexity
                
            except Exception:
                fitness = -1e6
            
            fitness_scores.append(fitness)
        
        self.fitness_scores = fitness_scores
        return fitness_scores
    
    def _expr_complexity(self, node: ExprNode) -> float:
        """Compute complexity of expression tree."""
        if node.is_terminal:
            return 1.0
        complexity = node.op.complexity
        for child in node.children:
            complexity += self._expr_complexity(child)
        return complexity
    
    def evolve_generation(self) -> List[List[ExprNode]]:
        """Evolve one generation."""
        # Sort by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        # Elite selection
        new_population = []
        for i in range(self.elite_size):
            new_population.append(self.population[sorted_indices[i]])
        
        # Fill rest with crossover and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child = self._crossover(parent1, parent2)
            else:
                child = [self._copy_expr(e) for e in parent1]
            
            # Mutation
            if random.random() < self.mutation_rate:
                child = self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        return new_population
    
    def _tournament_select(self, k: int = 3) -> List[ExprNode]:
        """Tournament selection."""
        indices = random.sample(range(len(self.population)), k)
        best_idx = max(indices, key=lambda i: self.fitness_scores[i])
        return self.population[best_idx]
    
    def _crossover(
        self,
        parent1: List[ExprNode],
        parent2: List[ExprNode],
    ) -> List[ExprNode]:
        """Crossover: swap some expressions between parents."""
        child = []
        for e1, e2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child.append(self._copy_expr(e1))
            else:
                child.append(self._copy_expr(e2))
        return child
    
    def _mutate(self, expressions: List[ExprNode]) -> List[ExprNode]:
        """Mutate: replace one expression with a new random one."""
        idx = random.randint(0, len(expressions) - 1)
        expressions[idx] = self.model.transform.sample_expression(self.max_depth)
        return expressions
    
    def _copy_expr(self, node: ExprNode) -> ExprNode:
        """Deep copy an expression tree."""
        if node.is_terminal:
            return ExprNode(
                terminal=node.terminal,
                constant=node.constant,
                const_type=node.const_type,
            )
        return ExprNode(
            op=node.op,
            children=[self._copy_expr(c) for c in node.children],
        )
    
    def get_best(self) -> Tuple[List[ExprNode], float]:
        """Get best individual."""
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx], self.fitness_scores[best_idx]


# =============================================================================
# Training Utilities
# =============================================================================

def train_weights_and_head(
    model: SimplifiedSetDSR,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    n_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Dict[str, List[float]]:
    """
    Train learnable weights and prediction head by gradient descent.
    
    The per-element transforms are frozen during this phase.
    """
    model = model.to(device)
    
    # Only train weights and head parameters
    params = list(model.head.parameters())
    if model.weight_net is not None:
        params += list(model.weight_net.parameters())
    if model.selector.method == "learnable":
        params.append(model.selector.gate_logits)
    
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    
    history = {"train_loss": [], "val_loss": [], "val_r2": []}
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            X, mask, y = batch
            X, mask, y = X.to(device), mask.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(X, mask)
            loss = F.mse_loss(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        history["train_loss"].append(epoch_loss / n_batches)
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_preds, all_targets = [], []
            
            with torch.no_grad():
                for batch in val_loader:
                    X, mask, y = batch
                    X, mask, y = X.to(device), mask.to(device), y.to(device)
                    pred = model(X, mask)
                    val_loss += F.mse_loss(pred, y).item()
                    all_preds.append(pred)
                    all_targets.append(y)
            
            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)
            
            # Compute R²
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            ss_res = ((all_targets - all_preds) ** 2).sum()
            ss_tot = ((all_targets - all_targets.mean(dim=0)) ** 2).sum() + 1e-8
            r2 = (1 - ss_res / ss_tot).item()
            history["val_r2"].append(r2)
    
    return history


def train_simplified_dsr(
    model: SimplifiedSetDSR,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    n_generations: int = 50,
    n_weight_epochs: int = 20,
    complexity_weight: float = 0.01,
    log_interval: int = 10,
    weight_retrain_interval: int = 10,
) -> Dict:
    """
    Full training loop for Simplified Set-DSR.
    
    Alternates between:
    1. Evolving per-element transforms (GP search)
    2. Training weights and head (gradient descent)
    """
    model = model.to(device)
    
    # Concatenate all training data for GP evaluation
    all_X, all_mask, all_y = [], [], []
    for batch in train_loader:
        X, mask, y = batch
        all_X.append(X)
        all_mask.append(mask)
        all_y.append(y)
    
    X_train = torch.cat(all_X, dim=0).to(device)
    mask_train = torch.cat(all_mask, dim=0).to(device)
    y_train = torch.cat(all_y, dim=0).to(device)
    
    # Initialize evolver
    evolver = TransformEvolver(model)
    
    history = {
        "best_fitness": [],
        "mean_fitness": [],
        "best_expressions": [],
    }
    
    print(f"\n{'='*60}")
    print("Training Simplified Set-DSR")
    print(f"{'='*60}")
    print(f"Population size: {evolver.population_size}")
    print(f"Generations: {n_generations}")
    print(f"Transforms: {model.n_transforms}")
    print(f"Summary features: {model.n_summary_features}")
    print(f"Top-K: {model.top_k}")
    print(f"{'='*60}\n")
    
    for gen in range(n_generations):
        # Evaluate fitness
        fitness_scores = evolver.evaluate_fitness(
            X_train, mask_train, y_train,
            complexity_weight=complexity_weight,
        )
        
        best_expr, best_fitness = evolver.get_best()
        mean_fitness = np.mean(fitness_scores)
        
        history["best_fitness"].append(best_fitness)
        history["mean_fitness"].append(mean_fitness)
        
        # Log
        if gen % log_interval == 0:
            print(f"Gen {gen:3d} | Best: {best_fitness:.4f} | Mean: {mean_fitness:.4f}")
            
            # Show best expressions
            model.transform.expressions = best_expr
            expr_strs = model.get_transform_expressions()
            for i, e in enumerate(expr_strs[:3]):  # show first 3
                print(f"  g{i}: {e}")
        
        # Periodically retrain weights
        if gen > 0 and gen % weight_retrain_interval == 0:
            print(f"  → Retraining weights and head...")
            model.transform.expressions = best_expr
            train_weights_and_head(
                model, train_loader, val_loader, device,
                n_epochs=n_weight_epochs, lr=1e-3,
            )
        
        # Evolve
        evolver.evolve_generation()
    
    # Final weight training
    print("\nFinal weight training...")
    best_expr, best_fitness = evolver.get_best()
    model.transform.expressions = best_expr
    final_history = train_weights_and_head(
        model, train_loader, val_loader, device,
        n_epochs=n_weight_epochs * 2, lr=1e-3,
    )
    
    history["final_train_loss"] = final_history["train_loss"]
    history["final_val_r2"] = final_history.get("val_r2", [])
    history["best_expressions"] = model.get_transform_expressions()
    history["selected_features"] = model.get_selected_feature_names()
    
    print(f"\nTraining complete!")
    print(f"Best fitness: {best_fitness:.4f}")
    if final_history.get("val_r2"):
        print(f"Final val R²: {final_history['val_r2'][-1]:.4f}")
    print(f"\nSelected features: {history['selected_features']}")
    
    return history


# =============================================================================
# CLI-Compatible Wrapper (for use with train_dsr.py patterns)
# =============================================================================

def create_simplified_model(
    n_features: int = 1,
    n_transforms: int = 4,
    output_dim: int = 6,
    top_k: int = 16,
    include_moments: bool = True,
    include_cumulants: bool = True,
    include_quantiles: bool = False,
    use_learnable_weights: bool = True,
    n_weight_kernels: int = 1,
    selection_method: str = "learnable",
    use_mlp_head: bool = True,
    use_symbolic_transforms: bool = True,
    use_top_k: bool = True,
    **kwargs,
) -> SimplifiedSetDSR:
    """Factory function to create a SimplifiedSetDSR model."""
    return SimplifiedSetDSR(
        n_features=n_features,
        n_transforms=n_transforms,
        output_dim=output_dim,
        top_k=top_k,
        include_moments=include_moments,
        include_cumulants=include_cumulants,
        include_quantiles=include_quantiles,
        use_learnable_weights=use_learnable_weights,
        n_weight_kernels=n_weight_kernels,
        selection_method=selection_method,
        use_mlp_head=use_mlp_head,
        use_symbolic_transforms=use_symbolic_transforms,
        use_top_k=use_top_k,
        **kwargs,
    )

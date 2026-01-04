import torch
import torch.nn.functional as F
import numpy as np
import os
from typing import Callable, Optional, Dict, Any

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _forward_components(model, x, mask):
    """Run SlotSetPool forward but return intermediate tensors for analysis.

    Returns dict with keys: h, logm, k, v, q, logits, attn, slots, slots_refined, y
    """
    B, N, D = x.shape
    h = model.phi(x)  # (B,N,H)
    logm = x[..., model.logm_idx:model.logm_idx+1]
    k = model.key(torch.cat([h, logm], dim=-1))
    v = model.val(h)
    q = model.queries.unsqueeze(0).expand(B, -1, -1)
    scale = model.H ** 0.5
    logits = (q @ k.transpose(1, 2)) / scale
    logits = logits.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
    attn = F.softmax(logits, dim=-1)
    attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
    slots = attn @ v
    slots_refined = slots + model.slot_ff(slots)
    y = model.head(slots_refined.reshape(B, model.K * model.H))
    return dict(h=h, logm=logm, k=k, v=v, q=q, logits=logits, attn=attn,
                slots=slots, slots_refined=slots_refined, y=y)


def slots_ablation(model, x, mask, target: Optional[torch.Tensor]=None, loss_fn: Optional[Callable]=None):
    """Compute per-slot importance by zeroing each slot and measuring output change.

    Returns a dict with:
      - "baseline_y": (B, out_dim)
      - "slot_deltas": (K,) average absolute change in model output (or loss if target+loss_fn provided)
      - "per_sample": (B,K) per-sample delta magnitudes
    """
    device = next(model.parameters()).device
    x = x.to(device)
    mask = mask.to(device)
    comps = _forward_components(model, x, mask)
    baseline_y = comps['y']

    B = x.shape[0]
    K = model.K

    per_sample = torch.zeros(B, K, device=device)

    if target is not None and loss_fn is not None:
        target = target.to(device)
        baseline_loss = loss_fn(baseline_y, target).detach()
    else:
        baseline_loss = None

    for i in range(K):
        slots = comps['slots_refined'].clone()
        slots[:, i, :] = 0.0
        y_i = model.head(slots.reshape(B, K * model.H))
        if baseline_loss is not None:
            loss_i = loss_fn(y_i, target).detach()
            per_sample[:, i] = (loss_i - baseline_loss).abs()
        else:
            per_sample[:, i] = (y_i - baseline_y).abs().mean(dim=1)

    slot_deltas = per_sample.mean(dim=0).cpu().numpy()
    return dict(baseline_y=baseline_y.detach().cpu(), slot_deltas=slot_deltas, per_sample=per_sample.detach().cpu().numpy())


def component_ablation(model, x, mask, target: Optional[torch.Tensor]=None, loss_fn: Optional[Callable]=None):
    """Ablate high-level components and measure effect.

    Components tested: 'logm' (zero), 'attn_uniform' (replace attn with uniform over valid N),
    'phi_zero' (zero phi outputs), 'v_zero' (zero values), 'slot_ff_zero', 'queries_zero'.
    """
    device = next(model.parameters()).device
    x = x.to(device)
    mask = mask.to(device)
    comps = _forward_components(model, x, mask)
    baseline_y = comps['y']

    B, N, D = x.shape
    results: Dict[str, Any] = {}

    def eval_y_from_slots(slots_refined):
        return model.head(slots_refined.reshape(B, model.K * model.H))

    def measure(y):
        if target is not None and loss_fn is not None:
            return loss_fn(y, target).detach().cpu().item()
        else:
            return (y - baseline_y).abs().mean().detach().cpu().item()

    # logm zero
    x_logm_zero = x.clone()
    x_logm_zero[..., model.logm_idx] = 0.0
    y_logm = model(x_logm_zero, mask)[0]
    results['logm_zero'] = measure(y_logm)

    # attn uniform: replace attn with uniform over valid galaxies
    valid = mask.unsqueeze(1)  # (B,1,N)
    n_valid = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)  # (B,1,1)
    uniform_attn = valid / n_valid  # broadcast to (B,1,N) then expand
    uniform_attn = uniform_attn.expand(-1, model.K, -1)
    slots_uniform = uniform_attn @ comps['v']
    slots_uniform_refined = slots_uniform + model.slot_ff(slots_uniform)
    y_attn_uniform = eval_y_from_slots(slots_uniform_refined)
    results['attn_uniform'] = measure(y_attn_uniform)

    # phi zero
    h_zero = torch.zeros_like(comps['h'])
    k_zero = model.key(torch.cat([h_zero, comps['logm']], dim=-1))
    v_zero = model.val(h_zero)
    logits = (comps['q'] @ k_zero.transpose(1, 2)) / (model.H ** 0.5)
    logits = logits.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
    attn = F.softmax(logits, dim=-1)
    attn = torch.where(torch.isfinite(attn), attn, torch.zeros_like(attn))
    slots = attn @ v_zero
    slots_ref = slots + model.slot_ff(slots)
    y_phi_zero = eval_y_from_slots(slots_ref)
    results['phi_zero'] = measure(y_phi_zero)

    # v zero
    slots_vzero = comps['attn'] @ torch.zeros_like(comps['v'])
    slots_vzero_ref = slots_vzero + model.slot_ff(slots_vzero)
    y_v_zero = eval_y_from_slots(slots_vzero_ref)
    results['v_zero'] = measure(y_v_zero)

    # slot_ff zero (skip refinement)
    slots_noref = comps['slots']  # before refinement
    y_noref = eval_y_from_slots(slots_noref)
    results['slot_ff_zero'] = measure(y_noref)

    # queries zero
    q_zero = torch.zeros_like(comps['q'])
    logits = (q_zero @ comps['k'].transpose(1, 2)) / (model.H ** 0.5)
    logits = logits.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
    attn_qzero = F.softmax(logits, dim=-1)
    attn_qzero = torch.where(torch.isfinite(attn_qzero), attn_qzero, torch.zeros_like(attn_qzero))
    slots_qzero = attn_qzero @ comps['v']
    slots_qzero_ref = slots_qzero + model.slot_ff(slots_qzero)
    y_qzero = eval_y_from_slots(slots_qzero_ref)
    results['queries_zero'] = measure(y_qzero)

    return dict(baseline=(baseline_y.detach().cpu().numpy()), effects=results)


def slot_gradients(model, x, mask, reduction: str = 'sum'):
    """Compute gradient of model output w.r.t. per-slot refined activations.

    Returns gradients of shape (B,K,H) and aggregated importance per slot (K,)
    """
    device = next(model.parameters()).device
    x = x.to(device)
    mask = mask.to(device)
    comps = _forward_components(model, x, mask)
    slots = comps['slots_refined']  # (B,K,H)
    slots = slots.detach().requires_grad_(True)
    y = model.head(slots.reshape(slots.shape[0], model.K * model.H))
    # Reduce outputs to scalar
    if reduction == 'sum':
        s = y.sum()
    else:
        s = y.mean()
    s.backward()
    grads = slots.grad  # (B,K,H)
    if grads is None:
        return None
    importance = grads.abs().mean(dim=-1).detach().cpu().numpy()  # (B,K)
    agg = importance.mean(axis=0)  # (K,)
    return dict(grads=grads.detach().cpu().numpy(), per_sample=importance, agg=agg)


def slot_correlation_with_logm(model, x, mask):
    """Compute Pearson correlation between each slot's activation norm and logm statistics.
    Returns (K,) correlations.
    """
    comps = _forward_components(model, x, mask)
    slots = comps['slots_refined'].detach().cpu().numpy()  # (B,K,H)
    logm = comps['logm'].detach().cpu().numpy()  # (B,N,1)
    # reduce slots per sample: mean norm across slot vector
    slot_norms = np.linalg.norm(slots, axis=-1)  # (B,K)
    # reduce logm per sample: mean logm across galaxies
    logm_mean = logm.mean(axis=1).squeeze(-1)  # (B,)
    corrs = []
    for k in range(slot_norms.shape[1]):
        s = slot_norms[:, k]
        if np.std(s) == 0 or np.std(logm_mean) == 0:
            corrs.append(0.0)
        else:
            corrs.append(np.corrcoef(s, logm_mean)[0, 1])
    return np.array(corrs)


def plot_bar(values, labels=None, title=None, out_path: Optional[str]=None):
    if plt is None:
        return
    plt.figure(figsize=(8, 4))
    if labels is None:
        labels = [str(i) for i in range(len(values))]
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels)
    if title:
        plt.title(title)
    plt.tight_layout()
    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
    else:
        plt.show()


def _attn_entropy(attn: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute normalized entropy per (B,K) for attention over N valid items.

    attn: (B,K,N) non-negative, sums to 1 over N for each (B,K)
    mask: (B,N) in {0,1}
    Returns: (B,K) entropy normalized by log(n_valid)
    """
    # Zero-out padded items; renormalize
    m = mask.unsqueeze(1).to(attn.dtype)  # (B,1,N)
    w = attn * m
    w = w / (w.sum(dim=-1, keepdim=True).clamp_min(eps))

    ent = -(w.clamp_min(eps) * w.clamp_min(eps).log()).sum(dim=-1)  # (B,K)
    n_valid = m.sum(dim=-1).clamp_min(1.0)  # (B,1)
    denom = n_valid.log().clamp_min(eps)
    return ent / denom


@torch.no_grad()
def slot_attention_summary_batch(model, x, mask, topk: int = 10):
    """Summarize per-slot attention behavior on a single batch.

    Returns a dict with:
      - attn: (B,K,N) attention weights
      - feat_mean_w: (K,D) attention-weighted mean of input features
      - feat_std_w:  (K,D) attention-weighted std of input features
      - entropy:     (K,) mean normalized attention entropy
      - topk_values: (K, B*topk, D) input features for top-k attended items per sample
      - topk_weights:(K, B*topk) corresponding attention weights
    """
    device = next(model.parameters()).device
    x = x.to(device)
    mask = mask.to(device)

    out = model(x, mask)
    if isinstance(out, (tuple, list)) and len(out) >= 2:
        attn = out[1]
    else:
        raise ValueError("Expected SlotSetPool to return (y, attn)")

    B, N, D = x.shape
    _, K, _ = attn.shape

    m = mask.unsqueeze(1).to(attn.dtype)  # (B,1,N)
    w = attn * m
    wsum = w.sum(dim=(0, 2)).clamp_min(1e-12)  # (K,)

    # attention-weighted mean/std over items and batch
    # sum over B and N
    x_exp = x.unsqueeze(1)  # (B,1,N,D)
    w_exp = w.unsqueeze(-1)  # (B,K,N,1)
    feat_sum = (w_exp * x_exp).sum(dim=(0, 2))  # (K,D)
    feat_mean = feat_sum / wsum.unsqueeze(-1)
    feat_sum2 = (w_exp * (x_exp ** 2)).sum(dim=(0, 2))
    feat_var = (feat_sum2 / wsum.unsqueeze(-1) - feat_mean ** 2).clamp_min(0.0)
    feat_std = feat_var.sqrt()

    # entropy per slot
    ent = _attn_entropy(attn, mask)  # (B,K)
    ent_mean = ent.mean(dim=0)  # (K,)

    # top-k items per (B,K)
    k = int(topk)
    if k < 1:
        k = 1
    k = min(k, N)
    # ensure padded items can't be selected
    masked_attn = attn.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
    topv, topi = torch.topk(masked_attn, k=k, dim=-1)
    topv = torch.where(torch.isfinite(topv), topv, torch.zeros_like(topv))

    # gather x for each slot
    # x: (B,N,D) -> expand to (B,K,N,D) for gather
    xk = x.unsqueeze(1).expand(B, K, N, D)
    idx = topi.unsqueeze(-1).expand(B, K, k, D)
    topx = torch.gather(xk, dim=2, index=idx)  # (B,K,k,D)

    # reshape to per-slot list
    topx_slot = topx.permute(1, 0, 2, 3).reshape(K, B * k, D)
    topv_slot = topv.permute(1, 0, 2).reshape(K, B * k)

    return {
        "attn": attn.detach().cpu(),
        "feat_mean_w": feat_mean.detach().cpu(),
        "feat_std_w": feat_std.detach().cpu(),
        "entropy": ent_mean.detach().cpu(),
        "topk_values": topx_slot.detach().cpu(),
        "topk_weights": topv_slot.detach().cpu(),
    }


def accumulate_slot_attention_over_loader(model, loader, device, max_batches: int = 10, topk: int = 10):
    """Accumulate slot attention summaries over multiple batches.

    Returns a dict with per-slot weighted mean/std (streaming), mean entropy,
    and concatenated top-k values/weights.
    """
    model.eval()

    feat_sum = None
    feat_sum2 = None
    wsum = None
    ent_sum = None
    n_ent = 0

    top_vals = None
    top_wts = None

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        if len(batch) == 3:
            X, mask, _y = batch
        else:
            X, _y = batch
            raise ValueError("Expected set loader to yield (X, mask, y) for SlotSetPool")

        X = X.to(device)
        mask = mask.to(device)

        out = model(X, mask)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            attn = out[1]
        else:
            raise ValueError("Expected SlotSetPool to return (y, attn)")

        B, N, D = X.shape
        _, K, _ = attn.shape

        m = mask.unsqueeze(1).to(attn.dtype)
        w = attn * m

        w_batch = w.sum(dim=(0, 2))  # (K,)
        w_batch = w_batch.detach().cpu()

        X_cpu = X.detach().cpu()
        w_cpu = w.detach().cpu()

        # streaming weighted moments over B,N
        x_exp = X_cpu.unsqueeze(1)  # (B,1,N,D)
        w_exp = w_cpu.unsqueeze(-1)  # (B,K,N,1)
        f_sum = (w_exp * x_exp).sum(dim=(0, 2))  # (K,D)
        f_sum2 = (w_exp * (x_exp ** 2)).sum(dim=(0, 2))

        if feat_sum is None:
            feat_sum = f_sum
            feat_sum2 = f_sum2
            wsum = w_batch
            ent_sum = _attn_entropy(attn.detach().cpu(), mask.detach().cpu()).sum(dim=0)
        else:
            feat_sum += f_sum
            feat_sum2 += f_sum2
            wsum += w_batch
            ent_sum += _attn_entropy(attn.detach().cpu(), mask.detach().cpu()).sum(dim=0)
        n_ent += B

        # top-k accumulation
        batch_summary = slot_attention_summary_batch(model, X, mask, topk=topk)
        btv = batch_summary["topk_values"].numpy()  # (K, B*k, D)
        btw = batch_summary["topk_weights"].numpy()  # (K, B*k)
        if top_vals is None:
            top_vals = [btv[k] for k in range(K)]
            top_wts = [btw[k] for k in range(K)]
        else:
            for k in range(K):
                top_vals[k] = np.concatenate([top_vals[k], btv[k]], axis=0)
                top_wts[k] = np.concatenate([top_wts[k], btw[k]], axis=0)

    if feat_sum is None:
        return None

    wsum = np.asarray(wsum, dtype=np.float64)
    wsum = np.clip(wsum, 1e-12, None)
    feat_sum = np.asarray(feat_sum, dtype=np.float64)
    feat_sum2 = np.asarray(feat_sum2, dtype=np.float64)

    mean = feat_sum / wsum[:, None]
    var = np.clip(feat_sum2 / wsum[:, None] - mean ** 2, 0.0, None)
    std = np.sqrt(var)
    ent = (np.asarray(ent_sum, dtype=np.float64) / max(1, n_ent)).astype(np.float64)

    return {
        "feat_mean_w": mean,
        "feat_std_w": std,
        "entropy": ent,
        "topk_values": top_vals,
        "topk_weights": top_wts,
    }


def plot_slot_feature_histograms(topk_values, out_path: str, feature_idx: int = 0, bins: int = 50, title: str = "slot top-k feature"):
    """Plot per-slot histograms of the given feature from top-k values.

    topk_values: list of length K, each element (M, D)
    """
    if plt is None:
        return
    K = len(topk_values)
    cols = 4
    rows = int(np.ceil(K / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axs = np.array(axs).reshape(-1)

    for k in range(K):
        ax = axs[k]
        v = np.asarray(topk_values[k])
        if v.ndim == 1:
            vals = v
        else:
            vals = v[:, feature_idx]
        ax.hist(vals, bins=bins, alpha=0.8)
        ax.set_title(f"slot {k}")
        ax.set_xscale("log")

    for j in range(K, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)

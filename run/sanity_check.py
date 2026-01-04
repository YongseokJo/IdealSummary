import sys
import os
# Ensure repo root is on path
repo_root = os.path.dirname(os.path.dirname(__file__))
src_dir = os.path.join(repo_root, "src")
# Ensure repo root and src are on path so imports like `set_dsr` work
sys.path.insert(0, src_dir)
sys.path.insert(0, repo_root)

import torch
from src.stats_sym import SimplifiedSetDSR
from src.stats_sym import SummaryStatistics


def main():
    device = torch.device('cpu')

    # Configuration matching user's run: MLP head on, no learnable weights, no top-k, no symbolic transforms
    cfg = dict(
        n_features=3,
        n_transforms=4,
        output_dim=2,
        top_k=16,
        include_moments=True,
        include_cumulants=True,
        include_quantiles=False,
        use_learnable_weights=False,
        weight_hidden_dims=[32,16],
        selection_method='learnable',
        use_mlp_head=True,
        head_hidden_dims=[64,32],
        dropout=0.1,
        soft_quantile_temperature=1.0,
        use_symbolic_transforms=False,
        use_top_k=False,
    )

    model = SimplifiedSetDSR(**cfg).to(device)

    B = 2
    N = 16
    D = cfg['n_features']

    # Synthetic batch
    X = torch.randn(B, N, D, device=device)
    mask = torch.ones(B, N, device=device)
    y = torch.randn(B, cfg['output_dim'], device=device)

    print('Model summary:')
    print(' use_symbolic_transforms =', model.use_symbolic_transforms)
    print(' use_learnable_weights  =', model.weight_net is not None)
    print(' use_top_k              =', model.use_top_k)
    print(' selector present       =', model.selector is not None)
    print(' n_summary_features     =', model.n_summary_features)
    print(' head input dim         =', model.head.mlp[0].in_features if isinstance(model.head.mlp, torch.nn.Sequential) else model.head.mlp.in_features)

    # Compute summary features
    features, weights = model.compute_summary_features(X, mask)
    print('features.shape =', features.shape)
    print('weights =', weights)

    # Forward pass
    pred = model(X, mask)
    print('pred.shape =', pred.shape)

    # Compute loss and backward to check gradients
    loss = torch.nn.functional.mse_loss(pred, y)
    loss.backward()
    print('loss =', loss.item())

    # Check gradients on head parameters
    grads_present = any((p.grad is not None and p.grad.abs().sum().item() > 0) for p in model.head.parameters())
    print('gradients on head present:', grads_present)

    # If weight_net exists, check its grads
    if model.weight_net is not None:
        wg = any((p.grad is not None and p.grad.abs().sum().item() > 0) for p in model.weight_net.parameters())
        print('gradients on weight_net present:', wg)

    print('Sanity check complete.')

    # --- Additional: check SummaryStatistics correctness on raw per-element values ---
    print('\nChecking SummaryStatistics on raw per-element values...')
    stats_net = SummaryStatistics(include_moments=True, include_cumulants=True, include_quantiles=False)

    # Use one feature column from X
    vals = X[:, :, 0]  # (B, N)

    # Case 1: no weights (uniform masked)
    stats_out = stats_net(vals, mask, None)  # (B, n_stats)
    stat_names = stats_net.stat_names
    print('stat_names:', stat_names)
    print('stats_out shape:', stats_out.shape)

    # Manual numpy computations for mean and variance
    vals_np = vals.cpu().numpy()
    mask_np = mask.cpu().numpy()
    for b in range(vals_np.shape[0]):
        valid = mask_np[b] == 1
        v = vals_np[b, valid]
        if v.size == 0:
            continue
        mean_np = v.mean()
        var_np = v.var()
        mean_mod = stats_out[b, stat_names.index('mean')].item()
        var_mod = stats_out[b, stat_names.index('var')].item()
        print(f'B={b} | mean_np={mean_np:.6f} mean_mod={mean_mod:.6f} diff={mean_mod-mean_np:.6e}')
        print(f'B={b} | var_np={var_np:.6f} var_mod={var_mod:.6f} diff={var_mod-var_np:.6e}')

    # Case 2: random weights
    print('\nChecking weighted statistics with random weights...')
    rand_w = torch.rand(B, N)
    rand_w = rand_w * mask  # zero out padding
    # normalize
    w_sum = rand_w.sum(dim=1, keepdim=True) + 1e-8
    w_norm = rand_w / w_sum
    stats_w = stats_net(vals, mask, w_norm)

    # Manual weighted mean/var
    for b in range(B):
        v = vals[b, mask[b]==1].cpu().numpy()
        w = w_norm[b, mask[b]==1].cpu().numpy()
        mean_w = (v * w).sum()
        var_w = ((v - mean_w) ** 2 * w).sum()
        mean_mod = stats_w[b, stat_names.index('mean')].item()
        var_mod = stats_w[b, stat_names.index('var')].item()
        print(f'B={b} weighted mean_np={mean_w:.6f} mean_mod={mean_mod:.6f} diff={mean_mod-mean_w:.6e}')
        print(f'B={b} weighted var_np={var_w:.6f} var_mod={var_mod:.6f} diff={var_mod-var_w:.6e}')


if __name__ == '__main__':
    main()

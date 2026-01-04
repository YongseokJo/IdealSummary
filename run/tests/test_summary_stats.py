import sys
import os
repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(repo_root, '..', 'src'))
sys.path.insert(0, os.path.join(repo_root, '..'))

import numpy as np
import torch
from stats_sym import SummaryStatistics


def test_summary_statistics_unweighted():
    torch.manual_seed(0)
    B = 3
    N = 10
    # random values
    vals = torch.randn(B, N)
    mask = torch.ones(B, N)

    stats_net = SummaryStatistics(include_moments=True, include_cumulants=True, include_quantiles=False)
    stats = stats_net(vals, mask, None)  # (B, n_stats)
    names = stats_net.stat_names

    for b in range(B):
        v = vals[b].numpy()
        mean_np = v.mean()
        var_np = v.var()
        mean_mod = stats[b, names.index('mean')].item()
        var_mod = stats[b, names.index('var')].item()
        assert np.allclose(mean_np, mean_mod, atol=1e-6)
        assert np.allclose(var_np, var_mod, atol=1e-6)


def test_summary_statistics_weighted():
    torch.manual_seed(1)
    B = 2
    N = 8
    vals = torch.randn(B, N)
    mask = torch.ones(B, N)

    # random positive weights
    w = torch.rand(B, N)
    w = w * mask
    w = w / (w.sum(dim=1, keepdim=True) + 1e-8)

    stats_net = SummaryStatistics(include_moments=True, include_cumulants=True, include_quantiles=False)
    stats = stats_net(vals, mask, w)
    names = stats_net.stat_names

    for b in range(B):
        v = vals[b].numpy()
        ww = w[b].numpy()
        mean_w = (v * ww).sum()
        var_w = ((v - mean_w) ** 2 * ww).sum()
        mean_mod = stats[b, names.index('mean')].item()
        var_mod = stats[b, names.index('var')].item()
        assert np.allclose(mean_w, mean_mod, atol=1e-6)
        assert np.allclose(var_w, var_mod, atol=1e-6)


if __name__ == '__main__':
    test_summary_statistics_unweighted()
    test_summary_statistics_weighted()
    print('SummaryStatistics tests passed')

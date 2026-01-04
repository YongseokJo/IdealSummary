import torch
import torch.nn as nn
from typing import List, Optional


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int, activation=nn.ReLU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



class DeepSetMultiPhi(nn.Module):
    def __init__(
        self,
        input_dim: int,
        phi_hiddens,
        phi_out_dims=None,
        rho_hidden=[128, 128],
        agg: str = "sum",
        out_dim: int = 1,
        append_count: bool = True,     # NEW
        append_logN: bool = True,      # NEW
        activation=nn.ReLU,
    ):
        super().__init__()
        assert agg in ("sum", "mean")
        self.agg = agg
        self.append_count = append_count
        self.append_logN = append_logN

        if phi_out_dims is None:
            phi_out_dims = [h[-1] for h in phi_hiddens]
        assert len(phi_hiddens) == len(phi_out_dims)

        self.phis = nn.ModuleList([
            MLP(input_dim, list(hid), int(outd), activation=activation)
            for hid, outd in zip(phi_hiddens, phi_out_dims)
        ])
        self.phi_out_dims = list(map(int, phi_out_dims))

        extra = (1 if append_count else 0) + (1 if append_logN else 0)
        rho_in_dim = sum(self.phi_out_dims) + extra
        self.rho = MLP(rho_in_dim, rho_hidden, out_dim, activation=activation)


    def _pool(self, h: torch.Tensor, mask: Optional[torch.Tensor], counts: Optional[torch.Tensor]) -> torch.Tensor:
        # h: (B, N, Hk)
        if self.agg == "sum":
            return h.sum(dim=1)
        else:
            if mask is None:
                return h.mean(dim=1)
            else:
                return h.sum(dim=1) / counts.to(dtype=h.dtype)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                counts: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        device, dtype = x.device, x.dtype

        if counts is None:
            if mask is None:
                counts = torch.full((B, 1), float(N), device=device, dtype=dtype)
                mask_f = None
            else:
                mask_bool = mask.bool()
                mask_f = mask_bool.unsqueeze(-1).to(dtype=dtype, device=device)  # (B,N,1)
                counts = mask_bool.sum(dim=1, keepdim=True).to(dtype=dtype).clamp_min(1.0)  # (B,1)
        else:
            # user-provided counts should be shape (B,1) or (B,)
            counts = counts.to(device=device, dtype=dtype)
            if counts.ndim == 1:
                counts = counts.unsqueeze(-1)
            mask_f = None if mask is None else mask.unsqueeze(-1).to(dtype=dtype, device=device)

        pooled_list = []
        x_flat = x.reshape(B * N, D)
        for phi in self.phis:
            h = phi(x_flat).reshape(B, N, -1)
            if mask_f is not None:
                h = h * mask_f
            if self.agg == "sum":
                pooled = h.sum(dim=1)
            else:  # mean
                pooled = h.sum(dim=1) / counts
            pooled_list.append(pooled)

        z = torch.cat(pooled_list, dim=1)

        # append N and/or logN
        extras = []
        if self.append_count:
            extras.append(counts)
        if self.append_logN:
            extras.append(torch.log(counts.clamp_min(1.0)))

        if extras:
            z = torch.cat([z] + extras, dim=1)

        out = self.rho(z)
        return out.squeeze(-1) if out.shape[-1] == 1 else out


    @torch.no_grad()
    def forward_streaming(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, chunk: int = 65536) -> torch.Tensor:
        """
        Streaming pooling over N for huge sets.
        x: (B, N, D)
        mask: (B, N) or None
        """
        B, N, D = x.shape
        device = x.device
        dtype = x.dtype

        # accumulators per φ
        z_list = [torch.zeros(B, Hk, device=device, dtype=dtype) for Hk in self.phi_out_dims]
        count = torch.zeros(B, 1, device=device, dtype=dtype)

        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            xb = x[:, s:e, :].reshape(-1, D)  # (B*(e-s), D)

            if mask is not None:
                mb = mask[:, s:e].unsqueeze(-1).to(dtype=dtype, device=device)  # (B,chunk,1)
                count += mb.sum(dim=1)  # (B,1)
            else:
                mb = None
                count += (e - s)

            # compute each φ on this chunk and accumulate pooled sums
            for k, phi in enumerate(self.phis):
                hb = phi(xb).reshape(B, e - s, self.phi_out_dims[k])  # (B,chunk,Hk)
                if mb is not None:
                    hb = hb * mb
                z_list[k] += hb.sum(dim=1)  # (B,Hk)

        if self.agg == "mean":
            denom = count.clamp_min(1.0)
            z_list = [zk / denom for zk in z_list]

        z = torch.cat(z_list, dim=1)

        extras = []
        if self.append_count:
            extras.append(count)
        if self.append_logN:
            extras.append(torch.log(count.clamp_min(1.0)))

        if extras:
            z = torch.cat([z] + extras, dim=1)

        out = self.rho(z)
        return out.squeeze(-1) if out.shape[-1] == 1 else out



class DeepSet(nn.Module):
    """Simple DeepSets implementation for regression.

    Forward expects input of shape (B, N, D) where N can be padded and a
    `mask` of shape (B, N) with 1 for real elements and 0 for padding.
    """

    def __init__(self, input_dim: int, phi_hidden: List[int] = [64, 64], rho_hidden: List[int] = [64, 64],
                 agg: str = "sum", out_dim: int = 1):
        super().__init__()
        assert agg in ("sum", "mean"), "agg must be 'sum' or 'mean'"
        self.agg = agg
        self.phi = MLP(input_dim, phi_hidden, phi_hidden[-1])
        self.rho = MLP(phi_hidden[-1], rho_hidden, out_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        h = self.phi(x.view(B * N, D)).view(B, N, -1)
        if mask is not None:
            # mask: (B, N) -> boolean -> float for multiplication
            mask_bool = mask.bool()
            mask_f = mask_bool.unsqueeze(-1).to(dtype=h.dtype, device=h.device)  # (B, N, 1)
            h = h * mask_f
            counts = mask_bool.sum(dim=1, keepdim=True).clamp_min(1)  # (B, 1)
        else:
            counts = None
        if self.agg == "sum":
            z = h.sum(dim=1)  # (B, H)
        else:  # mean
            if counts is None:
                z = h.mean(dim=1)
            else:
                z = h.sum(dim=1) / counts.to(dtype=h.dtype)
        out = self.rho(z)
        return out.squeeze(-1) if out.shape[-1] == 1 else out

    def forward_streaming(self, x, mask=None, chunk=65536):
        # x: (B, N, 1), mask: (B, N) or None
        B, N, D = x.shape
        device = x.device
        H = self.phi_out_dim

        z = torch.zeros(B, H, device=device, dtype=x.dtype)
        count = torch.zeros(B, 1, device=device, dtype=x.dtype)

        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            xb = x[:, s:e, :].reshape(-1, D)                 # (B*(e-s), 1)
            hb = self.phi(xb).reshape(B, e - s, H)          # (B, chunk, H)

            if mask is not None:
                mb = mask[:, s:e].unsqueeze(-1).to(hb.dtype) # (B, chunk, 1)
                hb = hb * mb
                count += mb.sum(dim=1)                       # (B,1)
            else:
                count += (e - s)

            z += hb.sum(dim=1)                               # (B,H)

        if self.agg == "mean":
            z = z / count.clamp_min(1.0)

        # optionally append logN
        logN = torch.log(count.clamp_min(1.0))
        z = torch.cat([z, logN], dim=1)

        return self.rho(z)



if __name__ == "__main__":
    # quick smoke test
    model = DeepSet(input_dim=4)
    x = torch.randn(2, 5, 4)
    mask = torch.tensor([[1,1,1,1,1],[1,1,0,0,0]])
    y = model(x, mask)
    print(y.shape)

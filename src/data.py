"""HDF5 dataset helpers for CAMELS LH files.

Provides a Dataset that returns the `SubhaloStellarMass` array for a
simulation index and the corresponding `parameters` vector as target.

Usage example:

```py
from src.data import HDF5SetDataset, hdf5_collate
from torch.utils.data import DataLoader

ds = HDF5SetDataset('data/camels_LH.hdf5', snap=90, param_keys=['Omega_m','sigma_8'])
loader = DataLoader(ds, batch_size=8, collate_fn=hdf5_collate)
for X, mask, y in loader:
    # X: (B, N, D), mask: (B, N), y: (B, K)
    pass
```
"""

from typing import List, Optional, Tuple
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch

class HDF5SetDataset(Dataset):
    """Dataset that indexes CAMELS LH HDF5 by simulation index.

    Each item is a tuple (x, y) where `x` is a variable-length 2D
    float32 array of shape (N_i, D) taken from the group
    `snap_{snap:03d}/SubhaloStellarMass` and `y` is a 1D float32 array
    containing the requested parameter values from `parameters`.

    Notes:
    - The class opens the HDF5 file lazily so it works with DataLoader
      workers. The file handle is stored per-instance and reopened on
      first access in each worker process.
    - If a single parameter is requested, `y` will be a scalar numpy
      float32. If multiple are requested, `y` will be a 1D numpy array
      of shape `(K,)`.
    """

    def __init__(self, h5_path: str, snap: int = 90, param_keys: Optional[List[str]] = None, data_field: str = "SubhaloStellarMass"):
        self.h5_path = h5_path
        self.snap = snap
        self.param_keys = param_keys
        # data_field can be a path relative to the snapshot group, e.g. "smf/phi"
        self.data_field = data_field
        self._f = None

        # open briefly to infer length and available param keys
        with h5py.File(self.h5_path, "r") as f:
            if "parameters" not in f:
                raise ValueError("HDF5 file missing 'parameters' group")
            p = f["parameters"]

            # New consolidated layout: /parameters/params -> (nsims, ndim)
            if "params" in p:
                params_ds = p["params"]
                self.length = int(params_ds.shape[0])

                # try to read param names if present (stored as bytes or strings)
                if "param_names" in p:
                    raw_names = p["param_names"][()]
                    # ensure a list of str
                    try:
                        param_list = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in raw_names]
                    except Exception:
                        param_list = [str(n) for n in raw_names]
                else:
                    # fallback default names
                    param_list = [f"param_{i}" for i in range(params_ds.shape[1])]

                # expose param_names and optional seed array to users
                self.param_names = param_list
                if "seed" in p:
                    try:
                        self.seeds = np.array(p["seed"][()], dtype=np.int32)
                    except Exception:
                        self.seeds = None
                else:
                    self.seeds = None

                if self.param_keys is None:
                    self.param_keys = param_list
                else:
                    # validate requested keys are present in param_names
                    for k in self.param_keys:
                        if k not in param_list:
                            raise KeyError(f"Requested param key '{k}' not found in parameters/param_names")
                    # compute indices into the consolidated params array for requested keys
                    try:
                        self._param_idx = [param_list.index(k) for k in self.param_keys]
                    except Exception:
                        self._param_idx = None

            else:
                # legacy layout: individual arrays under /parameters/<param_name>
                all_keys = list(p.keys())
                if self.param_keys is None:
                    self.param_keys = all_keys
                else:
                    for k in self.param_keys:
                        if k not in p:
                            raise KeyError(f"Requested param key '{k}' not found in parameters")
                # length determined by first param array
                self.length = int(p[self.param_keys[0]].shape[0])
                # expose param_names and seeds in legacy case as best-effort
                try:
                    self.param_names = list(self.param_keys)
                except Exception:
                    self.param_names = None
                if "seed" in p:
                    try:
                        self.seeds = np.array(p["seed"][()], dtype=np.int32)
                    except Exception:
                        self.seeds = None
                else:
                    self.seeds = None
            # determine maximum per-sample length for the requested data_field
            snap_group_name = f"snap_{self.snap:03d}"
            if snap_group_name not in f:
                raise KeyError(f"Snapshot group '{snap_group_name}' not found in HDF5 file")
            ds_group = f[snap_group_name]
            ds = ds_group
            for part in self.data_field.split('/'):
                if part not in ds:
                    raise KeyError(f"Dataset '{self.data_field}' not found under group {snap_group_name}")
                ds = ds[part]

            # compute max length across all samples for variable-length sets
            max_len = 0
            try:
                for i in range(self.length):
                    xi = ds[i]
                    # xi may be scalar, 1D vector, or 2D; treat length accordingly
                    if hasattr(xi, '__len__'):
                        l = len(xi)
                    else:
                        l = 1
                    if l > max_len:
                        max_len = l
            except Exception:
                # if indexing fails for some reason, fall back to length of first element
                try:
                    first = np.array(ds[0])
                    max_len = first.shape[0] if first.ndim >= 1 else 1
                except Exception:
                    max_len = 0

            self.max_size = int(max_len)

    def _ensure_open(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")

    def __len__(self):
        return int(self.length)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_open()
        f = self._f
        # read targets
        p = f["parameters"]

        # handle consolidated params dataset
        if "params" in p:
            params_ds = p["params"]
            y_full = np.asarray(params_ds[idx], dtype=np.float32)
            # if user requested a subset of param keys, select those indices
            if hasattr(self, "_param_idx") and self._param_idx is not None:
                try:
                    y = y_full[self._param_idx].astype(np.float32)
                except Exception:
                    # fallback to full vector if indexing fails
                    y = y_full.astype(np.float32)
            else:
                y = y_full.astype(np.float32)
            # ensure 1-D
            y = y.reshape(-1) if np.ndim(y) > 1 and y.shape[0] == 1 else y
            if y.ndim == 0:
                y = np.atleast_1d(y.astype(np.float32))
        else:
            # legacy per-key arrays
            y_vals = [np.array(p[k][idx], dtype=np.float32) for k in self.param_keys]
            if len(y_vals) == 1:
                y = y_vals[0].astype(np.float32)
            else:
                y = np.stack(y_vals).astype(np.float32)

        # read requested data field for the requested snapshot
        snap_group_name = f"snap_{self.snap:03d}"
        if snap_group_name not in f:
            raise KeyError(f"Snapshot group '{snap_group_name}' not found in HDF5 file")
        g = f[snap_group_name]
        # drill into nested fields (e.g. smf/phi)
        ds = g
        for part in self.data_field.split('/'):
            if part not in ds:
                raise KeyError(f"Dataset '{self.data_field}' not found under group {snap_group_name}")
            ds = ds[part]
        try:
            x = np.array(ds[idx], dtype=np.float32)
        except (IndexError, TypeError, KeyError) as e:
            # Provide a helpful error if indexing fails; dataset layout may
            # differ from expectations (e.g., ragged/compound types).
            raise RuntimeError(
                f"Failed to index SubhaloStellarMass at index {idx}. "
                "Inspect the file with your notebook to determine the correct layout."
            ) from e

        return x, y


def hdf5_collate(batch, max_size: int = None):
    """Collate function to pad variable-length sets into tensors.

    Returns (X, mask, y) where X is float32 tensor (B, N_max), mask is
    float32 tensor (B, N_max) and y is float32 tensor (B, K) or (B,) for
    single-target.
    """
    N = max_size
    B = len(batch)
    D = batch[0][0].shape[1] if batch[0][0].ndim == 2 else 1
    X = np.zeros((B, N, D), dtype=np.float32)
    mask = np.zeros((B, N), dtype=np.float32)

    ys = []
    for i, (xi, yi) in enumerate(batch):
        if D == 1:
            xi = xi.reshape(-1,1)
        _len = xi.shape[0]
        xi = np.array(xi, dtype=np.float32)
        X[i, :_len,:] = xi[:]
        mask[i, :_len] = 1.0
        ys.append(yi)

    y_arr = np.stack([np.array(v, dtype=np.float32) for v in ys])
    return torch.from_numpy(X), torch.from_numpy(mask), torch.from_numpy(y_arr)


def smf_collate(batch):
    """Collate for SMF fixed-length vectors: returns (X, y) where
    X is (B, D) and y is (B,) or (B, K).
    """
    Xs = [np.array(b[0], dtype=np.float32) for b in batch]
    ys = [np.array(b[1], dtype=np.float32) for b in batch]
    X = np.stack(Xs, axis=0)
    y_arr = np.stack(ys, axis=0)
    return torch.from_numpy(X), torch.from_numpy(y_arr)


__all__ = ["HDF5SetDataset", "hdf5_collate", "smf_collate"]


__all__ = ["HDF5SetDataset", "hdf5_collate"]

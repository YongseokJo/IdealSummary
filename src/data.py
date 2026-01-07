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

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch

class HDF5SetDataset(Dataset):
    """Dataset that indexes CAMELS LH HDF5 by simulation index.

    Each item is a tuple (x, y) where `x` is a variable-length 2D
    float32 array of shape (N_i, D) taken from the group
    `snap_{snap:03d}/<data_field>` and `y` is a 1D float32 array
    containing the requested parameter values from `parameters`.

    Notes:
    - The class opens the HDF5 file lazily so it works with DataLoader
      workers. The file handle is stored per-instance and reopened on
      first access in each worker process.
    - If a single parameter is requested, `y` will be a scalar numpy
      float32. If multiple are requested, `y` will be a 1D numpy array
      of shape `(K,)`.
    - Optional per-feature cuts can be provided via mass_min/mass_max with
      mass_feature_idx or mass_feature_name; each can be a scalar or a list.
    - When using mass_feature_name, feature names must be provided or stored
      on the HDF5 dataset as an attribute (e.g. feature_names). When
      data_field is a list, the feature names default to those field names.
    """

    def __init__(
        self,
        h5_path: str,
        snap: int = 90,
        param_keys: Optional[List[str]] = None,
        data_field: Union[str, Sequence[str]] = "SubhaloStellarMass",
        mass_min: Optional[Union[float, Sequence[Optional[float]]]] = None,
        mass_max: Optional[Union[float, Sequence[Optional[float]]]] = None,
        mass_feature_idx: Optional[Union[int, Sequence[int]]] = None,
        mass_feature_name: Optional[Union[str, Sequence[str]]] = None,
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.h5_path = h5_path
        self.snap = snap
        self.param_keys = param_keys
        # data_field can be a path relative to the snapshot group, e.g. "smf/phi"
        self.data_field = data_field
        self.data_fields = self._normalize_data_fields(data_field)
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.mass_feature_idx = mass_feature_idx
        self.mass_feature_name = mass_feature_name
        if feature_names is not None:
            self.feature_names = list(feature_names)
        elif len(self.data_fields) > 1:
            self.feature_names = list(self.data_fields)
        else:
            self.feature_names = None
        if len(self.data_fields) > 1 and self.feature_names is not None:
            if len(self.feature_names) != len(self.data_fields):
                raise ValueError(
                    "feature_names must match the number of data_field entries when using multiple data_field values"
                )
        self._cut_specs: List[Tuple[int, Optional[float], Optional[float]]] = []
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
            datasets = [self._get_dataset(ds_group, field, snap_group_name) for field in self.data_fields]

            if self.feature_names is None and len(self.data_fields) == 1:
                self._load_feature_names_from_ds(datasets[0])
            if self.feature_names is None and self.mass_feature_name is not None and len(self.data_fields) == 1:
                self.feature_names = [self.data_fields[0]]
            self._cut_specs = self._normalize_cut_specs()

            # compute max length across all samples for variable-length sets
            max_len = 0
            try:
                for i in range(self.length):
                    xi = self._read_sample_fields(datasets, i)
                    if self._cut_specs:
                        xi = self._apply_mass_cuts(np.asarray(xi))
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
                    first = np.array(self._read_sample_fields(datasets, 0))
                    if self._cut_specs:
                        first = self._apply_mass_cuts(first)
                    max_len = first.shape[0] if first.ndim >= 1 else 1
                except Exception:
                    max_len = 0

            self.max_size = int(max_len)

    def _ensure_open(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")

    def _normalize_data_fields(self, data_field: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(data_field, (list, tuple, np.ndarray)):
            fields = [str(f) for f in data_field]
        else:
            fields = [str(data_field)]
        if not fields:
            raise ValueError("data_field must contain at least one entry")
        return fields

    def _get_dataset(self, group, data_field: str, snap_group_name: str):
        ds = group
        for part in data_field.split('/'):
            if part not in ds:
                raise KeyError(f"Dataset '{data_field}' not found under group {snap_group_name}")
            ds = ds[part]
        return ds

    def _read_sample_fields(self, datasets, idx: int) -> np.ndarray:
        if len(datasets) == 1:
            return np.array(datasets[0][idx], dtype=np.float32)
        arrays = []
        for ds in datasets:
            xi = np.array(ds[idx], dtype=np.float32)
            if xi.ndim == 0:
                xi = xi.reshape(1)
            if xi.ndim == 1:
                xi = xi.reshape(-1, 1)
            elif xi.ndim == 2 and xi.shape[1] == 1:
                pass
            else:
                raise ValueError(
                    "Multiple data_field entries only support 1D vectors per field; "
                    "use a single data_field for multi-column datasets."
                )
            arrays.append(xi)
        lengths = {a.shape[0] for a in arrays}
        if len(lengths) != 1:
            raise ValueError("data_field entries have mismatched lengths for the same sample")
        return np.concatenate(arrays, axis=1)

    def _load_feature_names_from_ds(self, ds) -> None:
        if self.feature_names is not None:
            return
        for key in ("feature_names", "field_names", "column_names", "columns"):
            if key in ds.attrs:
                raw = ds.attrs[key]
                if isinstance(raw, (list, tuple, np.ndarray)):
                    names = [n.decode() if isinstance(n, (bytes, bytearray)) else str(n) for n in raw]
                else:
                    names = [raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)]
                self.feature_names = names
                return

    def _normalize_cut_specs(self) -> List[Tuple[int, Optional[float], Optional[float]]]:
        def _as_list(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple, np.ndarray)):
                return list(value)
            return [value]

        mins = _as_list(self.mass_min)
        maxs = _as_list(self.mass_max)
        idxs = _as_list(self.mass_feature_idx)
        names = _as_list(self.mass_feature_name)

        if mins is None and maxs is None:
            return []

        if names is not None and idxs is not None:
            raise ValueError("Specify either mass_feature_idx or mass_feature_name, not both")

        lengths = [len(v) for v in (mins, maxs, idxs, names) if v is not None]
        n_cuts = max(lengths) if lengths else 0

        if names is not None:
            if self.feature_names is None:
                raise ValueError(
                    "mass_feature_name requires feature_names to be provided or stored as a dataset attribute"
                )
            name_to_idx = {n: i for i, n in enumerate(self.feature_names)}
            try:
                idxs = [name_to_idx[n] for n in names]
            except KeyError as exc:
                missing = exc.args[0]
                raise ValueError(
                    f"Unknown feature name '{missing}' in mass_feature_name; "
                    f"available: {self.feature_names}"
                ) from exc

        if idxs is None:
            if n_cuts > 1:
                raise ValueError("mass_feature_idx must be provided when specifying multiple cuts")
            idxs = [0]
            n_cuts = 1

        def _broadcast(values, name):
            if values is None:
                return [None] * n_cuts
            if len(values) == 1 and n_cuts > 1:
                return values * n_cuts
            if len(values) != n_cuts:
                raise ValueError(f"{name} length {len(values)} does not match number of cuts {n_cuts}")
            return values

        mins = _broadcast(mins, "mass_min")
        maxs = _broadcast(maxs, "mass_max")
        idxs = _broadcast(idxs, "mass_feature_idx")

        specs = []
        for idx, min_v, max_v in zip(idxs, mins, maxs):
            min_val = float(min_v) if min_v is not None else None
            max_val = float(max_v) if max_v is not None else None
            if min_val is not None and max_val is not None and min_val > max_val:
                raise ValueError("mass_min must be <= mass_max for each cut")
            if min_val is None and max_val is None:
                continue
            specs.append((int(idx), min_val, max_val))

        return specs

    def _apply_mass_cuts(self, x: np.ndarray) -> np.ndarray:
        if not self._cut_specs:
            return x
        if x.ndim == 0:
            x = x.reshape(1)
        if x.ndim == 1:
            if len(self._cut_specs) > 1:
                raise ValueError("Multiple cuts require 2D inputs with feature columns")
            idx, min_v, max_v = self._cut_specs[0]
            if idx not in (0, -1):
                raise ValueError("mass_feature_idx must be 0 for 1D inputs")
            mask = np.ones(x.shape[0], dtype=bool)
            if min_v is not None:
                mask &= x >= min_v
            if max_v is not None:
                mask &= x <= max_v
            return x[mask]
        if x.ndim == 2:
            mask = np.ones(x.shape[0], dtype=bool)
            for idx, min_v, max_v in self._cut_specs:
                if idx < 0 or idx >= x.shape[1]:
                    raise ValueError(
                        f"mass_feature_idx {idx} out of bounds for data with {x.shape[1]} features"
                    )
                mass_vals = x[:, idx]
                if min_v is not None:
                    mask &= mass_vals >= min_v
                if max_v is not None:
                    mask &= mass_vals <= max_v
            return x[mask]
        raise ValueError(f"Unsupported data shape for mass cuts: {x.shape}")

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
        datasets = [self._get_dataset(g, field, snap_group_name) for field in self.data_fields]
        try:
            x = self._read_sample_fields(datasets, idx)
        except (IndexError, TypeError, KeyError, ValueError) as e:
            # Provide a helpful error if indexing fails; dataset layout may
            # differ from expectations (e.g., ragged/compound types).
            raise RuntimeError(
                f"Failed to index data_field(s) {self.data_fields} at index {idx}. "
                "Inspect the file with your notebook to determine the correct layout."
            ) from e

        if self._cut_specs:
            x = self._apply_mass_cuts(x)

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

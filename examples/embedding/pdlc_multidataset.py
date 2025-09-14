"""Multi-dataset PDLC training on TabPFNv2 embeddings (Classification)

This example trains a single PDLC head across multiple OpenML datasets.
The PDLC head learns to predict whether two embeddings belong to the same class.

Key features
- Loads multiple small/medium OpenML classification datasets
- Extracts TabPFNv2 row embeddings per dataset (train/test)
- Trains a shared PDLC head with pairwise BCE loss on pooled pairs
- Evaluates per-dataset each epoch by aggregating scores vs. that dataset's training set
- Logs training loss and per-dataset accuracies to a CSV; optional checkpointing

Requirements
- Full TabPFN package (pip install tabpfn)
- scikit-learn, numpy, torch, pandas (for logging convenience)
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding


def resolve_device(device_arg: str | torch.device | None = "auto") -> torch.device:
    """Resolve a torch.device similar to pdlc_head_test.py.

    - If given a torch.device, return it
    - If "auto"/None, prefer CUDA, then MPS, else CPU
    - If string like "cpu"/"cuda"/"mps", use it
    """
    if isinstance(device_arg, torch.device):
        return device_arg
    if device_arg is None or str(device_arg).lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(str(device_arg))


"""
Training script only. Evaluation utilities have been moved to a separate script.
"""


def ensure_embeddings_2d(emb: np.ndarray, n_samples_expected: int) -> np.ndarray:
    """Match the robust shaping used in pdlc_head_test.py

    - Squeeze singleton dims
    - Move the axis matching n_samples_expected to axis 0
    - Flatten remaining dims to a single embedding dim
    """
    emb = np.asarray(emb)
    emb = np.squeeze(emb)
    if emb.ndim == 1:
        if n_samples_expected == 1:
            return emb.reshape(1, -1)
        raise ValueError(
            f"Embedding is 1D with shape {emb.shape}, expected {n_samples_expected} samples."
        )
    candidate_axes = [ax for ax, sz in enumerate(emb.shape) if sz == n_samples_expected]
    if len(candidate_axes) == 0:
        diffs = [abs(sz - n_samples_expected) for sz in emb.shape]
        sample_axis = int(np.argmin(diffs))
    else:
        sample_axis = candidate_axes[0]
    emb = np.moveaxis(emb, sample_axis, 0)
    emb = emb.reshape(emb.shape[0], -1)
    return emb.astype(np.float32, copy=False)




def load_openml_dataset(
    name_or_id: str,
    test_size: float = 0.3,
    random_state: int = 42,
):
    kwargs = {"as_frame": True, "return_X_y": True}
    try:
        data_id = int(name_or_id)
        X, y = fetch_openml(data_id=data_id, **kwargs)
    except ValueError:
        X, y = fetch_openml(name=name_or_id, **kwargs)

    # Drop rows with any missing values
    X = X.copy()
    mask = ~X.isna().any(axis=1)
    X, y = X[mask], y[mask]

    # One-hot encode categoricals (safe and consistent pre-split)
    X = pd.get_dummies(X, dummy_na=False)

    # Label-encode target and convert X to float32
    le = LabelEncoder()
    y_num = le.fit_transform(y)
    X = X.to_numpy(dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_num, test_size=test_size, random_state=random_state, stratify=y_num
    )
    return X_train, X_test, y_train, y_test, le



@dataclass
class DatasetEntry:
    name: str
    E_train: np.ndarray
    y_train: np.ndarray
    E_test: np.ndarray
    y_test: np.ndarray
    class_to_indices: Dict[int, np.ndarray]


def get_embeddings(
    name: str,
    n_fold: int,
    seed: int,
) -> DatasetEntry:
    """Extract 2D embeddings consistent with pdlc_head_test.py logic."""
    X_train, X_test, y_train, y_test, _ = load_openml_dataset(name, random_state=seed)
    # Enforce hard limits before fitting TabPFN
    n_features = X_train.shape[1]
    n_classes = int(len(np.unique(y_train)))
    if (n_features > 500) or (n_classes > 10):
        raise ValueError(
            f"Dataset {name} skipped: features={n_features} (max 500), classes={n_classes} (max 10)"
        )
    clf = TabPFNClassifier(n_estimators=1, random_state=seed)
    emb = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)
    E_train_raw = emb.get_embeddings(X_train, y_train, X_test, data_source="train")
    E_test_raw = emb.get_embeddings(X_train, y_train, X_test, data_source="test")
    E_train = ensure_embeddings_2d(E_train_raw, n_samples_expected=X_train.shape[0])
    E_test = ensure_embeddings_2d(E_test_raw, n_samples_expected=X_test.shape[0])
    class_to_indices = {int(c): np.where(y_train == c)[0] for c in np.unique(y_train)}
    return DatasetEntry(
        name=name,
        E_train=E_train,
        y_train=y_train,
        E_test=E_test,
        y_test=y_test,
        class_to_indices=class_to_indices,
    )



class MultiDatasetAllPairs(torch.utils.data.Dataset):
    """All ordered pairs within each dataset, mirroring pdlc_head_test.py logic.

    - No cross-dataset pairs; only (i, j) within the same dataset
    - Includes self-pairs (i == j)
    - Target t = 1.0 if same class, 0.0 otherwise
    """

    def __init__(self, entries: List[DatasetEntry]):
        assert len(entries) > 0
        self.entries = entries

        # Validate emb dims are consistent
        dims = [e.E_train.shape[1] for e in entries]
        if len(set(dims)) != 1:
            raise ValueError(f"Embedding dimensions differ across datasets: {dims}")

        # Precompute cumulative lengths to map global index -> (dataset, i, j)
        self.N = [e.E_train.shape[0] for e in entries]
        self.block_sizes = [n * n for n in self.N]
        self.cum = np.cumsum([0] + self.block_sizes)  # len = D+1

    def __len__(self) -> int:
        return int(self.cum[-1])

    def __getitem__(self, idx: int):
        # Find dataset block
        d = int(np.searchsorted(self.cum, idx, side="right") - 1)
        local = idx - int(self.cum[d])
        n = self.N[d]
        i = local // n
        j = local % n
        e = self.entries[d]
        e1 = torch.from_numpy(e.E_train[i].astype(np.float32, copy=False))
        e2 = torch.from_numpy(e.E_train[j].astype(np.float32, copy=False))
        t = torch.tensor([float(e.y_train[i] == e.y_train[j])], dtype=torch.float32)
        return e1, e2, t


class BalancedMultiDatasetPairs(torch.utils.data.Dataset):
    """Dataset-balanced random pair sampler with self-pairs allowed.

    - Ensures equal number of sampled pairs per dataset per epoch
      by mapping each index to a dataset via idx % D (robust to DataLoader shuffling).
    - Within the chosen dataset, samples i, j uniformly with replacement
      (self-pairs possible) and sets target to 1.0 if same-class else 0.0.

    Parameters
    - entries: list of DatasetEntry
    - total_samples: number of pairs to draw in this epoch
    - seed: optional RNG seed
    """

    def __init__(self, entries: List[DatasetEntry], total_samples: int, seed: int | None = None):
        assert len(entries) > 0
        assert total_samples is not None and total_samples > 0
        self.entries = entries
        self.total_samples = int(total_samples)
        self.D = len(entries)
        # Store a base seed; sampling will be derived from (base_seed, idx)
        # to make results independent of DataLoader shuffling and num_workers.
        self.seed = int(seed if seed is not None else 0)

        dims = [e.E_train.shape[1] for e in entries]
        if len(set(dims)) != 1:
            raise ValueError(f"Embedding dimensions differ across datasets: {dims}")

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int):
        d = int(idx % self.D)  # balanced counts per dataset per epoch
        e = self.entries[d]
        n = e.E_train.shape[0]
        # Derive a deterministic RNG from (seed, idx) so results are reproducible
        # regardless of DataLoader shuffling or multiprocessing order.
        # Use 64-bit mixing constants (LCG-style) for stable seeds.
        mixed = (
            np.uint64(self.seed) * np.uint64(6364136223846793005)
            + np.uint64(idx) * np.uint64(1442695040888963407)
            + np.uint64(d)
        )
        rng = np.random.default_rng(int(np.uint64(mixed)))
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        e1 = torch.from_numpy(e.E_train[i].astype(np.float32, copy=False))
        e2 = torch.from_numpy(e.E_train[j].astype(np.float32, copy=False))
        t = torch.tensor([float(e.y_train[i] == e.y_train[j])], dtype=torch.float32)
        return e1, e2, t


class PDLCHead(nn.Module):
    def __init__(self, emb_dim: int, hidden: Tuple[int, int] | None = None, dropout: float = 0.0):
        super().__init__()
        if hidden is None:
            h1 = max(64, min(512, emb_dim))
            h2 = max(32, min(256, emb_dim // 2))
            hidden = (h1, h2)
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([e1, e2], dim=1))




def _pair_counts_entries(entries: List[DatasetEntry]) -> Tuple[int, int]:
    """Compute total same/different counts across datasets like pdlc_head_test.pair_counts.

    Within each dataset d:
      n_same_d = sum_c (count_{d,c}^2)  [includes self-pairs]
      n_total_d = N_d^2
      n_diff_d = n_total_d - n_same_d
    Return sums over all datasets.
    """
    n_same_total, n_diff_total = 0, 0
    for e in entries:
        _, counts = np.unique(e.y_train, return_counts=True)
        n_same = int((counts**2).sum())
        N = int(counts.sum())
        n_total = N * N
        n_diff = n_total - n_same
        n_same_total += n_same
        n_diff_total += n_diff
    return n_same_total, n_diff_total


def _pos_weight_for_entries(entries: List[DatasetEntry], balanced: bool) -> float:
    """Compute pos_weight matching the sampling distribution.

    - If balanced is False (all-pairs or uniform subsample of all pairs):
      use global counts across entries: pos_weight = n_diff / n_same.
    - If balanced is True (uniform over datasets, then uniform i,j within dataset):
      let p_same_d = sum_c n_{d,c}^2 / n_d^2; p_same = mean_d p_same_d; pos_weight = (1-p_same)/p_same.
    """
    if not entries:
        return 1.0
    if not balanced:
        n_same_all, n_diff_all = _pair_counts_entries(entries)
        return float(n_diff_all) / max(1.0, float(n_same_all))
    # balanced: average per-dataset same-prob
    ps: List[float] = []
    for e in entries:
        _, counts = np.unique(e.y_train, return_counts=True)
        n = int(counts.sum())
        if n <= 0:
            continue
        p_same_d = float((counts.astype(np.int64)**2).sum()) / float(n * n)
        ps.append(p_same_d)
    if not ps:
        return 1.0
    p_same = float(np.mean(ps))
    p_diff = max(0.0, 1.0 - p_same)
    if p_same <= 0:
        return 1.0
    return p_diff / p_same


@torch.no_grad()
def predict_pdlc(
    model: PDLCHead,
    E_train: np.ndarray,
    y_train: np.ndarray,
    E_test: np.ndarray,
    device: str | torch.device | None = None,
    batch_train: int | None = None,
    use_prior: bool | str = "auto",
    anchor_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Predict labels using the same aggregation as pdlc_head_test.py.

    - Average P(same) over (e_t, e_i) and (e_i, e_t)
    - Optional prior-based expansion (auto when min class count < 5)
    - Optional per-anchor weights
    """
    device_t = resolve_device(device)
    model = model.to(device_t).eval()

    E_train_t = torch.from_numpy(E_train.astype(np.float32, copy=False)).to(device_t)
    E_test_t = torch.from_numpy(E_test.astype(np.float32, copy=False)).to(device_t)

    classes = np.unique(y_train)
    class_to_mask: Dict[int, torch.Tensor] = {}
    for c in classes:
        class_to_mask[int(c)] = torch.tensor(y_train == c, device=device_t)

    # Map labels to contiguous [0..C-1] indices for vector ops
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    anchor_class_idx = np.array([class_to_idx[int(c)] for c in y_train], dtype=np.int64)
    anchor_class_idx_t = torch.tensor(anchor_class_idx, device=device_t, dtype=torch.long)

    # Priors per class in classes-order
    _, counts = np.unique(y_train, return_counts=True)
    priors_np = counts.astype(np.float64) / float(counts.sum())
    priors_t = torch.tensor(priors_np, device=device_t, dtype=torch.float32)

    # Decide whether to apply prior-adjustment
    if isinstance(use_prior, str) and use_prior.lower() == "auto":
        use_prior_flag = bool(counts.min() < 5)
    else:
        use_prior_flag = bool(use_prior)

    # Anchor weights (length N), default uniform and normalized
    N_train = E_train.shape[0]
    if anchor_weights is None:
        w_np = np.full(N_train, 1.0 / max(1, N_train), dtype=np.float64)
    else:
        w_np = np.asarray(anchor_weights, dtype=np.float64)
        assert w_np.shape[0] == N_train, f"anchor_weights length {w_np.shape[0]} != {N_train}"
        w_np = np.clip(w_np, 0.0, None)
        s = float(w_np.sum())
        w_np = w_np / s if s > 0 else np.full(N_train, 1.0 / max(1, N_train), dtype=np.float64)
    w_t = torch.tensor(w_np.astype(np.float32), device=device_t)

    preds: List[int] = []
    sigmoid = nn.Sigmoid()

    for t_idx in range(E_test_t.shape[0]):
        e_t = E_test_t[t_idx : t_idx + 1]  # (1, D)

        if use_prior_flag:
            test_class_scores = torch.zeros(len(classes), device=device_t, dtype=torch.float32)
        else:
            per_class_scores_sum = {int(c): 0.0 for c in classes}
            per_class_weight_sum = {int(c): 0.0 for c in classes}

        num_train = E_train_t.shape[0]
        block_size = num_train if (batch_train is None or batch_train <= 0) else int(batch_train)
        n_blocks = int(math.ceil(num_train / block_size))
        for b in range(n_blocks):
            s = b * block_size
            e = min(num_train, (b + 1) * block_size)
            block = E_train_t[s:e]

            # Forward for both orders; model outputs logit for 'same'
            logits1 = model(e_t.expand(block.shape[0], -1), block)
            logits2 = model(block, e_t.expand(block.shape[0], -1))
            p_same = 0.5 * (sigmoid(logits1) + sigmoid(logits2)).squeeze(1)

            if use_prior_flag:
                c_idx_blk = anchor_class_idx_t[s:e]
                w_blk = w_t[s:e]
                B = c_idx_blk.shape[0]
                one_minus_s = (1.0 - p_same).unsqueeze(1)
                denom = (1.0 - priors_t[c_idx_blk]).unsqueeze(1) + 1e-12
                L = one_minus_s * priors_t.unsqueeze(0) / denom  # (B, C)
                L[torch.arange(B, device=device_t), c_idx_blk] = p_same
                class_contrib = (L * w_blk.unsqueeze(1)).sum(dim=0)
                test_class_scores += class_contrib
            else:
                for c in classes:
                    mask = class_to_mask[int(c)][s:e]
                    if mask.any():
                        w_mask = w_t[s:e][mask]
                        per_class_scores_sum[int(c)] += float((p_same[mask] * w_mask).sum().item())
                        per_class_weight_sum[int(c)] += float(w_mask.sum().item())

        if use_prior_flag:
            total = float(test_class_scores.sum().item())
            if total > 0:
                test_class_scores = test_class_scores / total
            best_idx = int(torch.argmax(test_class_scores).item())
            best_class = int(classes[best_idx])
        else:
            class_means = {
                int(c): (per_class_scores_sum[int(c)] / max(1e-12, per_class_weight_sum[int(c)])) for c in classes
            }
            best_class = max(class_means.items(), key=lambda kv: kv[1])[0]
        preds.append(int(best_class))

    return np.array(preds, dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset PDLC on TabPFN embeddings")
    parser.add_argument(
        "--datasets",
        type=str,
        default="61,444,714,736,756,768,782",
        help="Comma-separated OpenML dataset names or IDs",
    )
    parser.add_argument(
        "--auto_small",
        action="store_true",
        help="Automatically select small OpenML datasets (<= max_rows). Overrides --datasets if set.",
    )
    parser.add_argument("--max_rows", type=int, default=1000, help="Max rows per dataset when using --auto_small")
    parser.add_argument("--max_datasets", type=int, default=5, help="Max number of datasets to use when auto-selecting")
    parser.add_argument("--n_fold", type=int, default=0, help="0 for vanilla, >=2 for K-fold embeddings")
    parser.add_argument("--pairs_per_epoch", type=int, default=None, help="Number of pairs per epoch (used by balanced sampling or subsampling)")
    parser.add_argument("--balanced_sampling", action="store_true", help="Use dataset-balanced random pair sampling per epoch")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_csv", type=str, default="pdlc_multi_log.csv")
    parser.add_argument("--ckpt", type=str, default="pdlc_multi.pt")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Determine dataset list
    if args.auto_small:
        try:
            import openml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "--auto_small requires the 'openml' package. Install with: pip install openml"
            ) from e

        print(f"Discovering small OpenML datasets (<= {args.max_rows} rows, <= 500 features, <= 10 classes)...")
        df = openml.datasets.list_datasets(output_format="dataframe")
        # Keep active classification datasets with <= max_rows and at least 2 classes
        df = df[(df.status == "active")]
        if "NumberOfInstances" in df.columns:
            df = df[df["NumberOfInstances"].fillna(0) <= args.max_rows]
        if "NumberOfFeatures" in df.columns:
            # Keep datasets with features <= 500
            df = df[df["NumberOfFeatures"].fillna(0) <= 500]
        if "NumberOfClasses" in df.columns:
            # Keep classification datasets with classes in [2, max_classes]
            n_classes = df["NumberOfClasses"].fillna(0)
            df = df[n_classes.between(2, 10, inclusive="both")]

        # Prefer the latest version per name to avoid duplicate name-versions; fall back to unique dids
        if {"name", "version"}.issubset(df.columns):
            df = df.sort_values(["name", "version"], ascending=[True, False])
            df = df.drop_duplicates(subset=["name"], keep="first")

        # Shuffle deterministically and limit, count
        df = df.sample(frac=1.0, random_state=args.seed).head(args.max_datasets)
        ids = [str(int(did)) for did in df["did"].tolist()]
        names = ids
        print(f"Selected dataset IDs: {ids}")
    else:
        # Load datasets and extract embeddings
        names = [s.strip() for s in args.datasets.split(",") if s.strip()]
        print(f"Datasets: {names}")
    entries: List[DatasetEntry] = []
    for name in names:
        print(f"Preparing dataset: {name}")
        try:
            entry = get_embeddings(
                name,
                n_fold=args.n_fold,
                seed=args.seed,
            )
        except Exception as e:
            # Skip datasets that fail during embedding extraction (e.g., ARPACK SVD failures)
            print(f"Skipping dataset {name} due to error: {type(e).__name__}: {e}")
            continue
        print(f" - E_train: {entry.E_train.shape}, E_test: {entry.E_test.shape}")
        entries.append(entry)

    if not entries:
        print("No datasets matched the constraints (<= 500 features and <= 10 classes). Exiting.")
        return

    # Split into train/val datasets (hold out one dataset for validation if possible)
    if not entries:
        print("No datasets matched the constraints (<= 500 features and <= 10 classes). Exiting.")
        return
    if len(entries) >= 2:
        val_entries = [entries[-1]]
        train_entries = entries[:-1]
        print(f"Validation holdout: {val_entries[0].name}")
    else:
        val_entries = []
        train_entries = entries

    # Build shared PDLC head
    emb_dim = train_entries[0].E_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PDLCHead(emb_dim=emb_dim, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # pos_weight aligned with sampling
    pw_val = _pos_weight_for_entries(train_entries, balanced=bool(args.balanced_sampling))
    pos_weight = torch.tensor([pw_val], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop
    best_mean_acc = -1.0
    log_rows: List[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.balanced_sampling:
            if args.pairs_per_epoch is None or args.pairs_per_epoch <= 0:
                # Fallback: allocate a conservative default if not provided
                # Default to 1e6 pairs per epoch split across datasets
                total_pairs = 1_000_000
            else:
                total_pairs = int(args.pairs_per_epoch)
            # Vary seed by epoch for diversity while keeping runs reproducible
            ds = BalancedMultiDatasetPairs(train_entries, total_samples=total_pairs, seed=args.seed + epoch)
            dl_gen = torch.Generator().manual_seed(args.seed + epoch)
            loader = torch.utils.data.DataLoader(
                ds,
                batch_size=args.batch,
                shuffle=True,
                drop_last=False,
                generator=dl_gen,
            )
        else:
            ds = MultiDatasetAllPairs(train_entries)
            # Optional uniform subsampling per epoch to limit compute
            if args.pairs_per_epoch is not None and args.pairs_per_epoch > 0:
                dl_gen = torch.Generator().manual_seed(args.seed + epoch)
                sampler = torch.utils.data.RandomSampler(
                    ds,
                    replacement=True,
                    num_samples=int(args.pairs_per_epoch),
                    generator=dl_gen,
                )
                loader = torch.utils.data.DataLoader(
                    ds,
                    batch_size=args.batch,
                    sampler=sampler,
                    drop_last=False,
                )
            else:
                dl_gen = torch.Generator().manual_seed(args.seed + epoch)
                loader = torch.utils.data.DataLoader(
                    ds,
                    batch_size=args.batch,
                    shuffle=True,
                    drop_last=False,
                    generator=dl_gen,
                )

        run_loss, n_batches = 0.0, 0
        for e1, e2, t in loader:
            e1, e2, t = e1.to(device), e2.to(device), t.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(e1, e2)
            loss = loss_fn(logits, t)
            loss.backward()
            opt.step()
            run_loss += float(loss.detach().cpu())
            n_batches += 1

        train_loss = run_loss / max(1, n_batches)

        # Evaluate
        model.eval()
        train_per_acc = {}
        for entry in train_entries:
            y_pred = predict_pdlc(model, entry.E_train, entry.y_train, entry.E_test, device=device, use_prior=True)
            acc = accuracy_score(entry.y_test, y_pred)
            train_per_acc[entry.name] = acc

        val_per_acc = {}
        for entry in val_entries:
            y_pred = predict_pdlc(model, entry.E_train, entry.y_train, entry.E_test, device=device, use_prior=True)
            acc = accuracy_score(entry.y_test, y_pred)
            val_per_acc[entry.name] = acc

        train_mean_acc = float(np.mean(list(train_per_acc.values()))) if train_per_acc else float("nan")
        val_mean_acc = float(np.mean(list(val_per_acc.values()))) if val_per_acc else float("nan")
        msg = (
            f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f} - train_mean: {train_mean_acc:.4f}"
            + (f" - val_mean: {val_mean_acc:.4f}" if val_per_acc else "")
        )
        # Print compact per-dataset summaries
        train_str = ", ".join(f"{k}:{v:.3f}" for k,v in train_per_acc.items())
        val_str = ", ".join(f"{k}:{v:.3f}" for k,v in val_per_acc.items())
        if train_str:
            msg += " - train: " + train_str
        if val_str:
            msg += " - val: " + val_str
        print(msg)

        # Log
        row = {"epoch": epoch, "loss": train_loss, "train_mean_acc": train_mean_acc}
        if val_per_acc:
            row["val_mean_acc"] = val_mean_acc
        for k, v in train_per_acc.items():
            row[f"acc_train_{k}"] = v
        for k, v in val_per_acc.items():
            row[f"acc_val_{k}"] = v
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(args.log_csv, index=False)

        # Checkpoint
        metric_for_ckpt = val_mean_acc if val_per_acc else train_mean_acc
        if metric_for_ckpt > best_mean_acc:
            best_mean_acc = metric_for_ckpt
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.ckpt)

    print(f"Training complete. Best metric: {best_mean_acc:.4f}. Logs: {args.log_csv}. Ckpt: {args.ckpt}")


if __name__ == "__main__":
    main()

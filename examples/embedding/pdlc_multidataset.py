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


def ensure_embeddings_2d(emb: np.ndarray, n_samples_expected: int) -> np.ndarray:
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    kwargs = {"as_frame": False, "return_X_y": True}
    try:
        data_id = int(name_or_id)
        X, y = fetch_openml(data_id=data_id, **kwargs)
    except ValueError:
        X, y = fetch_openml(name=name_or_id, **kwargs)

    le = LabelEncoder()
    y_num = le.fit_transform(y)

    mask = ~np.isnan(X).any(axis=1)
    X, y_num = X[mask], y_num[mask]
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


def build_entry(name: str, n_fold: int, seed: int) -> DatasetEntry:
    X_train, X_test, y_train, y_test, _ = load_openml_dataset(name, random_state=seed)
    clf = TabPFNClassifier(n_estimators=1, random_state=seed)
    emb = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)
    tr_raw = emb.get_embeddings(X_train, y_train, X_train, data_source="train")
    te_raw = emb.get_embeddings(X_train, y_train, X_test, data_source="test")
    E_train = ensure_embeddings_2d(tr_raw, n_samples_expected=X_train.shape[0])
    E_test = ensure_embeddings_2d(te_raw, n_samples_expected=X_test.shape[0])
    class_to_indices = {int(c): np.where(y_train == c)[0] for c in np.unique(y_train)}
    return DatasetEntry(
        name=name,
        E_train=E_train,
        y_train=y_train,
        E_test=E_test,
        y_test=y_test,
        class_to_indices=class_to_indices,
    )


@dataclass
class PairSamplingConfig:
    n_pairs: int = 100000
    pos_ratio: float = 0.5
    cross_neg_ratio: float = 0.3  # fraction of negatives sampled across datasets
    swap_prob: float = 0.5


class MultiDatasetPairs(torch.utils.data.Dataset):
    def __init__(self, entries: List[DatasetEntry], cfg: PairSamplingConfig, rng: np.random.Generator | None = None):
        assert len(entries) > 0
        self.entries = entries
        self.cfg = cfg
        self.rng = np.random.default_rng() if rng is None else rng

        # Validate emb dims are consistent
        dims = [e.E_train.shape[1] for e in entries]
        if len(set(dims)) != 1:
            raise ValueError(f"Embedding dimensions differ across datasets: {dims}")

        # Precompute per-dataset classes with >=2 samples for positives
        self.pos_ok = []  # list of (dataset_idx, class)
        for d_idx, e in enumerate(entries):
            for c, idx in e.class_to_indices.items():
                if idx.shape[0] >= 2:
                    self.pos_ok.append((d_idx, int(c)))
        if len(self.pos_ok) == 0:
            # Allow positives by sampling with replacement if absolutely needed
            for d_idx, e in enumerate(entries):
                for c in e.class_to_indices.keys():
                    self.pos_ok.append((d_idx, int(c)))

        self.dataset_indices = np.arange(len(entries))

    def __len__(self) -> int:
        return int(self.cfg.n_pairs)

    def _sample_positive(self) -> Tuple[np.ndarray, np.ndarray, float]:
        d_idx, c = self.pos_ok[self.rng.integers(0, len(self.pos_ok))]
        e = self.entries[d_idx]
        idx = e.class_to_indices[c]
        if idx.shape[0] >= 2:
            i, j = self.rng.choice(idx, size=2, replace=False)
        else:
            i, j = self.rng.choice(idx, size=2, replace=True)
        return e.E_train[i], e.E_train[j], 1.0

    def _sample_negative(self) -> Tuple[np.ndarray, np.ndarray, float]:
        cross = (self.rng.random() < self.cfg.cross_neg_ratio) and (len(self.entries) >= 2)
        if cross:
            d1, d2 = self.rng.choice(self.dataset_indices, size=2, replace=False)
            e1, e2 = self.entries[d1], self.entries[d2]
            i = int(self.rng.integers(0, e1.E_train.shape[0]))
            j = int(self.rng.integers(0, e2.E_train.shape[0]))
            return e1.E_train[i], e2.E_train[j], 0.0
        # within-dataset negative
        d_idx = int(self.rng.choice(self.dataset_indices))
        e = self.entries[d_idx]
        # pick two different classes
        classes = np.array(list(e.class_to_indices.keys()))
        if classes.shape[0] >= 2:
            c1, c2 = self.rng.choice(classes, size=2, replace=False)
        else:
            c1 = c2 = classes[0]
        i = int(self.rng.choice(e.class_to_indices[int(c1)]))
        j = int(self.rng.choice(e.class_to_indices[int(c2)]))
        # if same class by chance (degenerate), resample j until different or break
        tries = 0
        while (e.y_train[i] == e.y_train[j]) and tries < 3:
            j = int(self.rng.integers(0, e.E_train.shape[0]))
            tries += 1
        return e.E_train[i], e.E_train[j], 0.0

    def __getitem__(self, _: int):
        want_pos = self.rng.random() < self.cfg.pos_ratio
        if want_pos:
            e1, e2, t = self._sample_positive()
        else:
            e1, e2, t = self._sample_negative()
        if self.rng.random() < self.cfg.swap_prob:
            e1, e2 = e2, e1
        return (
            torch.from_numpy(e1.astype(np.float32, copy=False)),
            torch.from_numpy(e2.astype(np.float32, copy=False)),
            torch.tensor([t], dtype=torch.float32),
        )


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


@torch.no_grad()
def predict_pdlc(
    model: PDLCHead,
    E_train: np.ndarray,
    y_train: np.ndarray,
    E_test: np.ndarray,
    device: torch.device,
    batch_train: int = 4096,
) -> np.ndarray:
    model.eval()
    E_train_t = torch.from_numpy(E_train.astype(np.float32, copy=False)).to(device)
    E_test_t = torch.from_numpy(E_test.astype(np.float32, copy=False)).to(device)
    classes = np.unique(y_train)
    class_to_mask: Dict[int, torch.Tensor] = {int(c): torch.tensor(y_train == c, device=device) for c in classes}
    preds: List[int] = []
    sigmoid = nn.Sigmoid()
    for t_idx in range(E_test_t.shape[0]):
        e_t = E_test_t[t_idx : t_idx + 1]
        per_sum = {int(c): 0.0 for c in classes}
        per_cnt = {int(c): 0 for c in classes}
        num_train = E_train_t.shape[0]
        n_blocks = int(math.ceil(num_train / batch_train))
        for b in range(n_blocks):
            s = b * batch_train
            e = min(num_train, (b + 1) * batch_train)
            block = E_train_t[s:e]
            logits1 = model(e_t.expand(block.shape[0], -1), block)
            logits2 = model(block, e_t.expand(block.shape[0], -1))
            probs = 0.5 * (sigmoid(logits1) + sigmoid(logits2)).squeeze(1)
            for c in classes:
                mask = class_to_mask[int(c)][s:e]
                if mask.any():
                    per_sum[int(c)] += probs[mask].sum().item()
                    per_cnt[int(c)] += int(mask.sum().item())
        best = max(((c, per_sum[c] / max(1, per_cnt[c])) for c in classes), key=lambda kv: kv[1])[0]
        preds.append(int(best))
    return np.array(preds, dtype=int)


def main():
    parser = argparse.ArgumentParser(description="Multi-dataset PDLC on TabPFN embeddings")
    parser.add_argument(
        "--datasets",
        type=str,
        default="phoneme,yeast,vehicle",
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
    parser.add_argument("--pairs_per_epoch", type=int, default=100000)
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

        print(f"Discovering small OpenML datasets (<= {args.max_rows} rows)...")
        df = openml.datasets.list_datasets(output_format="dataframe")
        # Keep active classification datasets with <= max_rows and at least 2 classes
        df = df[(df.status == "active")]
        if "NumberOfInstances" in df.columns:
            df = df[df["NumberOfInstances"].fillna(0) <= args.max_rows]
        if "NumberOfClasses" in df.columns:
            df = df[df["NumberOfClasses"].fillna(0) >= 2]

        # Prefer the latest version per name to avoid duplicate name-versions; fall back to unique dids
        if {"name", "version"}.issubset(df.columns):
            df = df.sort_values(["name", "version"], ascending=[True, False])
            df = df.drop_duplicates(subset=["name"], keep="first")

        # Shuffle deterministically and limit count
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
        entry = build_entry(name, n_fold=args.n_fold, seed=args.seed)
        print(f" - E_train: {entry.E_train.shape}, E_test: {entry.E_test.shape}")
        entries.append(entry)

    # Build shared PDLC head
    emb_dim = entries[0].E_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PDLCHead(emb_dim=emb_dim, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training loop
    best_mean_acc = -1.0
    log_rows: List[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        cfg = PairSamplingConfig(n_pairs=args.pairs_per_epoch)
        ds = MultiDatasetPairs(entries, cfg, rng=rng)
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False)

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

        # Evaluate per dataset
        model.eval()
        per_acc = {}
        for entry in entries:
            y_pred = predict_pdlc(model, entry.E_train, entry.y_train, entry.E_test, device=device)
            acc = accuracy_score(entry.y_test, y_pred)
            per_acc[entry.name] = acc

        mean_acc = float(np.mean(list(per_acc.values()))) if per_acc else float("nan")
        print(f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f} - mean_acc: {mean_acc:.4f} - " + ", ".join(f"{k}:{v:.3f}" for k,v in per_acc.items()))

        # Log
        row = {"epoch": epoch, "loss": train_loss, "mean_acc": mean_acc}
        for k, v in per_acc.items():
            row[f"acc_{k}"] = v
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(args.log_csv, index=False)

        # Checkpoint
        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.ckpt)

    print(f"Training complete. Best mean accuracy: {best_mean_acc:.4f}. Logs: {args.log_csv}. Ckpt: {args.ckpt}")


if __name__ == "__main__":
    main()

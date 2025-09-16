"""PDLC Head on TabPFNv2 Embeddings (Classification)

This example demonstrates how to:
- Load a small/medium OpenML classification dataset
- Extract row embeddings from TabPFNv2
- Train a Pairwise Discriminative Learning of Classes (PDLC) head on pairs of embeddings
- Predict labels for the test set by aggregating pairwise "same-class" scores per class
"""

from __future__ import annotations

import argparse
import os
import random
import math
from collections import defaultdict
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import openml

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding


def resolve_device(device_arg: str | torch.device | None = "auto") -> torch.device:
    """Resolve a single torch.device based on availability and user preference.

    - If `device_arg` is a `torch.device`, return it.
    - If `device_arg` is "auto" or None: prefer CUDA, then MPS, else CPU.
    - If `device_arg` is a string like "cpu", "cuda", or "mps": use it.
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


def ensure_embeddings_2d(emb: np.ndarray, n_samples_expected: int) -> np.ndarray:
    """Ensure embeddings have shape (n_samples, emb_dim).

    Handles common TabPFN outputs like (D, N, 1) or (D, N) by:
    - Squeezing singleton dims
    - Finding the axis that matches n_samples_expected and moving it to axis 0
    - Flattening all remaining axes into a single embedding dimension
    """
    emb = np.asarray(emb)
    emb = np.squeeze(emb)  # drop size-1 dimensions like trailing (.., 1)

    if emb.ndim == 1:
        # Ambiguous: cannot determine sample axis; treat as a single sample
        if n_samples_expected == 1:
            return emb.reshape(1, -1)
        raise ValueError(
            f"Embedding is 1D with shape {emb.shape}, but expected {n_samples_expected} samples."
        )

    # Identify the sample axis: prefer exact match to n_samples_expected
    candidate_axes = [ax for ax, sz in enumerate(emb.shape) if sz == n_samples_expected]
    if len(candidate_axes) == 0:
        # Fall back: pick the axis with size closest to expected if unique
        diffs = [abs(sz - n_samples_expected) for sz in emb.shape]
        sample_axis = int(np.argmin(diffs))
    else:
        sample_axis = candidate_axes[0]

    # Move sample axis to front and flatten remaining dimensions as embedding dim
    emb = np.moveaxis(emb, sample_axis, 0)
    emb = emb.reshape(emb.shape[0], -1)
    return emb.astype(np.float32, copy=False)


def _preprocess_frame_to_numeric(
    X_df: "pd.DataFrame",
) -> Tuple[np.ndarray, List[int]]:
    """Convert a mixed-type DataFrame into a numeric array and collect categorical indices.

    - Numeric columns: fill NaN with median
    - Non-numeric columns: factorize to integer codes (with '__MISSING__' for NaN)
    Returns (X_numeric, categorical_indices)
    """
    cat_idx: List[int] = []
    X_proc = np.zeros((len(X_df), X_df.shape[1]), dtype=np.float32)
    for i, col_name in enumerate(X_df.columns):
        col = X_df[col_name]
        if pd.api.types.is_numeric_dtype(col):
            col_num = pd.to_numeric(col, errors="coerce")
            if col_num.isna().any():
                fill = col_num.median()
                if pd.isna(fill):
                    fill = 0.0
                col_num = col_num.fillna(fill)
            X_proc[:, i] = col_num.astype(np.float32).to_numpy()
        else:
            # Treat as categorical
            col_cat = col.astype("string").fillna("__MISSING__").astype("category")
            codes = col_cat.cat.codes.astype(np.int64).to_numpy()
            X_proc[:, i] = codes.astype(np.float32)
            cat_idx.append(i)
    return X_proc, cat_idx


def load_openml_dataset(
    name_or_id: str,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, List[int]]:
    """Load a small/medium OpenML classification dataset.

    Parameters
    - name_or_id: OpenML dataset name (e.g., "phoneme") or numeric id as string (e.g., "1489").
    - Returns X_train, X_test, y_train_num, y_test_num, label_encoder
    """
    # Try numeric id, else name, with optional version via "name:version"
    kwargs = {"as_frame": True, "return_X_y": True}
    name = name_or_id
    version = None
    if isinstance(name_or_id, str) and ":" in name_or_id:
        parts = name_or_id.split(":", 1)
        name = parts[0]
        try:
            version = int(parts[1])
        except Exception:
            version = None
    try:
        data_id = int(name)
        X, y = fetch_openml(data_id=data_id, **kwargs)
    except ValueError:
        if version is not None:
            X, y = fetch_openml(name=name, version=version, **kwargs)
        else:
            # Pin to version=1 when multiple versions exist to avoid ambiguity
            X, y = fetch_openml(name=name, version=1, **kwargs)

    # Ensure pandas DataFrame/Series
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Encode labels to integers
    le = LabelEncoder()
    y_num = le.fit_transform(y)

    # Preprocess features
    X_num, cat_idx = _preprocess_frame_to_numeric(X)

    # Decide whether stratification is feasible
    def _can_stratify(y: np.ndarray, tsize: float) -> bool:
        classes, counts = np.unique(y, return_counts=True)
        if counts.min() < 2:
            return False
        # Ensure each class can appear in both splits
        for c in counts:
            test_c = int(np.floor(c * tsize))
            train_c = c - test_c
            if test_c < 1 or train_c < 1:
                return False
        return True

    if _can_stratify(y_num, test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            X_num, y_num, test_size=test_size, random_state=random_state, stratify=y_num
        )
    else:
        warnings.warn(
            "Stratified split not possible (rare class too small). Falling back to random split.",
            UserWarning,
            stacklevel=2,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X_num, y_num, test_size=test_size, random_state=random_state, stratify=None
        )
    return X_train, X_test, y_train, y_test, le, cat_idx


class EmbeddingPairsDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        E: np.ndarray,  # (N, D)
        y: np.ndarray,  # (N,)
    ) -> None:
        assert E.ndim == 2
        assert E.shape[0] == y.shape[0]
        self.E = E.astype(np.float32, copy=False)
        self.y = y
        self.N = E.shape[0]

    def __len__(self) -> int:
        return self.N * self.N

    def __getitem__(self,idx):
        i = idx//self.N
        j = idx%self.N
        e1 = torch.from_numpy(self.E[i])
        e2 = torch.from_numpy(self.E[j])
        t = torch.tensor([float(self.y[i] == self.y[j])], dtype=torch.float32)  # 1=different, 0=same
        return e1, e2, t


class PDLCHead(nn.Module):
    """Small MLP that takes two embeddings and predicts same/different class.

    Input: concatenation [e1, e2] of dimension 2*D.
    Output: logit (unnormalized score); apply sigmoid for probability.
    """

    def __init__(self, emb_dim: int, hidden: Tuple[int, int] | None = None, dropout: float = 0.0):
        super().__init__()
        if hidden is None:
            # Reasonable defaults scaled by emb size
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
        # e1, e2: (B, D)
        x = torch.cat([e1, e2], dim=1)
        return self.net(x)  # (B, 1)


def pair_counts(y):
    _, counts = np.unique(y, return_counts=True)
    n_same = int((counts**2).sum())
    N = int(counts.sum())
    n_diff = N*N - n_same
    return n_same, n_diff

def train_pdlc_head(E_train: np.ndarray, 
                    y_train: np.ndarray, 
                    batch_size: int = 1024, 
                    epochs: int = 10, 
                    lr: float = 1e-3, 
                    weight_decay: float = 1e-4, 
                    device: str | torch.device | None = None, 
                    seed: int | None = None,
                    ) -> PDLCHead:
    E_train = E_train.astype(np.float32, copy=False)
    emb_dim = E_train.shape[1]
    device = resolve_device(device)

    # Deterministic sampling and shuffling
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    dataset = EmbeddingPairsDataset(E_train, y_train)
    num_pairs = len(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=gen,
    )

    model = PDLCHead(emb_dim=emb_dim, dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_same, n_diff = pair_counts(y_train)
    pos_weight = torch.tensor([n_diff/max(1, n_same)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # target: 1=different, 0=same to match PDLL

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for e1, e2, t in loader:
            e1 = e1.to(device)
            e2 = e2.to(device)
            t = t.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(e1, e2)
            loss = loss_fn(logits, t)
            loss.backward()
            opt.step()

            running_loss += float(loss.detach().cpu())
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

    return model.eval()


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
    """Predict class labels for test embeddings using PDLC aggregation.

    For a test embedding e_t, compare against all training embeddings e_i.
    Aggregate sigmoid(logit) scores per class and pick the max-mean class.
    To improve symmetry, average scores for (e_t, e_i) and (e_i, e_t).

    - batch_train: anchors per block for memory control. If None or <= 0,
      use all anchors in a single block (i.e., no chunking).
    """
    device = resolve_device(device)
    model = model.to(device).eval()

    E_train_t = torch.from_numpy(E_train.astype(np.float32, copy=False)).to(device)
    E_test_t = torch.from_numpy(E_test.astype(np.float32, copy=False)).to(device)

    classes = np.unique(y_train)
    class_to_mask: Dict[int, torch.Tensor] = {}
    for c in classes:
        class_to_mask[int(c)] = torch.tensor(y_train == c, device=device)

    # Map labels to contiguous [0..C-1] indices for vector ops
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    anchor_class_idx = np.array([class_to_idx[int(c)] for c in y_train], dtype=np.int64)
    anchor_class_idx_t = torch.tensor(anchor_class_idx, device=device, dtype=torch.long)

    # Priors per class in classes-order
    _, counts = np.unique(y_train, return_counts=True)
    priors_np = counts.astype(np.float64) / float(counts.sum())
    priors_t = torch.tensor(priors_np, device=device, dtype=torch.float32)

    # Decide whether to apply prior-adjustment (pdll 'use_prior' behavior)
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
        # Clip negatives, renormalize
        w_np = np.clip(w_np, 0.0, None)
        s = float(w_np.sum())
        w_np = w_np / s if s > 0 else np.full(N_train, 1.0 / max(1, N_train), dtype=np.float64)
    w_t = torch.tensor(w_np.astype(np.float32), device=device)

    preds: List[int] = []
    sigmoid = nn.Sigmoid()

    for t_idx in range(E_test_t.shape[0]):
        e_t = E_test_t[t_idx : t_idx + 1]  # (1, D)

        # We'll compute scores in blocks over training set to limit memory
        if use_prior_flag:
            test_class_scores = torch.zeros(len(classes), device=device, dtype=torch.float32)
        else:
            per_class_scores_sum = {int(c): 0.0 for c in classes}
            per_class_weight_sum = {int(c): 0.0 for c in classes}

        num_train = E_train_t.shape[0]
        block_size = num_train if (batch_train is None or batch_train <= 0) else int(batch_train)
        n_blocks = int(math.ceil(num_train / block_size))
        for b in range(n_blocks):
            s = b * block_size
            e = min(num_train, (b + 1) * block_size)
            block = E_train_t[s:e]  # (B, D)

            # Forward for both orders; model outputs logit for 'same'
            logits1 = model(e_t.expand(block.shape[0], -1), block)  # (B, 1)
            logits2 = model(block, e_t.expand(block.shape[0], -1))
            p_same = 0.5 * (sigmoid(logits1) + sigmoid(logits2)).squeeze(1)  # (B,)
            p_diff = 1.0 - p_same  # similarity = P(different) = 1 - P(same)

            if use_prior_flag:
                # Prior-based expansion to class likelihood per anchor, then weight and sum
                c_idx_blk = anchor_class_idx_t[s:e]  # (B,)
                w_blk = w_t[s:e]                    # (B,)
                B = c_idx_blk.shape[0]

                # L = ((1 - s) * prior / (1 - prior[class_of_anchor]))  with replacement of own class by s
                one_minus_s = (1.0 - p_same).unsqueeze(1)  # (B,1)
                denom = (1.0 - priors_t[c_idx_blk]).unsqueeze(1) + 1e-12
                L = one_minus_s * priors_t.unsqueeze(0) / denom  # (B, C)
                L[torch.arange(B, device=device), c_idx_blk] = p_same  # set own-class similarity
                # Apply anchor weights and sum
                class_contrib = (L * w_blk.unsqueeze(1)).sum(dim=0)  # (C,)
                test_class_scores += class_contrib
            else:
                # Weighted mean similarity per class (pdll 'no prior' branch)
                for c in classes:
                    mask = class_to_mask[int(c)][s:e]
                    if mask.any():
                        w_mask = w_t[s:e][mask]
                        per_class_scores_sum[int(c)] += float((p_same[mask] * w_mask).sum().item())
                        per_class_weight_sum[int(c)] += float(w_mask.sum().item())

        # Finalize per-class scores and choose the best
        if use_prior_flag:
            total = float(test_class_scores.sum().item())
            if total > 0:
                test_class_scores = test_class_scores / total  # normalize to sum=1 like pdll
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
    parser = argparse.ArgumentParser(description="PDLC head on TabPFN embeddings (classification) with OpenML 10-fold CV")
    parser.add_argument(
        "--dataset",
        type=str,
        default=61,
        help="OpenML dataset name or numeric id (default: iris)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_fold", type=int, default=0, help="n_fold=0 for vanilla, >=2 for K-fold embeddings")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1024, help="Batch size for PDLC training")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"], help="Device for TabPFN and PDLC")
    parser.add_argument("--task_id", type=int, default=None, help="OpenML task id (overrides dataset lookup)")
    args = parser.parse_args()

    # Set global seeds and determinism for reproducibility
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    X_train, X_test, y_train, y_test, le, cat_idx = load_openml_dataset(name_or_id=str(args.dataset))
    print(len(X_train), "train samples,", len(X_test), "test samples")
    dev = resolve_device(args.device)
    print(f"Using device: {dev}")
    clf = TabPFNClassifier(n_estimators=1, random_state=42)
    embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=args.n_fold)
    E_train = embedding_extractor.get_embeddings(X_train,y_train,X_test, data_source="train")
    E_test = embedding_extractor.get_embeddings(X_train,y_train,X_test, data_source="test")
    

    # PDLC head on embeddings
    pdlc = train_pdlc_head(
        E_train[0],
        y_train,
        batch_size=int(args.batch),
        epochs=int(args.epochs),
        device=dev,
        seed=args.seed,
    )
    y_pred_pdlc = predict_pdlc(pdlc, E_train[0], y_train, E_test[0], device=dev, batch_train=None)

    accuracy = accuracy_score(y_test, y_pred_pdlc)
    f1_macro = f1_score(y_test, y_pred_pdlc, average="macro")
    print(f"PDLC on TabPFN embeddings - Accuracy: {accuracy:.4f}, F1-macro: {f1_macro:.4f}")


if __name__ == "__main__":
    main()

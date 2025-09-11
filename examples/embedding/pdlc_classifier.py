"""PDLC Head on TabPFNv2 Embeddings (Classification)

This example demonstrates how to:
- Load a small/medium OpenML classification dataset
- Extract row embeddings from TabPFNv2
- Train a Pairwise Discriminative Learning of Classes (PDLC) head on pairs of embeddings
- Predict labels for the test set by aggregating pairwise "same-class" scores per class

Requirements:
- Full TabPFN package (pip install tabpfn)
- scikit-learn, numpy, torch

Notes:
- The TabPFN client (tabpfn-client) does not provide embeddings. Ensure the full package is installed.
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


from pdll import PairwiseDifferenceClassifier


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


def resolve_openml_dataset_id(name_or_id: str) -> int:
    try:
        return int(name_or_id.split(":", 1)[0])
    except Exception:
        pass
    name = name_or_id
    version = 1
    if ":" in name_or_id:
        parts = name_or_id.split(":", 1)
        name = parts[0]
        try:
            version = int(parts[1])
        except Exception:
            version = 1
    ds = openml.datasets.get_dataset(name=name, version=version)
    return ds.dataset_id


def find_classification_task_for_dataset(dataset_id: int, folds: int = 10) -> int:
    df = openml.tasks.list_tasks(output_format="dataframe", size=None)
    df = df[(df["did"] == dataset_id)]
    if "task_type" in df.columns:
        df = df[df["task_type"].str.contains("Supervised Classification", na=False)]
    elif "task_type_id" in df.columns:
        df = df[df["task_type_id"] == 1]
    if "NumberOfFolds" in df.columns:
        df = df[df["NumberOfFolds"] == folds]
    if "NumberOfRepeats" in df.columns:
        df = df.sort_values(["NumberOfRepeats"], ascending=[True])
    if df.empty:
        raise ValueError(f"No OpenML classification task with {folds}-folds for dataset {dataset_id}")
    return int(df.iloc[0]["tid"])


def load_openml_task_data(task_id: int) -> tuple[np.ndarray, np.ndarray, List[int], LabelEncoder, object]:
    task = openml.tasks.get_task(task_id)
    res = task.get_X_and_y(dataset_format="dataframe")
    if isinstance(res, tuple):
        if len(res) >= 2:
            X, y = res[0], res[1]
        else:
            raise ValueError("Unexpected return from task.get_X_and_y")
    else:
        # Some older APIs may return only X; not expected for classification tasks
        raise ValueError("task.get_X_and_y did not return (X, y)")
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    y = pd.Series(y)
    le = LabelEncoder()
    y_num = le.fit_transform(y)
    X_num, cat_idx = _preprocess_frame_to_numeric(X)
    return X_num, y_num, cat_idx, le, task


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


@dataclass
class PairSamplingConfig:
    n_pairs: int = 50000


class EmbeddingPairsDataset(torch.utils.data.Dataset):
    """Random on-the-fly sampling of ordered pairs, close to PDLL cross-merge.

    - Samples i, j uniformly from all training indices (includes self-pairs)
    - Target t = 1.0 if different class, 0.0 if same class (PDLL convention)
    - No class balancing; distribution reflects data priors
    """

    def __init__(
        self,
        E: np.ndarray,  # (N, D)
        y: np.ndarray,  # (N,)
        cfg: PairSamplingConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        assert E.ndim == 2
        assert E.shape[0] == y.shape[0]
        self.E = E.astype(np.float32, copy=False)
        self.y = y
        self.cfg = cfg
        self.rng = np.random.default_rng() if rng is None else rng
        self.N = E.shape[0]

    def __len__(self) -> int:
        return int(self.cfg.n_pairs)

    def __getitem__(self, _: int):
        i = int(self.rng.integers(0, self.N))
        j = int(self.rng.integers(0, self.N))
        e1 = torch.from_numpy(self.E[i])
        e2 = torch.from_numpy(self.E[j])
        t = torch.tensor([float(self.y[i] != self.y[j])], dtype=torch.float32)  # 1=different, 0=same
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


def train_pdlc_head(
    E_train: np.ndarray,
    y_train: np.ndarray,
    n_pairs: int = 50000,
    batch_size: int = 1024,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | torch.device | None = None,
    seed: int | None = None,
) -> PDLCHead:
    E_train = E_train.astype(np.float32, copy=False)
    emb_dim = E_train.shape[1]
    device = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Deterministic sampling and shuffling
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    cfg = PairSamplingConfig(n_pairs=n_pairs)
    dataset = EmbeddingPairsDataset(E_train, y_train, cfg, rng=np.random.default_rng(seed))
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
    loss_fn = nn.BCEWithLogitsLoss()  # target: 1=different, 0=same to match PDLL

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
    batch_train: int = 4096,
) -> np.ndarray:
    """Predict class labels for test embeddings using PDLC aggregation.

    For a test embedding e_t, compare against all training embeddings e_i.
    Aggregate sigmoid(logit) scores per class and pick the max-mean class.
    To improve symmetry, average scores for (e_t, e_i) and (e_i, e_t).
    """
    device = torch.device(device) if device is not None else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model = model.to(device).eval()

    E_train_t = torch.from_numpy(E_train.astype(np.float32, copy=False)).to(device)
    E_test_t = torch.from_numpy(E_test.astype(np.float32, copy=False)).to(device)

    classes = np.unique(y_train)
    class_to_mask: Dict[int, torch.Tensor] = {}
    for c in classes:
        class_to_mask[int(c)] = torch.tensor(y_train == c, device=device)

    preds: List[int] = []
    sigmoid = nn.Sigmoid()

    for t_idx in range(E_test_t.shape[0]):
        e_t = E_test_t[t_idx : t_idx + 1]  # (1, D)

        # We'll compute scores in blocks over training set to limit memory
        per_class_scores_sum = {int(c): 0.0 for c in classes}
        per_class_counts = {int(c): 0 for c in classes}

        num_train = E_train_t.shape[0]
        n_blocks = int(math.ceil(num_train / batch_train))
        for b in range(n_blocks):
            s = b * batch_train
            e = min(num_train, (b + 1) * batch_train)
            block = E_train_t[s:e]  # (B, D)

            # Forward for both orders; model outputs logit for 'different'
            logits1 = model(e_t.expand(block.shape[0], -1), block)  # (B, 1)
            logits2 = model(block, e_t.expand(block.shape[0], -1))
            p_diff = 0.5 * (sigmoid(logits1) + sigmoid(logits2)).squeeze(1)  # (B,)
            probs = 1.0 - p_diff  # similarity = P(same) = 1 - P(different)

            # Aggregate per class
            for c in classes:
                mask = class_to_mask[int(c)][s:e]
                if mask.any():
                    score = probs[mask].sum().item()
                    cnt = int(mask.sum().item())
                    per_class_scores_sum[int(c)] += score
                    per_class_counts[int(c)] += cnt

        # Compute mean score per class and choose the best (PDLL normalizes; argmax is invariant)
        class_means = {c: (per_class_scores_sum[c] / max(1, per_class_counts[c])) for c in classes}
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
    parser.add_argument(
        "--pairs",
        type=int,
        default=None,
        help="Deprecated/ignored: number of pairs now derived from dataset size",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=1024, help="Batch size for PDLC training")
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"], help="Device for TabPFN and PDLC")
    parser.add_argument("--folds", type=int, default=10, help="CV folds to use from OpenML task")
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

    # Resolve task and load data
    if hasattr(args, 'task_id') and args.task_id is not None:
        tid = args.task_id
    else:
        did = resolve_openml_dataset_id(args.dataset)
        tid = find_classification_task_for_dataset(did, folds=args.folds)
    print(f"Using OpenML task id: {tid}")
    X_all, y_all, cat_idx, le, task = load_openml_task_data(tid)

    # Accumulators across folds
    acc_et_raw, f1_et_raw = [], []
    acc_tabpfn_raw, f1_tabpfn_raw = [], []
    acc_et_emb, f1_et_emb = [], []
    acc_pdll, f1_pdll = [], []
    acc_pdll_raw, f1_pdll_raw = [], []
    acc_pdlc, f1_pdlc = [], []

    repeats = getattr(task, "repeat", 1) if hasattr(task, "repeat") else 1
    folds = getattr(task, "folds", args.folds) if hasattr(task, "folds") else args.folds
    for repeat in range(repeats):
        for fold in range(folds):
            tr_idx, te_idx = task.get_train_test_split_indices(repeat=repeat, fold=fold)
            X_train, X_test = X_all[tr_idx], X_all[te_idx]
            y_train, y_test = y_all[tr_idx], y_all[te_idx]

            # ExtraTrees on raw
            et_raw = ExtraTreesClassifier(class_weight='balanced', n_jobs=-1, random_state=args.seed)
            et_raw.fit(X_train, y_train)
            y_pred = et_raw.predict(X_test)
            acc_et_raw.append(accuracy_score(y_test, y_pred))
            f1_et_raw.append(f1_score(y_test, y_pred, average='macro'))

            # TabPFNClassifier on raw
            tp_clf = TabPFNClassifier(
                device=None if args.device=="auto" else args.device,
                random_state=args.seed,
                categorical_features_indices=cat_idx if len(cat_idx) > 0 else None,
            )
            tp_clf.fit(X_train, y_train)
            y_pred_tp = tp_clf.predict(X_test)
            acc_tabpfn_raw.append(accuracy_score(y_test, y_pred_tp))
            f1_tabpfn_raw.append(f1_score(y_test, y_pred_tp, average='macro'))

            # Embeddings via TabPFN
            emb_extractor = TabPFNEmbedding(tabpfn_clf=tp_clf, n_fold=args.n_fold)
            train_emb_raw = emb_extractor.get_embeddings(X_train, y_train, X_train, data_source="train")
            test_emb_raw = emb_extractor.get_embeddings(X_train, y_train, X_test, data_source="test")
            E_train = ensure_embeddings_2d(train_emb_raw, n_samples_expected=X_train.shape[0])
            E_test = ensure_embeddings_2d(test_emb_raw, n_samples_expected=X_test.shape[0])

            # ExtraTrees on embeddings
            et_emb = ExtraTreesClassifier(class_weight='balanced', n_jobs=-1, random_state=args.seed)
            et_emb.fit(E_train, y_train)
            y_pred_emb = et_emb.predict(E_test)
            acc_et_emb.append(accuracy_score(y_test, y_pred_emb))
            f1_et_emb.append(f1_score(y_test, y_pred_emb, average='macro'))

            # PDLL on embeddings
            df_train = pd.DataFrame(E_train)
            df_test = pd.DataFrame(E_test)
            pdll_clf = PairwiseDifferenceClassifier(
                estimator=ExtraTreesClassifier(class_weight='balanced', n_jobs=-1, random_state=args.seed),
            )
            pdll_clf.fit(df_train, y_train)
            y_pred_pdll = pdll_clf.predict(df_test)
            acc_pdll.append(accuracy_score(y_test, y_pred_pdll))
            f1_pdll.append(f1_score(y_test, y_pred_pdll, average='macro'))

            # PDLL on raw features
            df_train_raw = pd.DataFrame(X_train)
            df_test_raw = pd.DataFrame(X_test)
            pdll_raw = PairwiseDifferenceClassifier(
                estimator=ExtraTreesClassifier(class_weight='balanced', n_jobs=-1, random_state=args.seed),
            )
            pdll_raw.fit(df_train_raw, y_train)
            y_pred_pdll_raw = pdll_raw.predict(df_test_raw)
            acc_pdll_raw.append(accuracy_score(y_test, y_pred_pdll_raw))
            f1_pdll_raw.append(f1_score(y_test, y_pred_pdll_raw, average='macro'))

            # PDLC head on embeddings
            num_pairs = int(X_train.shape[0] * X_train.shape[0])
            pdlc = train_pdlc_head(
                E_train,
                y_train,
                n_pairs=num_pairs,
                batch_size=int(args.batch),
                epochs=int(args.epochs),
                device=None if args.device=="auto" else args.device,
                seed=args.seed,
            )
            y_pred_pdlc = predict_pdlc(pdlc, E_train, y_train, E_test, device=None if args.device=="auto" else args.device)
            acc_pdlc.append(accuracy_score(y_test, y_pred_pdlc))
            f1_pdlc.append(f1_score(y_test, y_pred_pdlc, average='macro'))

    def avg(xs):
        return float(np.mean(xs)) if len(xs) else float("nan")

    print("Results averaged over OpenML CV splits:")
    print(f"ExtraTrees raw     - Acc: {avg(acc_et_raw):.4f}, Macro-F1: {avg(f1_et_raw):.4f}")
    print(f"TabPFN raw        - Acc: {avg(acc_tabpfn_raw):.4f}, Macro-F1: {avg(f1_tabpfn_raw):.4f}")
    print(f"ExtraTrees embed  - Acc: {avg(acc_et_emb):.4f}, Macro-F1: {avg(f1_et_emb):.4f}")
    print(f"PDLL (embeddings) - Acc: {avg(acc_pdll):.4f}, Macro-F1: {avg(f1_pdll):.4f}")
    print(f"PDLL (raw)        - Acc: {avg(acc_pdll_raw):.4f}, Macro-F1: {avg(f1_pdll_raw):.4f}")
    #print(f"PDLC head         - Acc: {avg(acc_pdlc):.4f}, Macro-F1: {avg(f1_pdlc):.4f}")


if __name__ == "__main__":
    main()

"""Evaluate a trained PDLC head on OpenML datasets using 10-fold CV.

Loads a PDLC head checkpoint (trained on TabPFN embeddings) and evaluates it
across a list of OpenML dataset IDs. For each dataset, it finds the 10-fold
classification task, extracts TabPFN embeddings per split, predicts with the
PDLC head (using prior-adjusted aggregation), and logs mean accuracy and
macro-F1 across splits. Robust to OOM via CPU fallback and chunked anchors.
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from tabpfn_extensions import TabPFNClassifier
from tabpfn_extensions.embedding import TabPFNEmbedding


def resolve_device(device_arg: str | torch.device | None = "auto") -> torch.device:
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


def _preprocess_frame_to_numeric(
    X_df: "pd.DataFrame",
) -> Tuple[np.ndarray, List[int]]:
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
            col_cat = col.astype("string").fillna("__MISSING__").astype("category")
            codes = col_cat.cat.codes.astype(np.int64).to_numpy()
            X_proc[:, i] = codes.astype(np.float32)
            cat_idx.append(i)
    return X_proc, cat_idx


def load_openml_dataset_by_id(did: int) -> tuple[np.ndarray, np.ndarray, List[int], LabelEncoder]:
    X, y = fetch_openml(data_id=int(did), as_frame=True, return_X_y=True)
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    y = pd.Series(y)
    le = LabelEncoder()
    y_num = le.fit_transform(y)
    X_num, cat_idx = _preprocess_frame_to_numeric(X)
    return X_num, y_num, cat_idx, le


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
    device: str | torch.device | None = None,
    use_prior: bool | str = "auto",
    anchor_weights: np.ndarray | None = None,
) -> np.ndarray:
    device_t = resolve_device(device)
    model = model.to(device_t).eval()

    E_train_t = torch.from_numpy(E_train.astype(np.float32, copy=False)).to(device_t)
    E_test_t = torch.from_numpy(E_test.astype(np.float32, copy=False)).to(device_t)

    classes = np.unique(y_train)
    class_to_mask: dict[int, torch.Tensor] = {}
    for c in classes:
        class_to_mask[int(c)] = torch.tensor(y_train == c, device=device_t)

    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    anchor_class_idx = np.array([class_to_idx[int(c)] for c in y_train], dtype=np.int64)
    anchor_class_idx_t = torch.tensor(anchor_class_idx, device=device_t, dtype=torch.long)

    _, counts = np.unique(y_train, return_counts=True)
    priors_np = counts.astype(np.float64) / float(counts.sum())
    priors_t = torch.tensor(priors_np, device=device_t, dtype=torch.float32)

    if isinstance(use_prior, str) and use_prior.lower() == "auto":
        use_prior_flag = bool(counts.min() < 5)
    else:
        use_prior_flag = bool(use_prior)

    N_train = E_train.shape[0]
    if anchor_weights is None:
        w_np = np.full(N_train, 1.0 / max(1, N_train), dtype=np.float64)
    else:
        w_np = np.asarray(anchor_weights, dtype=np.float64)
        assert w_np.shape[0] == N_train
        w_np = np.clip(w_np, 0.0, None)
        s = float(w_np.sum())
        w_np = w_np / s if s > 0 else np.full(N_train, 1.0 / max(1, N_train), dtype=np.float64)
    w_t = torch.tensor(w_np.astype(np.float32), device=device_t)

    preds: List[int] = []
    sigmoid = nn.Sigmoid()
    for t_idx in range(E_test_t.shape[0]):
        e_t = E_test_t[t_idx : t_idx + 1]

        if use_prior_flag:
            test_class_scores = torch.zeros(len(classes), device=device_t, dtype=torch.float32)
        else:
            per_class_scores_sum = {int(c): 0.0 for c in classes}
            per_class_weight_sum = {int(c): 0.0 for c in classes}

        num_train = E_train_t.shape[0]
        s = 0
        e = num_train
        block = E_train_t[s:e]

        logits1 = model(e_t.expand(block.shape[0], -1), block)
        logits2 = model(block, e_t.expand(block.shape[0], -1))
        p_same = 0.5 * (sigmoid(logits1) + sigmoid(logits2)).squeeze(1)

        if use_prior_flag:
            c_idx_blk = anchor_class_idx_t[s:e]
            w_blk = w_t[s:e]
            B = c_idx_blk.shape[0]
            one_minus_s = (1.0 - p_same).unsqueeze(1)
            denom = (1.0 - priors_t[c_idx_blk]).unsqueeze(1) + 1e-12
            L = one_minus_s * priors_t.unsqueeze(0) / denom
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


def evaluate_datasets_with_ckpt(
    dataset_ids: List[int],
    ckpt_path: str,
    n_fold: int,
    seed: int,
    device: str | torch.device | None = None,
    log_csv: str | None = None,
) -> pd.DataFrame:
    dev = resolve_device(device)
    rows: List[dict] = []
    first_model_built = False
    model: PDLCHead | None = None
    state = torch.load(ckpt_path, map_location=dev)

    for did in dataset_ids:
        name = str(int(did))
        try:
            X_all, y_all, cat_idx, le = load_openml_dataset_by_id(int(name))
            n_features = X_all.shape[1]
            n_classes = int(len(np.unique(y_all)))
            if n_features > 500 or n_classes > 10:
                rows.append({"dataset": name, "status": "skipped", "reason": f"hard limits: features={n_features}, classes={n_classes}"})
                continue

            # Stratified 10-fold CV without OpenML task lookup
            accs: List[float] = []
            f1s: List[float] = []
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            for tr_idx, te_idx in skf.split(X_all, y_all):
                X_train, X_test = X_all[tr_idx], X_all[te_idx]
                y_train, y_test = y_all[tr_idx], y_all[te_idx]
                def _extract(use_cpu: bool = False):
                    clf = TabPFNClassifier(n_estimators=1, random_state=seed, inference_config={"PREPROCESS_TRANSFORMS": [{"name": "none", "categorical_name": "numeric", "append_original": False, "subsample_features": -1, "global_transformer_name": None}]})
                    emb = TabPFNEmbedding(tabpfn_clf=clf, n_fold=n_fold)
                    E_tr_raw = emb.get_embeddings(X_train, y_train, X_train, data_source="train")
                    E_te_raw = emb.get_embeddings(X_train, y_train, X_test, data_source="test")
                    E_tr = ensure_embeddings_2d(E_tr_raw, n_samples_expected=X_train.shape[0])
                    E_te = ensure_embeddings_2d(E_te_raw, n_samples_expected=X_test.shape[0])
                    return E_tr, E_te

                # No CPU fallback: fail fast on OOM
                E_train, E_test = _extract(use_cpu=False)

                emb_dim = int(E_train.shape[1])
                if not first_model_built:
                    model = PDLCHead(emb_dim=emb_dim, dropout=0.1)
                    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
                    model = model.to(dev).eval()
                    first_model_built = True
                else:
                    if model is not None and model.net[0].in_features != 2 * emb_dim:
                        rows.append({
                            "dataset": name,
                            "status": "skipped",
                            "reason": f"embedding dim {emb_dim} incompatible with loaded head",
                        })
                        accs = []
                        f1s = []
                        break

                if model is None:
                    rows.append({"dataset": name, "status": "error", "reason": "model not initialized"})
                    accs = []
                    f1s = []
                    break

                # No CPU fallback: compare against all anchors at once
                y_pred = predict_pdlc(model, E_train, y_train, E_test, device=dev, use_prior=True)

                accs.append(float(accuracy_score(y_test, y_pred)))
                f1s.append(float(f1_score(y_test, y_pred, average="macro")))

            if accs == []:
                continue

            rows.append({
                "dataset": name,
                "status": "ok",
                "acc_mean": float(np.mean(accs)),
                "f1_macro_mean": float(np.mean(f1s)),
                "splits": int(len(accs)),
            })
        except ValueError as e:
            rows.append({"dataset": name, "status": "skipped", "reason": str(e)})
        except (RuntimeError, MemoryError) as e:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            rows.append({"dataset": name, "status": "oom", "reason": f"{type(e).__name__}: {e}"})
        except Exception as e:
            rows.append({"dataset": name, "status": "error", "reason": f"{type(e).__name__}: {e}"})

    df = pd.DataFrame(rows)
    if log_csv is not None:
        df.to_csv(log_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate PDLC head on OpenML datasets (10-fold CV)")
    parser.add_argument("--ckpt", type=str, default="pdlc_multi.pt", help="Path to PDLC head checkpoint")
    parser.add_argument("--dataset_ids", type=str, default=None, help="Comma-separated OpenML dataset IDs; if omitted, use ALL_DATASET_IDS constant inside this script")
    parser.add_argument("--n_fold", type=int, default=0, help="n_fold used for embedding extraction (0=vanilla, >=2=K-fold)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    # All anchors are compared at once; no chunk size argument
    parser.add_argument("--eval_log_csv", type=str, default="pdlc_eval_log.csv")
    args = parser.parse_args()

    # Default dataset IDs (edit here if desired)
    if args.dataset_ids is None:
        dataset_ids = [
            43, 48, 59, 61, 164, 333, 377, 444, 464, 475, 714, 717, 721, 733, 736,
            744, 750, 756, 766, 767, 768, 773, 779, 782, 784, 788, 792, 793, 811,
            812, 814, 824, 850, 853, 860, 863, 870, 873, 877, 879, 880, 889, 895,
            896, 902, 906, 909, 911, 915, 918, 925, 932, 933, 935, 936, 937, 969,
            973, 974, 1005, 1011, 1012, 1054, 1063, 1065, 1073, 1100, 1115, 1413,
            1467, 1480, 1488, 1490, 1499, 1510, 1511, 1523, 1554, 1556, 1600, 4329,
            40663, 40681, 41568, 41977, 41978, 42011, 42021, 42026, 42051, 42066,
            42071, 42186, 42700, 43859, 44149, 44151, 44344, 45711, 1049, 1067, 12,
            1464, 1475, 1487, 1489, 1494, 181, 188, 23, 31, 3, 40498, 40670, 40701,
            40900, 40975, 40981, 40982, 40983, 40984, 41143, 41144, 41145, 41146,
            41156, 4538, 54,
        ]
    else:
        dataset_ids = [int(x.strip()) for x in args.dataset_ids.split(",") if x.strip()]

    dev = resolve_device(args.device)
    df = evaluate_datasets_with_ckpt(
        dataset_ids=dataset_ids,
        ckpt_path=args.ckpt,
        n_fold=args.n_fold,
        seed=args.seed,
        device=dev,
        log_csv=args.eval_log_csv,
    )
    ok = df[df["status"] == "ok"]
    mean_acc = float(ok["acc_mean"].mean()) if not ok.empty else float("nan")
    mean_f1 = float(ok["f1_macro_mean"].mean()) if not ok.empty else float("nan")
    print(f"Evaluated {len(df)} datasets ({len(ok)} ok). Mean acc={mean_acc:.4f}, macro-F1={mean_f1:.4f}. Logs: {args.eval_log_csv}")


if __name__ == "__main__":
    main()

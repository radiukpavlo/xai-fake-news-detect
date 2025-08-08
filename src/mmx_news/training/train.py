from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from ..data.loaders import Article, _read_isot, _read_smoke, stratified_splits
from ..features.mechanisms import EmbeddingBackend, FeatureComputer
from ..models.classifiers import make_classifier
from ..models.transition import fit_transition
from ..utils.constants import INV_LABEL_MAP, LABEL_MAP
from ..utils.io import dump_json
from .utils import init_run


def _load_articles(cfg: Dict, mode: str) -> List[Article]:
    if mode == "smoke":
        p = Path(cfg["data"]["smoke"]["path"])
        return _read_smoke(p)
    else:
        root = Path(cfg["data"]["root"])
        return _read_isot(root)


def _segment_lead(text: str, max_sent: int) -> Tuple[str, List[str]]:
    from ..data.preprocess import simple_sentence_split

    sents = simple_sentence_split(text)
    lead = " ".join(sents[: min(2, len(sents))])  # first two sentences
    body_sents = sents[:max_sent]
    return lead, body_sents


def _build_embeddings(arts: List[Article], cfg: Dict) -> np.ndarray:
    max_body_sent = int(cfg["embeddings"]["max_body_sentences"])
    w_title, w_lead, w_body = map(float, cfg["embeddings"]["weights"])
    backend = cfg["embeddings"].get("backend", "auto")
    model_name = cfg["embeddings"]["model_name"]
    emb = EmbeddingBackend(backend=backend, model_name=model_name)

    embs = []
    for a in arts:
        lead, body_sents = _segment_lead(a.text, max_body_sent)
        parts = [a.title, lead, " ".join(body_sents)]
        e = emb.encode(parts)
        title_vec, lead_vec, body_vec = e[0], e[1], np.mean(emb.encode(body_sents or [""]), axis=0)
        z = w_title * title_vec + w_lead * lead_vec + w_body * body_vec
        n = np.linalg.norm(z) + 1e-12
        embs.append(z / n)
    return np.asarray(embs, dtype=np.float32)


def _load_lexicon(path: str | Path) -> List[str]:
    try:
        return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []


def _build_features(arts: List[Article], cfg: Dict) -> Tuple[np.ndarray, List[Dict]]:
    pos_lex = _load_lexicon(cfg["features"]["sentiment_lexicons"]["pos"])
    neg_lex = _load_lexicon(cfg["features"]["sentiment_lexicons"]["neg"])
    ul_paths = [Path(p) for p in cfg["features"]["ul_lexicons"]]
    emb = EmbeddingBackend(backend=cfg["embeddings"]["backend"], model_name=cfg["embeddings"]["model_name"])
    fc = FeatureComputer(emb_backend=emb, pos_lex=pos_lex, neg_lex=neg_lex, ul_lexicons=ul_paths)

    feats: List[List[float]] = []
    evidences: List[Dict] = []
    for a in arts:
        lead, _ = _segment_lead(a.text, int(cfg["embeddings"]["max_body_sentences"]))
        pr = fc.paraphrasing_ratio(a.text)
        sr = fc.subjectivity_ratio(a.text, threshold=float(cfg["features"]["subjectivity_threshold"]))
        hs = fc.headline_lead_coherence(a.title, lead)
        ul = fc.unusual_language_share(a.text)
        sp, nc = fc.sentiment_and_consistency(a.text)
        sq = fc.selective_quoting(a.text)

        raw_values = [pr.value, sr.value, hs.value, ul.value, sp.value, nc.value, sq.value]
        feats.append(raw_values)
        evidences.append(
            {
                "id": a.id,
                "features": [
                    {"name": pr.name, "value": pr.value, "evidence": pr.evidence.__dict__, "meta": pr.meta},
                    {"name": sr.name, "value": sr.value, "evidence": sr.evidence.__dict__, "meta": sr.meta},
                    {"name": hs.name, "value": hs.value, "evidence": hs.evidence.__dict__, "meta": hs.meta},
                    {"name": ul.name, "value": ul.value, "evidence": ul.evidence.__dict__, "meta": ul.meta},
                    {"name": sp.name, "value": sp.value, "evidence": sp.evidence.__dict__, "meta": sp.meta},
                    {"name": nc.name, "value": nc.value, "evidence": nc.evidence.__dict__, "meta": nc.meta},
                    {"name": sq.name, "value": sq.value, "evidence": sq.evidence.__dict__, "meta": sq.meta},
                ],
            }
        )
    return np.asarray(feats, dtype=np.float32), evidences


def _minmax_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    # avoid zero width
    width = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    return lo, width


def _minmax_apply(X: np.ndarray, lo: np.ndarray, width: np.ndarray) -> np.ndarray:
    return (X - lo) / width


def run_training(cfg: Dict, seed: int, mode: str = "full") -> None:
    run_id, out_dir = init_run(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    arts = _load_articles(cfg, mode=mode)
    y = np.array([a.label for a in arts], dtype=int)

    # Build splits
    n_splits = int(cfg["data"]["splits"])
    split_ratio = tuple(cfg["data"]["split_ratio"])
    seed_list = list(cfg["data"]["seed_list"])
    splits = stratified_splits(
        arts,
        n_splits=n_splits,
        split_ratio=split_ratio,
        seed_list=seed_list,
        dedupe_enabled=bool(cfg["data"]["dedupe"]["enabled"]),
        dedupe_threshold_bits=int(cfg["data"]["dedupe"]["threshold_bits"]),
    )

    # Use the split corresponding to the provided seed (if in list); else use first
    if seed in splits:
        idx = splits[seed]
    else:
        idx = next(iter(splits.values()))

    # Embeddings (A) and Interpretable features (B)
    A = _build_embeddings(arts, cfg)
    B_raw, evidences = _build_features(arts, cfg)

    # Calibration (min-max on train only)
    lo, width = _minmax_fit(B_raw[idx["train_idx"]])
    B = _minmax_apply(B_raw, lo, width)

    # Transition on training
    T = fit_transition(A[idx["train_idx"]], B[idx["train_idx"]], tol=float(cfg["transition"]["svd_tolerance"]))

    # Projected features
    Bhat = A @ T

    # Select features for training/evaluation
    y_train = y[idx["train_idx"]]
    y_val = y[idx["val_idx"]]
    y_test = y[idx["test_idx"]]

    B_train = B[idx["train_idx"]]
    B_val = B[idx["val_idx"]]
    B_test = B[idx["test_idx"]]

    Bhat_train = Bhat[idx["train_idx"]]
    Bhat_val = Bhat[idx["val_idx"]]
    Bhat_test = Bhat[idx["test_idx"]]

    # Train classifier on B (primary)
    clf = make_classifier(
        cfg["classifier"]["type"],
        C=float(cfg["classifier"]["C"]),
        gamma=float(cfg["classifier"].get("gamma", 0.5)),
    )
    clf.fit(B_train, y_train)

    # Evaluate on validation using projected features (fidelity) and B
    y_pred_val = clf.predict(B_val)
    report_val = classification_report(y_val, y_pred_val, digits=4, zero_division=0, target_names=["fake", "real"])
    if hasattr(clf, "decision_function"):
        scores_val = clf.decision_function(B_val)
        auc_val = float(roc_auc_score(y_val, scores_val))
    else:
        auc_val = float("nan")

    # Evaluate on test
    y_pred_test = clf.predict(B_test)
    report_test = classification_report(y_test, y_pred_test, digits=4, zero_division=0, target_names=["fake", "real"])
    if hasattr(clf, "decision_function"):
        scores_test = clf.decision_function(B_test)
        auc_test = float(roc_auc_score(y_test, scores_test))
    else:
        auc_test = float("nan")

    # Confusion matrices
    cm_val = confusion_matrix(y_val, y_pred_val).tolist()
    cm_test = confusion_matrix(y_test, y_pred_test).tolist()

    # Persist artifacts
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "A.npy", A)
    np.save(out_dir / "B.npy", B)
    np.save(out_dir / "B_raw.npy", B_raw)
    np.save(out_dir / "Bhat.npy", Bhat)
    np.save(out_dir / "split_indices.npy", np.stack([idx["train_idx"], idx["val_idx"], idx["test_idx"]], axis=0))
    np.save(out_dir / "minmax_lo.npy", lo)
    np.save(out_dir / "minmax_width.npy", width)

    # Evidence per article
    dump_json(evidences, out_dir / "evidence.json")

    # Transition and model
    np.save(out_dir / "T.npy", T)
    import joblib  # ensure on demand

    joblib.dump(clf, out_dir / "best.joblib")

    # Reports
    with open(out_dir / "val_report.txt", "w", encoding="utf-8") as f:
        f.write(report_val + f"\nAUC={auc_val:.4f}\n")
    with open(out_dir / "test_report.txt", "w", encoding="utf-8") as f:
        f.write(report_test + f"\nAUC={auc_test:.4f}\n")

    # Transition fidelity: distance correlation between B and Bhat on validation
    from scipy.spatial.distance import pdist

    dist_B = pdist(B_val, metric="euclidean")
    dist_Bhat = pdist(Bhat_val, metric="euclidean")
    corr = float(np.corrcoef(dist_B, dist_Bhat)[0, 1])
    fidelity = {"pearson_corr_euclid_B_vs_Bhat_val": corr}
    dump_json(fidelity, out_dir / "transition_fidelity.json")

    # Export metrics csv
    metrics_row = {
        "seed": seed,
        "precision_val": float(np.nan_to_num(np.mean((y_pred_val == 1) & (y_val == 1)))),
        "precision_test": float(np.nan_to_num(np.mean((y_pred_test == 1) & (y_test == 1)))),
        "auc_val": auc_val,
        "auc_test": auc_test,
    }
    pd.DataFrame([metrics_row]).to_csv(out_dir / "metrics.csv", index=False)


__all__ = ["run_training"]

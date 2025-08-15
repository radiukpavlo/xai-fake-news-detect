from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..data.loaders import Article, _read_isot, _read_smoke, stratified_splits


from ..features.mechanisms import EmbeddingBackend, FeatureComputer


def _segment_lead(text: str, max_sent: int) -> Tuple[str, List[str]]:
    from ..data.preprocess import simple_sentence_split

    sents = simple_sentence_split(text)
    lead = " ".join(sents[: min(2, len(sents))])  # first two sentences
    body_sents = sents[:max_sent]
    return lead, body_sents


def _build_embeddings(arts: List[Article], cfg: Config) -> np.ndarray:
    max_body_sent = cfg.embeddings.max_body_sentences
    w_title, w_lead, w_body = cfg.embeddings.weights
    backend = cfg.embeddings.backend
    model_name = cfg.embeddings.model_name
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


def _build_features(arts: List[Article], cfg: Config) -> Tuple[np.ndarray, List[Dict]]:
    pos_lex = _load_lexicon(cfg.features.sentiment_lexicons.pos)
    neg_lex = _load_lexicon(cfg.features.sentiment_lexicons.neg)
    ul_paths = [Path(p) for p in cfg.features.ul_lexicons]
    emb = EmbeddingBackend(backend=cfg.embeddings.backend, model_name=cfg.embeddings.model_name)
    fc = FeatureComputer(emb_backend=emb, pos_lex=pos_lex, neg_lex=neg_lex, ul_lexicons=ul_paths)

    feats: List[List[float]] = []
    evidences: List[Dict] = []
    for a in arts:
        lead, _ = _segment_lead(a.text, cfg.embeddings.max_body_sentences)
        pr = fc.paraphrasing_ratio(a.text)
        sr = fc.subjectivity_ratio(a.text, threshold=cfg.features.subjectivity_threshold)
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


from ..models.classifiers import make_classifier
from ..models.transition import fit_transition


def _minmax_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    # avoid zero width
    width = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    return lo, width


def _minmax_apply(X: np.ndarray, lo: np.ndarray, width: np.ndarray) -> np.ndarray:
    return (X - lo) / width


def build_features_and_embeddings(
    articles: List[Article], cfg: Config
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Builds embeddings and features for the articles."""
    A = _build_embeddings(articles, cfg)
    B_raw, evidences = _build_features(articles, cfg)
    return A, B_raw, evidences


from sklearn.metrics import classification_report, roc_auc_score


def train_model(
    A_train: np.ndarray, B_train: np.ndarray, y_train: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, Any, np.ndarray, np.ndarray]:
    """Trains the transition model and the classifier."""
    # Calibration (min-max on train only)
    lo, width = _minmax_fit(B_train)
    B_train_scaled = _minmax_apply(B_train, lo, width)

    # Transition on training
    T = fit_transition(A_train, B_train_scaled, tol=cfg.transition.svd_tolerance)

    # Train classifier on B (primary)
    clf = make_classifier(
        cfg.classifier.type,
        C=cfg.classifier.C,
        gamma=cfg.classifier.gamma,
    )
    clf.fit(B_train_scaled, y_train)

    return T, clf, lo, width


import joblib
import pandas as pd
from ..utils.io import dump_json


def evaluate_model(
    clf: Any,
    B_val: np.ndarray,
    y_val: np.ndarray,
    B_test: np.ndarray,
    y_test: np.ndarray,
    lo: np.ndarray,
    width: np.ndarray,
) -> Tuple[str, str, float, float]:
    """Evaluates the model on the validation and test sets."""
    B_val_scaled = _minmax_apply(B_val, lo, width)
    B_test_scaled = _minmax_apply(B_test, lo, width)

    # Evaluate on validation using projected features (fidelity) and B
    y_pred_val = clf.predict(B_val_scaled)
    report_val = classification_report(y_val, y_pred_val, digits=4, zero_division=0, target_names=["fake", "real"])
    if hasattr(clf, "decision_function"):
        scores_val = clf.decision_function(B_val_scaled)
        auc_val = float(roc_auc_score(y_val, scores_val))
    else:
        auc_val = float("nan")

    # Evaluate on test
    y_pred_test = clf.predict(B_test_scaled)
    report_test = classification_report(y_test, y_pred_test, digits=4, zero_division=0, target_names=["fake", "real"])
    if hasattr(clf, "decision_function"):
        scores_test = clf.decision_function(B_test_scaled)
        auc_test = float(roc_auc_score(y_test, scores_test))
    else:
        auc_test = float("nan")

    return report_val, report_test, auc_val, auc_test


def save_artifacts(
    out_dir: Path,
    A: np.ndarray,
    B: np.ndarray,
    B_raw: np.ndarray,
    Bhat: np.ndarray,
    idx: Dict[str, np.ndarray],
    lo: np.ndarray,
    width: np.ndarray,
    evidences: List[Dict],
    T: np.ndarray,
    clf: Any,
    report_val: str,
    report_test: str,
    auc_val: float,
    auc_test: float,
    seed: int,
    articles: List[Article],
) -> None:
    """Saves all the artifacts from the training run."""
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "A.npy", A)
    np.save(out_dir / "B.npy", B)
    np.save(out_dir / "B_raw.npy", B_raw)
    np.save(out_dir / "Bhat.npy", Bhat)
    np.savez(out_dir / "split_indices.npz", **idx)
    np.save(out_dir / "minmax_lo.npy", lo)
    np.save(out_dir / "minmax_width.npy", width)

    # Evidence per article
    dump_json(evidences, out_dir / "evidence.json")

    # Transition and model
    np.save(out_dir / "T.npy", T)

    joblib.dump(clf, out_dir / "best.joblib")

    # Reports
    with open(out_dir / "val_report.txt", "w", encoding="utf-8") as f:
        f.write(report_val + f"\nAUC={auc_val:.4f}\n")
    with open(out_dir / "test_report.txt", "w", encoding="utf-8") as f:
        f.write(report_test + f"\nAUC={auc_test:.4f}\n")

    # Transition fidelity: distance correlation between B and Bhat on validation
    from scipy.spatial.distance import pdist

    B_val = B[idx["val_idx"]]
    Bhat_val = Bhat[idx["val_idx"]]
    dist_B = pdist(B_val, metric="euclidean")
    dist_Bhat = pdist(Bhat_val, metric="euclidean")
    corr = float(np.corrcoef(dist_B, dist_Bhat)[0, 1])
    fidelity = {"pearson_corr_euclid_B_vs_Bhat_val": corr}
    dump_json(fidelity, out_dir / "transition_fidelity.json")

    # Export metrics csv
    y_pred_val = clf.predict(_minmax_apply(B[idx["val_idx"]], lo, width))
    y_pred_test = clf.predict(_minmax_apply(B[idx["test_idx"]], lo, width))
    y_val = np.array([a.label for a in articles])[idx["val_idx"]]
    y_test = np.array([a.label for a in articles])[idx["test_idx"]]

    metrics_row = {
        "seed": seed,
        "precision_val": float(np.nan_to_num(np.mean((y_pred_val == 1) & (y_val == 1)))),
        "precision_test": float(np.nan_to_num(np.mean((y_pred_test == 1) & (y_test == 1)))),
        "auc_val": auc_val,
        "auc_test": auc_test,
    }
    pd.DataFrame([metrics_row]).to_csv(out_dir / "metrics.csv", index=False)


from ..utils.config import Config


def prepare_data(cfg: Config, mode: str) -> Tuple[List[Article], np.ndarray, Dict[str, np.ndarray]]:
    """Loads articles and creates splits."""
    if mode == "smoke":
        p = Path(cfg.data.smoke.path)
        articles = _read_smoke(p)
    else:
        root = Path(cfg.data.root)
        articles = _read_isot(root)

    y = np.array([a.label for a in articles], dtype=int)

    # Build splits
    splits = stratified_splits(
        articles,
        n_splits=cfg.data.splits,
        split_ratio=cfg.data.split_ratio,
        seed_list=cfg.data.seed_list,
        dedupe_enabled=cfg.data.dedupe.enabled,
        dedupe_threshold_bits=cfg.data.dedupe.threshold_bits,
    )

    # Use the split corresponding to the provided seed (if in list); else use first
    # This is not ideal, but it's how the original code worked.
    # A better approach would be to have the seed as an argument to this function.
    # For now, I will keep the original logic.
    seed = cfg.repro.global_seed
    if seed in splits:
        idx = splits[seed]
    else:
        idx = next(iter(splits.values()))

    return articles, y, idx

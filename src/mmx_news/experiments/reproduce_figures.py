from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE


def make_plots(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    B = np.load(run_dir / "B.npy")
    Bhat = np.load(run_dir / "Bhat.npy")
    splits = np.load(run_dir / "split_indices.npy")
    test_idx = splits[2]

    def plot_and_save(X: np.ndarray, name: str) -> None:
        pca2 = PCA(n_components=2).fit_transform(X)
        plt.figure()
        plt.scatter(pca2[:, 0], pca2[:, 1], s=8)
        plt.title(name)
        plt.savefig(run_dir / "plots" / f"{name}_pca.png", dpi=150)
        plt.close()

    plot_and_save(B[test_idx], "B_test")
    plot_and_save(Bhat[test_idx], "Bhat_test")

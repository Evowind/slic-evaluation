"""
Parameter tuning rigoureux pour SLIC
Comparaison multi-métrique, normalisée et contrôlée
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

from src.methods.slic.slic_original import SLIC
from src.preprocessing.image_loader import ImageLoader
from src.evaluation.metrics import compute_metrics_multiple_gt


# ------------------------------------------------------------
# UTILITAIRES
# ------------------------------------------------------------

def normalize(values, higher_is_better=True, eps=1e-8):
    values = np.array(values, dtype=np.float64)
    vmin, vmax = values.min(), values.max()

    if vmax - vmin < eps:
        return np.ones_like(values)

    norm = (values - vmin) / (vmax - vmin)
    return norm if higher_is_better else 1.0 - norm


def within_tolerance(real, target, tol_ratio=0.1):
    return abs(real - target) / target <= tol_ratio


# ------------------------------------------------------------
# GRID SEARCH COMPARATIF
# ------------------------------------------------------------

def grid_search(
    image,
    ground_truths,
    n_segments_values,
    compactness_values,
    target_n_segments,
    n_tol_ratio=0.1,
    max_iter=10
):
    """
    Recherche en grille avec :
    - contrôle du nombre réel de superpixels
    - score multi-métrique normalisé
    - pénalisation de la variance
    """

    print("=" * 70)
    print("GRID SEARCH COMPARATIF (RIGOUREUX)")
    print("=" * 70)

    raw_results = []

    for n_seg, comp in product(n_segments_values, compactness_values):
        print(f"Test n_segments={n_seg}, compactness={comp}")

        start = time.time()
        slic = SLIC(n_segments=n_seg, compactness=comp, max_iter=max_iter)
        labels = slic.fit(image)
        elapsed = time.time() - start

        metrics = compute_metrics_multiple_gt(labels, ground_truths)

        n_real = metrics['n_superpixels']

        if not within_tolerance(n_real, target_n_segments, n_tol_ratio):
            print(f"  Rejeté (n_real={n_real})")
            continue

        raw_results.append({
            'params': (n_seg, comp),
            'time': elapsed,
            'metrics': metrics
        })

        print(
            f"  n_real={n_real}, "
            f"ASA={metrics['asa']:.4f}, "
            f"UE={metrics['under_segmentation_error']:.4f}"
        )

    if len(raw_results) == 0:
        raise RuntimeError("Aucune configuration valide après filtrage n_superpixels")

    return compute_scores(raw_results)


# ------------------------------------------------------------
# SCORE MULTI-MÉTRIQUE
# ------------------------------------------------------------

def compute_scores(results):
    """
    Calcule un score composite normalisé et stable
    """

    asa = [r['metrics']['asa'] for r in results]
    ue = [r['metrics']['under_segmentation_error'] for r in results]
    cue = [r['metrics']['corrected_under_segmentation_error'] for r in results]
    ev = [r['metrics']['explained_variation'] for r in results]
    br = [r['metrics']['boundary_recall'] for r in results]

    asa_std = [r['metrics']['asa_std'] for r in results]
    ue_std = [r['metrics']['under_segmentation_error_std'] for r in results]
    br_std = [r['metrics']['boundary_recall_std'] for r in results]

    time_vals = [r['time'] for r in results]

    # Normalisation
    asa_n = normalize(asa, True)
    ev_n = normalize(ev, True)
    br_n = normalize(br, True)

    ue_n = normalize(ue, False)
    cue_n = normalize(cue, False)

    asa_std_n = normalize(asa_std, False)
    ue_std_n = normalize(ue_std, False)
    br_std_n = normalize(br_std, False)

    time_n = normalize(time_vals, False)

    for i, r in enumerate(results):
        score = (
            1.5 * asa_n[i] +
            1.0 * ev_n[i] +
            0.5 * br_n[i] +
            1.0 * ue_n[i] +
            0.5 * cue_n[i] +
            0.5 * asa_std_n[i] +
            0.5 * ue_std_n[i] +
            0.25 * br_std_n[i] +
            0.5 * time_n[i]
        )

        r['score'] = score

    results.sort(key=lambda x: x['score'], reverse=True)

    return results


# ------------------------------------------------------------
# VISUALISATION
# ------------------------------------------------------------

def visualize_top_k(results, image, k=6):
    from skimage.segmentation import mark_boundaries

    k = min(k, len(results))
    fig, axes = plt.subplots(1, k, figsize=(4 * k, 4))

    if image.max() > 1.0:
        image = image / 255.0

    for i in range(k):
        r = results[i]
        slic = SLIC(
            n_segments=r['params'][0],
            compactness=r['params'][1],
            max_iter=10
        )
        labels = slic.fit(image)

        marked = mark_boundaries(image, labels, mode='thick')
        axes[i].imshow(marked)
        axes[i].set_title(
            f"n={r['params'][0]}, m={r['params'][1]}\nScore={r['score']:.3f}"
        )
        axes[i].axis('off')

    plt.tight_layout()
    return fig


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser("SLIC parameter tuning rigoureux")
    parser.add_argument('--image', required=True)
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    loader = ImageLoader()
    image = loader.load_image(args.image)

    image_name = Path(args.image).stem
    gt_data = loader.load_bsds500_groundtruth(
        split='test',
        image_name=image_name
    )
    ground_truths = gt_data[0]

    results = grid_search(
        image=image,
        ground_truths=ground_truths,
        n_segments_values=[100, 150, 200, 250, 300],
        compactness_values=[5, 10, 15, 20],
        target_n_segments=200,
        n_tol_ratio=0.1
    )

    print("\n" + "=" * 70)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 70)

    for r in results[:5]:
        print(
            f"n={r['params'][0]}, m={r['params'][1]}, "
            f"score={r['score']:.4f}, "
            f"ASA={r['metrics']['asa']:.4f}, "
            f"UE={r['metrics']['under_segmentation_error']:.4f}"
        )

    fig = visualize_top_k(results, image, k=6)

    if args.save:
        out = Path("results/slic/parameters")
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / "grid_search_topk.png", dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    main()

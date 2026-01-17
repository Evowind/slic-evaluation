"""
Script de comparaison entre SLIC original et SLIC IPOL amélioré
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from src.methods.slic.slic_original import SLIC
from src.methods.slic.slic_ipol import SLIC_IPOL
from src.preprocessing.image_loader import ImageLoader
from src.evaluation.metrics import compute_all_metrics, format_metrics, boundary_recall, under_segmentation_error
from src.evaluation.visualize import visualize_segmentation


def compare_versions(image_path, n_segments=200, compactness=10, gt_dir=None):
    """
    Compare SLIC original vs SLIC IPOL.
    
    Args:
        image_path: Path to image
        n_segments: Number of superpixels
        compactness: Compactness parameter
    """
    print("="*70)
    print("COMPARAISON SLIC ORIGINAL vs SLIC IPOL")
    print("="*70)
    
    # Load image
    loader = ImageLoader()
    image = loader.load_image(image_path)
    print(f"\nImage: {image_path}")
    print(f"Size: {image.shape}")
    print(f"Parameters: n_segments={n_segments}, compactness={compactness}\n")

    # =====================================================================
    # LOAD GROUND TRUTH (BSDS500)
    # =====================================================================
    gt_list = None
    if gt_dir is not None:
        print("\nChargement du ground truth BSDS500…")

        loader_gt = ImageLoader(data_dir=gt_dir)

        image_name = Path(image_path).stem
        try:
            gt_all = loader_gt.load_bsds500_groundtruth(split='global', image_name=image_name)
            if len(gt_all) > 0:
                gt_list = gt_all[0]
                print(f"  Ground truth trouvé ({len(gt_list)} segmentations)")
            else:
                print("  Aucun GT trouvé")
        except Exception as e:
            print(f"  Erreur chargement GT : {e}")
            gt_list = None

    
    # =========================================================================
    # SLIC ORIGINAL
    # =========================================================================
    print("\n" + "-"*70)
    print("1. SLIC ORIGINAL")
    print("-"*70)
    
    slic_original = SLIC(
        n_segments=n_segments,
        compactness=compactness,
        max_iter=10,
    )
    
    start_time = time.time()
    labels_original = slic_original.fit(image)
    time_original = time.time() - start_time
    
    metrics_original = compute_all_metrics(labels_original)
    
    print(f"\nRésultats SLIC Original:")
    print(f"  Temps d'exécution: {time_original:.3f}s")
    print(f"  Superpixels générés: {metrics_original['n_superpixels']}")
    print(f"  Compacité: {metrics_original['compactness']:.4f}")
    print(f"  Régularité: {metrics_original['regularity']:.4f}")

    # GT Evaluation ORIGINAL
    if gt_list is not None:
        br_original = []
        ue_original = []
        for gt in gt_list:
            br_original.append(boundary_recall(labels_original, gt))
            ue_original.append(under_segmentation_error(labels_original, gt))
        metrics_original['boundary_recall'] = float(np.mean(br_original))
        metrics_original['underseg_error'] = float(np.mean(ue_original))
        print(f"  Boundary Recall (GT): {metrics_original['boundary_recall']:.4f}")
        print(f"  Undersegmentation Error (GT): {metrics_original['underseg_error']:.4f}")
    
    # =========================================================================
    # SLIC IPOL
    # =========================================================================
    print("\n" + "-"*70)
    print("2. SLIC IPOL (Amélioré)")
    print("-"*70)
    
    slic_ipol = SLIC_IPOL(
        n_segments=n_segments,
        compactness=compactness,
        max_iter=10,
    )
    
    start_time = time.time()
    labels_ipol = slic_ipol.fit(image)
    time_ipol = time.time() - start_time
    
    metrics_ipol = compute_all_metrics(labels_ipol)
    
    print(f"\nRésultats SLIC IPOL:")
    print(f"  Temps d'exécution: {time_ipol:.3f}s")
    print(f"  Superpixels générés: {metrics_ipol['n_superpixels']}")
    print(f"  Compacité: {metrics_ipol['compactness']:.4f}")
    print(f"  Régularité: {metrics_ipol['regularity']:.4f}")

    # GT Evaluation IPOL
    if gt_list is not None:
        br_ipol = []
        ue_ipol = []
        for gt in gt_list:
            br_ipol.append(boundary_recall(labels_ipol, gt))
            ue_ipol.append(under_segmentation_error(labels_ipol, gt))
        metrics_ipol['boundary_recall'] = float(np.mean(br_ipol))
        metrics_ipol['underseg_error'] = float(np.mean(ue_ipol))
        print(f"  Boundary Recall (GT): {metrics_ipol['boundary_recall']:.4f}")
        print(f"  Undersegmentation Error (GT): {metrics_ipol['underseg_error']:.4f}")
    
    # =========================================================================
    # COMPARAISON
    # =========================================================================
    print("\n" + "="*70)
    print("COMPARAISON")
    print("="*70)
    
    print(f"\nDifférence de temps: {(time_ipol - time_original):.3f}s " +
          f"({(time_ipol/time_original - 1)*100:+.1f}%)")
    
    print(f"\nDifférence de superpixels: " +
          f"{metrics_ipol['n_superpixels'] - metrics_original['n_superpixels']} " +
          f"({(metrics_ipol['n_superpixels']/metrics_original['n_superpixels'] - 1)*100:+.1f}%)")
    
    print(f"\nDifférence de compacité: " +
          f"{metrics_ipol['compactness'] - metrics_original['compactness']:+.4f} " +
          f"({(metrics_ipol['compactness']/metrics_original['compactness'] - 1)*100:+.1f}%)")
    
    print(f"\nDifférence de régularité: " +
          f"{metrics_ipol['regularity'] - metrics_original['regularity']:+.4f} " +
          f"({(metrics_ipol['regularity']/metrics_original['regularity'] - 1)*100:+.1f}%)")

    # =====================================================================
    # GT COMPARISON SUMMARY
    # =====================================================================
    if gt_list is not None:
        print("\n" + "="*70)
        print("COMPARAISON AVEC GROUND TRUTH BSDS500")
        print("="*70)

        print(f"\nBoundary Recall:")
        print(f"  Original : {metrics_original['boundary_recall']:.4f}")
        print(f"  IPOL     : {metrics_ipol['boundary_recall']:.4f}")

        print(f"\nUndersegmentation Error:")
        print(f"  Original : {metrics_original['underseg_error']:.4f}")
        print(f"  IPOL     : {metrics_ipol['underseg_error']:.4f}")

    
    # =========================================================================
    # VISUALISATION
    # =========================================================================
    print("\n" + "-"*70)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("-"*70)
    
    # 2 rows × 4 columns (adding GT col)
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Image Originale', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('Image Originale', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    from skimage import segmentation
    if image.max() > 1:
        img_norm = image / 255.0
    else:
        img_norm = image
    
    marked_original = segmentation.mark_boundaries(
        img_norm, labels_original, 
        color=(1, 1, 0), mode='thick'
    )
    axes[0, 1].imshow(marked_original)
    axes[0, 1].set_title(
        f'SLIC Original\n({metrics_original["n_superpixels"]} SP, {time_original:.2f}s)',
        fontsize=12, fontweight='bold'
    )
    axes[0, 1].axis('off')
    
    marked_ipol = segmentation.mark_boundaries(
        img_norm, labels_ipol,
        color=(1, 1, 0), mode='thick'
    )
    axes[1, 1].imshow(marked_ipol)
    axes[1, 1].set_title(
        f'SLIC IPOL\n({metrics_ipol["n_superpixels"]} SP, {time_ipol:.2f}s)',
        fontsize=12, fontweight='bold', color='green'
    )
    axes[1, 1].axis('off')
    
    from skimage import color as skcolor
    colored_original = skcolor.label2rgb(labels_original, img_norm, kind='avg')
    axes[0, 2].imshow(colored_original)
    axes[0, 2].set_title(
        f'Couleurs Moyennes\nCompacité: {metrics_original["compactness"]:.3f}',
        fontsize=11
    )
    axes[0, 2].axis('off')
    
    colored_ipol = skcolor.label2rgb(labels_ipol, img_norm, kind='avg')
    axes[1, 2].imshow(colored_ipol)
    axes[1, 2].set_title(
        f'Couleurs Moyennes\nCompacité: {metrics_ipol["compactness"]:.3f}',
        fontsize=11, color='green'
    )
    axes[1, 2].axis('off')

    # =====================================================================
    # Ground Truth Visualisation (new)
    # =====================================================================
    if gt_list is not None:
        from skimage import segmentation

        gt0 = gt_list[0]  # first annotator only

        marked_gt = segmentation.mark_boundaries(
            img_norm, gt0, color=(1, 0, 0), mode="thick"
        )

        axes[0, 3].imshow(gt0, cmap="tab20")
        axes[0, 3].set_title(
            'GT Segmentation\nAnnotateur 1',
            fontsize=12, fontweight='bold'
        )
        axes[0, 3].axis('off')

        axes[1, 3].imshow(marked_gt)
        axes[1, 3].set_title(
            'Contours GT sur Image',
            fontsize=12, fontweight='bold'
        )
        axes[1, 3].axis('off')
    else:
        axes[0, 3].axis('off')
        axes[1, 3].axis('off')
    
    plt.suptitle(
        f'Comparaison SLIC Original vs IPOL (n={n_segments}, m={compactness})',
        fontsize=16, fontweight='bold'
    )
    plt.tight_layout()
    
    results_dir = Path('results/slic/comparison')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = Path(image_path).stem
    fig_path = results_dir / f'{image_name}_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure sauvegardée: {fig_path}")
    
    plt.show()
    
    # =========================================================================
    # METRICS BAR CHART
    # =========================================================================
    fig_metrics, axes_metrics = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ['SLIC Original', 'SLIC IPOL']
    compactness_vals = [metrics_original['compactness'], metrics_ipol['compactness']]
    regularity_vals = [metrics_original['regularity'], metrics_ipol['regularity']]
    n_sp_vals = [metrics_original['n_superpixels'], metrics_ipol['n_superpixels']]
    
    bars1 = axes_metrics[0].bar(methods, compactness_vals, 
                                color=['steelblue', 'limegreen'], alpha=0.7)
    axes_metrics[0].set_ylabel('Score')
    axes_metrics[0].set_title('Compacité', fontweight='bold')
    axes_metrics[0].set_ylim([0, 1])
    axes_metrics[0].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars1, compactness_vals):
        height = bar.get_height()
        axes_metrics[0].text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    bars2 = axes_metrics[1].bar(methods, regularity_vals,
                                color=['steelblue', 'limegreen'], alpha=0.7)
    axes_metrics[1].set_ylabel('Score')
    axes_metrics[1].set_title('Régularité', fontweight='bold')
    axes_metrics[1].set_ylim([0, 1])
    axes_metrics[1].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars2, regularity_vals):
        height = bar.get_height()
        axes_metrics[1].text(bar.get_x() + bar.get_width()/2., height,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    bars3 = axes_metrics[2].bar(methods, n_sp_vals,
                                color=['steelblue', 'limegreen'], alpha=0.7)
    axes_metrics[2].set_ylabel('Nombre')
    axes_metrics[2].set_title('Nombre de Superpixels', fontweight='bold')
    axes_metrics[2].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars3, n_sp_vals):
        height = bar.get_height()
        axes_metrics[2].text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(val)}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Comparaison des Métriques', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    metrics_path = results_dir / f'{image_name}_metrics.png'
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    print(f"Métriques sauvegardées: {metrics_path}")
    
    plt.show()
    
    print("\n" + "="*70)
    print("COMPARAISON TERMINÉE")
    print("="*70)
    
    return {
        'original': {'labels': labels_original, 'metrics': metrics_original, 'time': time_original},
        'ipol': {'labels': labels_ipol, 'metrics': metrics_ipol, 'time': time_ipol},
        'groundtruth_count': len(gt_list) if gt_list is not None else 0
    }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare SLIC Original vs SLIC IPOL'
    )
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image')
    parser.add_argument('--g', type=str, default=None,
                       help='Path to directory containing ground truth (BSDS500)')
    parser.add_argument('--n_segments', type=int, default=200,
                       help='Number of superpixels')
    parser.add_argument('--compactness', type=float, default=10.0,
                       help='Compactness parameter')
    
    args = parser.parse_args()

    compare_versions(args.image, args.n_segments, args.compactness, gt_dir=args.g)


if __name__ == "__main__":
    main()

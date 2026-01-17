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
from src.evaluation.metrics import compute_all_metrics, compute_metrics_multiple_gt
from src.evaluation.visualize import visualize_segmentation


def compare_versions(image_path, n_segments=200, compactness=10, gt_dir=None):
    """
    Compare SLIC original vs SLIC IPOL.
    
    Args:
        image_path: Path to image
        n_segments: Number of superpixels
        compactness: Compactness parameter
        gt_dir: Path to ground truth directory
    """
    print("="*80)
    print("COMPARAISON SLIC ORIGINAL vs SLIC IPOL")
    print("="*80)
    
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
    print("\n" + "-"*80)
    print("1. SLIC ORIGINAL")
    print("-"*80)
    
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
    print(f"\nMétriques internes:")
    print(f"  Superpixels générés: {metrics_original['n_superpixels']}")
    print(f"  Compacité: {metrics_original['compactness']:.4f}")
    print(f"  Régularité: {metrics_original['regularity']:.4f}")

    # GT Evaluation ORIGINAL
    gt_metrics_original = None
    if gt_list is not None:
        print(f"\nMétriques avec Ground Truth ({len(gt_list)} annotateurs):")
        gt_metrics_original = compute_metrics_multiple_gt(labels_original, gt_list)
        
        print(f"  Boundary Recall: {gt_metrics_original['boundary_recall']:.4f} ± {gt_metrics_original['boundary_recall_std']:.4f}")
        print(f"  Under-Segmentation Error: {gt_metrics_original['under_segmentation_error']:.4f} ± {gt_metrics_original['under_segmentation_error_std']:.4f}")
        print(f"  Corrected Under-Seg. Error: {gt_metrics_original['corrected_under_segmentation_error']:.4f} ± {gt_metrics_original['corrected_under_segmentation_error_std']:.4f}")
        print(f"  ASA: {gt_metrics_original['asa']:.4f} ± {gt_metrics_original['asa_std']:.4f}")
        print(f"  Precision: {gt_metrics_original['precision']:.4f} ± {gt_metrics_original['precision_std']:.4f}")
        print(f"  Contour Density: {gt_metrics_original['contour_density']:.4f} ± {gt_metrics_original['contour_density_std']:.4f}")
        print(f"  Explained Variation: {gt_metrics_original['explained_variation']:.4f} ± {gt_metrics_original['explained_variation_std']:.4f}")
        print(f"  Global Regularity: {gt_metrics_original['global_regularity']:.4f}")
        
        metrics_original.update(gt_metrics_original)
    
    # =========================================================================
    # SLIC IPOL
    # =========================================================================
    print("\n" + "-"*80)
    print("2. SLIC IPOL (Amélioré)")
    print("-"*80)
    
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
    print(f"\nMétriques internes:")
    print(f"  Superpixels générés: {metrics_ipol['n_superpixels']}")
    print(f"  Compacité: {metrics_ipol['compactness']:.4f}")
    print(f"  Régularité: {metrics_ipol['regularity']:.4f}")

    # GT Evaluation IPOL
    gt_metrics_ipol = None
    if gt_list is not None:
        print(f"\nMétriques avec Ground Truth ({len(gt_list)} annotateurs):")
        gt_metrics_ipol = compute_metrics_multiple_gt(labels_ipol, gt_list)
        
        print(f"  Boundary Recall: {gt_metrics_ipol['boundary_recall']:.4f} ± {gt_metrics_ipol['boundary_recall_std']:.4f}")
        print(f"  Under-Segmentation Error: {gt_metrics_ipol['under_segmentation_error']:.4f} ± {gt_metrics_ipol['under_segmentation_error_std']:.4f}")
        print(f"  Corrected Under-Seg. Error: {gt_metrics_ipol['corrected_under_segmentation_error']:.4f} ± {gt_metrics_ipol['corrected_under_segmentation_error_std']:.4f}")
        print(f"  ASA: {gt_metrics_ipol['asa']:.4f} ± {gt_metrics_ipol['asa_std']:.4f}")
        print(f"  Precision: {gt_metrics_ipol['precision']:.4f} ± {gt_metrics_ipol['precision_std']:.4f}")
        print(f"  Contour Density: {gt_metrics_ipol['contour_density']:.4f} ± {gt_metrics_ipol['contour_density_std']:.4f}")
        print(f"  Explained Variation: {gt_metrics_ipol['explained_variation']:.4f} ± {gt_metrics_ipol['explained_variation_std']:.4f}")
        print(f"  Global Regularity: {gt_metrics_ipol['global_regularity']:.4f}")
        
        metrics_ipol.update(gt_metrics_ipol)
    
    # =========================================================================
    # COMPARAISON DÉTAILLÉE
    # =========================================================================
    print("\n" + "="*80)
    print("COMPARAISON DÉTAILLÉE")
    print("="*80)
    
    print(f"\n{'Métrique':<35} {'Original':<12} {'IPOL':<12} {'Diff':<12} {'%':<8}")
    print("-"*80)
    
    # Métriques de base
    print(f"{'Temps (s)':<35} {time_original:<12.3f} {time_ipol:<12.3f} {time_ipol-time_original:<+12.3f} {(time_ipol/time_original-1)*100:<+8.1f}")
    print(f"{'Nombre de superpixels':<35} {metrics_original['n_superpixels']:<12d} {metrics_ipol['n_superpixels']:<12d} {metrics_ipol['n_superpixels']-metrics_original['n_superpixels']:<+12d} {(metrics_ipol['n_superpixels']/metrics_original['n_superpixels']-1)*100:<+8.1f}")
    print(f"{'Compacité':<35} {metrics_original['compactness']:<12.4f} {metrics_ipol['compactness']:<12.4f} {metrics_ipol['compactness']-metrics_original['compactness']:<+12.4f} {(metrics_ipol['compactness']/metrics_original['compactness']-1)*100:<+8.1f}")
    print(f"{'Régularité':<35} {metrics_original['regularity']:<12.4f} {metrics_ipol['regularity']:<12.4f} {metrics_ipol['regularity']-metrics_original['regularity']:<+12.4f} {(metrics_ipol['regularity']/metrics_original['regularity']-1)*100:<+8.1f}")
    
    # Métriques GT
    if gt_list is not None:
        print("\n" + "-"*80)
        print("MÉTRIQUES AVEC GROUND TRUTH")
        print("-"*80)
        
        gt_metrics = [
            ('Boundary Recall ↑', 'boundary_recall'),
            ('Under-Seg. Error ↓', 'under_segmentation_error'),
            ('Corrected Under-Seg. Error ↓', 'corrected_under_segmentation_error'),
            ('ASA ↑', 'asa'),
            ('Precision ↑', 'precision'),
            ('Contour Density', 'contour_density'),
            ('Explained Variation ↑', 'explained_variation'),
            ('Global Regularity ↑', 'global_regularity'),
        ]
        
        for name, key in gt_metrics:
            if key in metrics_original:
                val_orig = metrics_original[key]
                val_ipol = metrics_ipol[key]
                diff = val_ipol - val_orig
                pct = (val_ipol/val_orig - 1) * 100 if val_orig != 0 else 0
                print(f"{name:<35} {val_orig:<12.4f} {val_ipol:<12.4f} {diff:<+12.4f} {pct:<+8.1f}")

    # =========================================================================
    # VISUALISATION PRINCIPALE
    # =========================================================================
    print("\n" + "-"*80)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("-"*80)
    
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

    # Ground Truth Visualisation
    if gt_list is not None:
        gt0 = gt_list[0]
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
    # GRAPHIQUES DE MÉTRIQUES COMPLÈTES
    # =========================================================================
    if gt_list is not None:
        # Figure avec toutes les métriques GT
        fig_metrics, axes_metrics = plt.subplots(3, 3, figsize=(18, 15))
        axes_metrics = axes_metrics.flatten()
    else:
        # Figure avec métriques internes seulement
        fig_metrics, axes_metrics = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ['SLIC Original', 'SLIC IPOL']
    colors = ['steelblue', 'limegreen']
    
    # Métriques internes
    metrics_to_plot = [
        ('Compacité', 'compactness', 0, 1, True),
        ('Régularité', 'regularity', 0, 1, True),
        ('Nombre de Superpixels', 'n_superpixels', None, None, False),
    ]
    
    for idx, (title, key, ymin, ymax, is_float) in enumerate(metrics_to_plot):
        vals = [metrics_original[key], metrics_ipol[key]]
        bars = axes_metrics[idx].bar(methods, vals, color=colors, alpha=0.7)
        axes_metrics[idx].set_ylabel('Score' if is_float else 'Nombre')
        axes_metrics[idx].set_title(title, fontweight='bold')
        if ymin is not None and ymax is not None:
            axes_metrics[idx].set_ylim([ymin, ymax])
        axes_metrics[idx].grid(axis='y', alpha=0.3)
        
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            fmt = f'{val:.3f}' if is_float else f'{int(val)}'
            axes_metrics[idx].text(bar.get_x() + bar.get_width()/2., height,
                                fmt, ha='center', va='bottom', fontsize=10)
    
    # Métriques GT
    if gt_list is not None:
        gt_metrics_to_plot = [
            ('Boundary Recall', 'boundary_recall', 0, 1),
            ('Under-Seg. Error', 'under_segmentation_error', 0, None),
            ('Corrected Under-Seg. Error', 'corrected_under_segmentation_error', 0, None),
            ('ASA', 'asa', 0, 1),
            ('Precision', 'precision', 0, 1),
            ('Contour Density', 'contour_density', 0, None),
        ]
        
        for idx, (title, key, ymin, ymax) in enumerate(gt_metrics_to_plot, start=3):
            if key in metrics_original:
                vals = [metrics_original[key], metrics_ipol[key]]
                stds = [metrics_original.get(f'{key}_std', 0), metrics_ipol.get(f'{key}_std', 0)]
                
                bars = axes_metrics[idx].bar(methods, vals, yerr=stds, 
                                           color=colors, alpha=0.7, capsize=5)
                axes_metrics[idx].set_ylabel('Score')
                axes_metrics[idx].set_title(f'{title} (GT)', fontweight='bold')
                if ymin is not None:
                    if ymax is not None:
                        axes_metrics[idx].set_ylim([ymin, ymax])
                    else:
                        axes_metrics[idx].set_ylim(bottom=ymin)
                axes_metrics[idx].grid(axis='y', alpha=0.3)
                
                for bar, val in zip(bars, vals):
                    height = bar.get_height()
                    axes_metrics[idx].text(bar.get_x() + bar.get_width()/2., height,
                                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Désactiver les axes non utilisés
        for idx in range(9, len(axes_metrics)):
            axes_metrics[idx].axis('off')
    
    plt.suptitle('Comparaison Complète des Métriques', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    metrics_path = results_dir / f'{image_name}_metrics_complete.png'
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    print(f"Métriques complètes sauvegardées: {metrics_path}")
    
    plt.show()
    
    print("\n" + "="*80)
    print("COMPARAISON TERMINÉE")
    print("="*80)
    
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
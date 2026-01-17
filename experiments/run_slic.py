"""
Script d'exécution de la méthode SLIC
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.methods.slic.slic_original import SLIC
from src.preprocessing.image_loader import ImageLoader
from src.evaluation.metrics import compute_all_metrics, format_metrics
from src.evaluation.visualize import visualize_segmentation
from src.methods.slic.utils import compute_superpixel_statistics


def run_slic_single_image(image_path, n_segments=200, compactness=10, 
                         max_iter=10, save_results=True):
    """
    Exécute SLIC sur une seule image.
    
    Args:
        image_path: Chemin vers l'image
        n_segments: Nombre de superpixels
        compactness: Paramètre de compacité
        max_iter: Nombre d'itérations
        save_results: Si True, sauvegarde les résultats
    """
    print("=" * 60)
    print("EXÉCUTION SLIC")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Paramètres: n_segments={n_segments}, compactness={compactness}")
    print()
    
    # Charger l'image
    loader = ImageLoader()
    image = loader.load_image(image_path)
    print(f"Taille de l'image: {image.shape}")
    
    # Appliquer SLIC
    print("Application de SLIC...")
    slic = SLIC(n_segments=n_segments, compactness=compactness, max_iter=max_iter)
    labels = slic.fit(image)
    
    # Statistiques
    stats = compute_superpixel_statistics(image, labels)
    #print(f"\nNombre réel de superpixels: {int(stats['n_superpixels'])}") TODO
    #print(f"Taille moyenne: {stats['mean_size']:.1f} ± {stats['std_size']:.1f} pixels") TODO
    #print(f"Taille min/max: {int(stats['min_size'])}/{int(stats['max_size'])} pixels") TODO
    
    # Métriques
    metrics = compute_all_metrics(labels)
    print(f"\n{format_metrics(metrics)}")
    
    # Visualisation
    fig = visualize_segmentation(image, labels, 
                             title=f"SLIC (n={n_segments}, m={compactness})",
                             show_boundaries=True, boundary_color=(1, 1, 0))
    
    # Sauvegarder
    if save_results:
        results_dir = Path('results/slic')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder la figure
        image_name = Path(image_path).stem
        fig_path = results_dir / f"{image_name}_seg.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure sauvegardée: {fig_path}")
        
        # Sauvegarder les labels
        labels_path = results_dir / f"{image_name}_labels.npy"
        np.save(labels_path, labels)
        print(f"Labels sauvegardés: {labels_path}")
    
    plt.show()
    
    return labels, metrics


def run_slic_batch(image_dir, n_segments=200, compactness=10, max_iter=10):
    """
    Exécute SLIC sur un dossier d'images.
    
    Args:
        image_dir: Dossier contenant les images
        n_segments: Nombre de superpixels
        compactness: Paramètre de compacité
        max_iter: Nombre d'itérations
    """
    from glob import glob
    
    image_paths = glob(os.path.join(image_dir, '*.jpg')) + \
                  glob(os.path.join(image_dir, '*.png'))
    
    print(f"Trouvé {len(image_paths)} images")
    
    results = []
    
    for img_path in image_paths:
        print(f"\nTraitement: {img_path}")
        labels, metrics = run_slic_single_image(
            img_path, n_segments, compactness, max_iter, save_results=True
        )
        results.append({
            'image': img_path,
            'labels': labels,
            'metrics': metrics
        })
    
    return results


def main():
    """
    Point d'entrée principal.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Exécution SLIC')
    parser.add_argument('--image', type=str, help='Chemin vers une image')
    parser.add_argument('--image_dir', type=str, help='Dossier d\'images')
    parser.add_argument('--n_segments', type=int, default=200, 
                       help='Nombre de superpixels')
    parser.add_argument('--compactness', type=float, default=10.0, 
                       help='Paramètre de compacité')
    parser.add_argument('--max_iter', type=int, default=10, 
                       help='Nombre d\'itérations')
    
    args = parser.parse_args()
    
    if args.image:
        run_slic_single_image(args.image, args.n_segments, 
                             args.compactness, args.max_iter)
    elif args.image_dir:
        run_slic_batch(args.image_dir, args.n_segments, 
                      args.compactness, args.max_iter)
    else:
        print("Veuillez spécifier --image ou --image_dir")
        print("\nExemple d'utilisation:")
        print("python run_slic.py --image path/to/image.jpg --n_segments 300 --compactness 15")


if __name__ == "__main__":
    main()
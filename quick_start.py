"""
Script de démarrage rapide pour tester SLIC
Créez simplement une image test ou utilisez une de vos images
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from src.methods.slic.slic_original import SLIC
from src.evaluation.metrics import compute_all_metrics, format_metrics
from src.evaluation.visualize import visualize_segmentation


def create_test_image():
    """Crée une image test synthétique avec des régions colorées"""
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # Fond bleu
    image[:, :] = [100, 150, 200]
    
    # Cercle rouge
    y, x = np.ogrid[:300, :400]
    mask_circle = (x - 150)**2 + (y - 150)**2 <= 50**2
    image[mask_circle] = [220, 50, 50]
    
    # Rectangle vert
    image[50:100, 250:350] = [50, 200, 50]
    
    # Rectangle jaune
    image[200:250, 250:350] = [230, 230, 50]
    
    # Ajouter du bruit
    noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


def example_1_basic_usage():
    """Exemple 1: Utilisation basique de SLIC"""
    print("\n" + "="*70)
    print("EXEMPLE 1: UTILISATION BASIQUE")
    print("="*70)
    
    # Créer une image test
    image = create_test_image()
    
    # Appliquer SLIC
    print("\nApplication de SLIC avec paramètres par défaut...")
    slic = SLIC(n_segments=100, compactness=10, max_iter=10)
    labels = slic.fit(image)
    
    print(f"Nombre de superpixels générés: {len(np.unique(labels))}")
    
    # Visualiser
    fig = visualize_segmentation(image, labels, 
                                 title="SLIC - Utilisation basique")
    plt.savefig('results/example_basic.png', dpi=150, bbox_inches='tight')
    print("Figure sauvegardée: results/example_basic.png")
    plt.show()


def example_2_parameter_comparison():
    """Exemple 2: Comparaison de différents paramètres"""
    print("\n" + "="*70)
    print("EXEMPLE 2: COMPARAISON DE PARAMÈTRES")
    print("="*70)
    
    image = create_test_image()
    
    # Tester différentes configurations
    configs = [
        (50, 10, "Peu de superpixels (50)"),
        (200, 10, "Beaucoup de superpixels (200)"),
        (100, 1, "Faible compacité (m=1)"),
        (100, 40, "Forte compacité (m=40)")
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (n_seg, comp, title) in enumerate(configs):
        print(f"\nTest: {title}...")
        slic = SLIC(n_segments=n_seg, compactness=comp, max_iter=10)
        labels = slic.fit(image)
        
        from skimage import segmentation
        if image.max() > 1:
            img_norm = image / 255.0
        else:
            img_norm = image
        marked = segmentation.mark_boundaries(img_norm, labels, 
                                             color=(1, 1, 0), mode='thick')
        
        axes[idx].imshow(marked)
        axes[idx].set_title(f'{title}\n({len(np.unique(labels))} superpixels)')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('results/example_comparison.png', dpi=150, bbox_inches='tight')
    print("\nFigure sauvegardée: results/example_comparison.png")
    plt.show()


def example_3_metrics_evaluation():
    """Exemple 3: Évaluation avec métriques"""
    print("\n" + "="*70)
    print("EXEMPLE 3: ÉVALUATION AVEC MÉTRIQUES")
    print("="*70)
    
    image = create_test_image()
    
    # Tester plusieurs configurations et comparer les métriques
    configs = [
        {'n_segments': 50, 'compactness': 10, 'name': 'Config A'},
        {'n_segments': 100, 'compactness': 10, 'name': 'Config B'},
        {'n_segments': 200, 'compactness': 10, 'name': 'Config C'},
        {'n_segments': 100, 'compactness': 20, 'name': 'Config D'},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{config['name']}: n_segments={config['n_segments']}, "
              f"compactness={config['compactness']}")
        
        slic = SLIC(n_segments=config['n_segments'], 
                   compactness=config['compactness'], 
                   max_iter=10)
        labels = slic.fit(image)
        
        metrics = compute_all_metrics(labels)
        metrics['config'] = config['name']
        results.append(metrics)
        
        print(f"  Superpixels: {metrics['n_superpixels']}")
        print(f"  Compacité: {metrics['compactness']:.4f}")
        print(f"  Régularité: {metrics['regularity']:.4f}")
    
    # Visualiser les métriques
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = [r['config'] for r in results]
    compactness_scores = [r['compactness'] for r in results]
    regularity_scores = [r['regularity'] for r in results]
    n_superpixels = [r['n_superpixels'] for r in results]
    
    axes[0].bar(names, compactness_scores, color='steelblue', alpha=0.7)
    axes[0].set_title('Compacité')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(names, regularity_scores, color='forestgreen', alpha=0.7)
    axes[1].set_title('Régularité')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    
    axes[2].bar(names, n_superpixels, color='coral', alpha=0.7)
    axes[2].set_title('Nombre de superpixels')
    axes[2].set_ylabel('Nombre')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/example_metrics.png', dpi=150, bbox_inches='tight')
    print("\nFigure sauvegardée: results/example_metrics.png")
    plt.show()


def example_4_real_image():
    """Exemple 4: Application sur une vraie image (si disponible)"""
    print("\n" + "="*70)
    print("EXEMPLE 4: APPLICATION SUR UNE VRAIE IMAGE")
    print("="*70)
    
    try:
        from src.preprocessing.image_loader import ImageLoader
        
        loader = ImageLoader(data_dir='data')
        images, paths = loader.load_bsds500_images(split='test', max_images=1)
        
        if images:
            image = images[0]
            print(f"Image chargée: {paths[0]}")
            print(f"Taille: {image.shape}")
            
            # Appliquer SLIC
            slic = SLIC(n_segments=200, compactness=10, max_iter=10)
            labels = slic.fit(image)
            
            # Visualiser
            fig = visualize_segmentation(image, labels,
                                        title="SLIC sur image réelle (BSDS500)")
            plt.savefig('results/example_real_image.png', dpi=150, bbox_inches='tight')
            print("Figure sauvegardée: results/example_real_image.png")
            plt.show()
            
            # Métriques
            metrics = compute_all_metrics(labels)
            print(format_metrics(metrics))
        else:
            print("Aucune image trouvée dans le dataset BSDS500")
            print("Veuillez télécharger le dataset et le placer dans data/BSDS500/")
    
    except Exception as e:
        print(f"Impossible de charger une image réelle: {e}")
        print("Utilisation d'une image synthétique à la place")
        example_1_basic_usage()


def main():
    """Fonction principale"""
    # Créer le dossier de résultats
    import os
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "#"*70)
    print("# DÉMONSTRATION DE LA MÉTHODE SLIC")
    print("#"*70)
    
    # Exécuter les exemples
    example_1_basic_usage()
    example_2_parameter_comparison()
    example_3_metrics_evaluation()
    example_4_real_image()
    
    print("\n" + "#"*70)
    print("# DÉMONSTRATION TERMINÉE")
    print("#"*70)
    print("\nTous les résultats ont été sauvegardés dans le dossier 'results/'")
    print("\nPour aller plus loin:")
    print("  - Consultez les notebooks dans notebooks/")
    print("  - Lancez les scripts d'expérimentation dans experiments/")
    print("  - Lisez le README.md pour plus d'informations")


if __name__ == "__main__":
    main()
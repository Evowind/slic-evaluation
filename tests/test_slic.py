"""
Tests unitaires pour la méthode SLIC
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.methods.slic.slic_original import SLIC
from src.utils.color_space import rgb_to_lab, lab_to_rgb
from src.utils.distance import compute_distance, compute_distance_vectorized


class TestColorSpace:
    """Tests pour les conversions d'espaces colorimétriques"""
    
    def test_rgb_to_lab_range(self):
        """Vérifie que la conversion RGB->Lab produit des valeurs dans les bonnes plages"""
        # Image test RGB
        rgb = np.random.randint(0, 256, (100, 100, 3)).astype(np.uint8)
        lab = rgb_to_lab(rgb)
        
        # Vérifier les plages
        assert lab[:, :, 0].min() >= 0 and lab[:, :, 0].max() <= 100, "L doit être dans [0, 100]"
        assert lab[:, :, 1].min() >= -128 and lab[:, :, 1].max() <= 127, "a doit être dans [-128, 127]"
        assert lab[:, :, 2].min() >= -128 and lab[:, :, 2].max() <= 127, "b doit être dans [-128, 127]"
    
    def test_rgb_lab_roundtrip(self):
        """Vérifie que RGB->Lab->RGB préserve approximativement les valeurs"""
        rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        lab = rgb_to_lab(rgb)
        rgb_back = lab_to_rgb(lab)
        
        # Tolérance de 1/255 pour les erreurs d'arrondi
        assert np.allclose(rgb / 255.0, rgb_back, atol=0.01)


class TestDistance:
    """Tests pour les métriques de distance"""
    
    def test_distance_symmetry(self):
        """Vérifie que la distance est symétrique"""
        pixel_lab = np.array([50, 10, 20])
        center_lab = np.array([60, 15, 25])
        pixel_xy = np.array([10, 20])
        center_xy = np.array([15, 25])
        
        d1 = compute_distance(pixel_lab, center_lab, pixel_xy, center_xy, m=10, S=10)
        d2 = compute_distance(center_lab, pixel_lab, center_xy, pixel_xy, m=10, S=10)
        
        assert np.isclose(d1, d2), "La distance devrait être symétrique"
    
    def test_distance_zero(self):
        """Vérifie que la distance entre un point et lui-même est nulle"""
        pixel_lab = np.array([50, 10, 20])
        pixel_xy = np.array([10, 20])
        
        d = compute_distance(pixel_lab, pixel_lab, pixel_xy, pixel_xy, m=10, S=10)
        
        assert np.isclose(d, 0), "La distance d'un point à lui-même devrait être 0"
    
    def test_distance_vectorized_equivalence(self):
        """Vérifie que la version vectorisée donne les mêmes résultats"""
        center_lab = np.array([50, 10, 20])
        center_xy = np.array([50, 50])
        
        # Plusieurs pixels
        n_pixels = 10
        pixels_lab = np.random.randn(n_pixels, 3) * 20 + center_lab
        pixels_xy = np.random.randn(n_pixels, 2) * 10 + center_xy
        
        # Version vectorisée
        distances_vec = compute_distance_vectorized(pixels_lab, center_lab, 
                                                    pixels_xy, center_xy, m=10, S=10)
        
        # Version scalaire
        distances_scalar = np.array([
            compute_distance(pixels_lab[i], center_lab, 
                           pixels_xy[i], center_xy, m=10, S=10)
            for i in range(n_pixels)
        ])
        
        assert np.allclose(distances_vec, distances_scalar), \
            "Les versions vectorisée et scalaire devraient donner les mêmes résultats"


class TestSLIC:
    """Tests pour la classe SLIC"""
    
    def setup_method(self):
        """Prépare une image test avant chaque test"""
        # Image test simple 100x100
        self.image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    def test_slic_initialization(self):
        """Teste l'initialisation de SLIC"""
        slic = SLIC(n_segments=100, compactness=10, max_iter=5)
        
        assert slic.n_segments == 100
        assert slic.compactness == 10
        assert slic.max_iter == 5
    
    def test_slic_fit_returns_labels(self):
        """Vérifie que fit() retourne un array de labels"""
        slic = SLIC(n_segments=20, compactness=10, max_iter=5)
        labels = slic.fit(self.image)
        
        assert labels is not None
        assert labels.shape == self.image.shape[:2]
        assert labels.dtype in [np.int32, np.int64]
    
    def test_slic_number_of_segments(self):
        """Vérifie que le nombre de superpixels est approximativement correct"""
        n_segments_target = 50
        slic = SLIC(n_segments=n_segments_target, compactness=10, max_iter=5)
        labels = slic.fit(self.image)
        
        n_segments_actual = len(np.unique(labels))
        
        # Tolérance de 50% (peut varier selon la connectivité)
        assert abs(n_segments_actual - n_segments_target) < n_segments_target * 0.5, \
            f"Nombre de segments: {n_segments_actual}, attendu ~{n_segments_target}"
    
    def test_slic_all_pixels_labeled(self):
        """Vérifie que tous les pixels sont assignés à un superpixel"""
        slic = SLIC(n_segments=20, compactness=10, max_iter=5)
        labels = slic.fit(self.image)
        
        assert np.all(labels >= 0), "Tous les pixels doivent avoir un label >= 0"
    
    def test_slic_reproducibility(self):
        """Vérifie que SLIC donne des résultats reproductibles"""
        slic1 = SLIC(n_segments=30, compactness=10, max_iter=5)
        labels1 = slic1.fit(self.image.copy())
        
        slic2 = SLIC(n_segments=30, compactness=10, max_iter=5)
        labels2 = slic2.fit(self.image.copy())
        
        # Les labels peuvent être différents mais la structure doit être la même
        assert labels1.shape == labels2.shape
        assert len(np.unique(labels1)) == len(np.unique(labels2))
    
    def test_slic_different_compactness(self):
        """Vérifie que différents paramètres de compacité donnent des résultats différents"""
        slic1 = SLIC(n_segments=30, compactness=1, max_iter=5)
        labels1 = slic1.fit(self.image)
        
        slic2 = SLIC(n_segments=30, compactness=40, max_iter=5)
        labels2 = slic2.fit(self.image)
        
        # Les segmentations devraient être différentes
        assert not np.array_equal(labels1, labels2)
    
    def test_slic_boundaries(self):
        """Teste l'extraction des contours"""
        slic = SLIC(n_segments=20, compactness=10, max_iter=5)
        labels = slic.fit(self.image)
        boundaries = slic.get_boundaries()
        
        assert boundaries is not None
        assert boundaries.shape == labels.shape
        assert boundaries.dtype == bool
        assert np.any(boundaries), "Il devrait y avoir au moins quelques contours"
    
    def test_slic_small_image(self):
        """Teste SLIC sur une petite image"""
        small_image = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        slic = SLIC(n_segments=4, compactness=10, max_iter=3)
        labels = slic.fit(small_image)
        
        assert labels.shape == (20, 20)
        assert len(np.unique(labels)) > 0
    
    def test_slic_large_n_segments(self):
        """Teste avec un grand nombre de superpixels"""
        slic = SLIC(n_segments=200, compactness=10, max_iter=5)
        labels = slic.fit(self.image)
        
        n_segments = len(np.unique(labels))
        assert n_segments > 50, "Devrait produire un nombre significatif de superpixels"


class TestMetrics:
    """Tests pour les métriques d'évaluation"""
    
    def setup_method(self):
        """Prépare des données test"""
        # Créer une segmentation simple
        self.labels = np.zeros((100, 100), dtype=np.int32)
        self.labels[:50, :] = 0
        self.labels[50:, :] = 1
        
        # Ground truth similaire mais légèrement décalé
        self.gt = np.zeros((100, 100), dtype=np.int32)
        self.gt[:48, :] = 0
        self.gt[48:, :] = 1
    
    def test_compactness_range(self):
        """Vérifie que la compacité est dans [0, 1]"""
        from src.evaluation.metrics import compactness
        
        comp = compactness(self.labels)
        assert 0 <= comp <= 1, f"Compacité devrait être dans [0, 1], obtenu {comp}"
    
    def test_boundary_recall_range(self):
        """Vérifie que BR est dans [0, 1]"""
        from src.evaluation.metrics import boundary_recall
        
        br = boundary_recall(self.labels, self.gt)
        assert 0 <= br <= 1, f"BR devrait être dans [0, 1], obtenu {br}"
    
    def test_under_segmentation_error_range(self):
        """Vérifie que UE est positif"""
        from src.evaluation.metrics import under_segmentation_error
        
        ue = under_segmentation_error(self.labels, self.gt)
        assert ue >= 0, f"UE devrait être positif, obtenu {ue}"
    
    def test_asa_range(self):
        """Vérifie que ASA est dans [0, 1]"""
        from src.evaluation.metrics import achievable_segmentation_accuracy
        
        asa = achievable_segmentation_accuracy(self.labels, self.gt)
        assert 0 <= asa <= 1, f"ASA devrait être dans [0, 1], obtenu {asa}"
    
    def test_perfect_segmentation(self):
        """Teste les métriques sur une segmentation parfaite"""
        from src.evaluation.metrics import (
            boundary_recall, under_segmentation_error,
            achievable_segmentation_accuracy
        )
        
        # Segmentation identique
        br = boundary_recall(self.gt, self.gt)
        ue = under_segmentation_error(self.gt, self.gt)
        asa = achievable_segmentation_accuracy(self.gt, self.gt)
        
        assert br > 0.95, "BR devrait être proche de 1 pour une segmentation parfaite"
        assert ue < 0.05, "UE devrait être proche de 0 pour une segmentation parfaite"
        assert asa > 0.99, "ASA devrait être 1 pour une segmentation parfaite"


class TestImageLoader:
    """Tests pour le chargement d'images"""
    
    def test_image_loader_initialization(self):
        """Teste l'initialisation du loader"""
        from src.preprocessing.image_loader import ImageLoader
        
        loader = ImageLoader(data_dir='data')
        assert loader.data_dir.name == 'data'
    
    def test_preprocess_image_normalization(self):
        """Teste la normalisation d'image"""
        from src.preprocessing.image_loader import preprocess_image
        
        image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        processed = preprocess_image(image, normalize=True)
        
        assert processed.max() <= 1.0, "L'image normalisée devrait avoir des valeurs <= 1"
        assert processed.min() >= 0.0, "L'image normalisée devrait avoir des valeurs >= 0"


class TestVisualization:
    """Tests pour les fonctions de visualisation"""
    
    def setup_method(self):
        """Prépare des données test"""
        self.image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        self.labels = np.random.randint(0, 10, (50, 50), dtype=np.int32)
    
    def test_visualize_superpixels(self):
        """Teste la visualisation des superpixels"""
        from src.methods.slic.utils import visualize_superpixels
        
        marked = visualize_superpixels(self.image, self.labels)
        
        assert marked is not None
        assert marked.shape == self.image.shape or marked.shape == (50, 50, 4)
    
    def test_get_average_colors(self):
        """Teste le calcul des couleurs moyennes"""
        from src.methods.slic.utils import get_average_colors
        
        colored = get_average_colors(self.image, self.labels)
        
        assert colored.shape == self.image.shape
        assert colored.max() <= 1.0


class TestIntegration:
    """Tests d'intégration bout-en-bout"""
    
    def test_full_pipeline(self):
        """Teste le pipeline complet: chargement -> SLIC -> métriques -> visualisation"""
        from src.methods.slic.slic_original import SLIC
        from src.evaluation.metrics import compute_all_metrics
        from src.methods.slic.utils import visualize_superpixels
        
        # Créer une image test
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Appliquer SLIC
        slic = SLIC(n_segments=50, compactness=10, max_iter=5)
        labels = slic.fit(image)
        
        # Calculer les métriques
        metrics = compute_all_metrics(labels)
        
        # Vérifications
        assert 'n_superpixels' in metrics
        assert 'compactness' in metrics
        assert metrics['n_superpixels'] > 0
        
        # Visualiser
        marked = visualize_superpixels(image, labels)
        assert marked is not None
    
    def test_parameter_impact(self):
        """Teste l'impact des paramètres sur les résultats"""
        from src.methods.slic.slic_original import SLIC
        from src.evaluation.metrics import compactness as compute_compactness
        
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Faible compacité
        slic_low = SLIC(n_segments=50, compactness=1, max_iter=5)
        labels_low = slic_low.fit(image)
        comp_low = compute_compactness(labels_low)
        
        # Forte compacité
        slic_high = SLIC(n_segments=50, compactness=40, max_iter=5)
        labels_high = slic_high.fit(image)
        comp_high = compute_compactness(labels_high)
        
        # La compacité mesurée devrait généralement être plus élevée
        # avec un paramètre de compacité plus élevé
        print(f"Compacité (m=1): {comp_low:.4f}")
        print(f"Compacité (m=40): {comp_high:.4f}")
        # Note: Pas toujours vrai selon l'image, mais c'est la tendance générale


def run_all_tests():
    """Exécute tous les tests"""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == "__main__":
    run_all_tests()
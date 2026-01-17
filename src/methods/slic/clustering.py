"""
K-means adapté pour SLIC
Implémente le clustering spécifique à la méthode SLIC
"""
import numpy as np
from src.utils.distance import compute_distance_vectorized


class SLICCluster:
    """
    Représente un cluster (superpixel) dans l'algorithme SLIC.
    
    Attributes:
        center: Centre du cluster [L, a, b, x, y]
        pixels: Liste des pixels assignés au cluster
        label: Identifiant du cluster
    """
    
    def __init__(self, center, label):
        """
        Initialise un cluster.
        
        Args:
            center: Position initiale [L, a, b, x, y]
            label: Identifiant unique du cluster
        """
        self.center = np.array(center, dtype=np.float64)
        self.pixels = []
        self.label = label
    
    def add_pixel(self, pixel_coords):
        """
        Ajoute un pixel au cluster.
        
        Args:
            pixel_coords: Coordonnées du pixel [y, x]
        """
        self.pixels.append(pixel_coords)
    
    def clear_pixels(self):
        """Vide la liste des pixels."""
        self.pixels = []
    
    def update_center(self, lab_image):
        """
        Met à jour le centre du cluster comme la moyenne des pixels assignés.
        
        Args:
            lab_image: Image en espace Lab (H, W, 3)
        """
        if not self.pixels:
            return
        
        pixels_array = np.array(self.pixels)
        y_coords = pixels_array[:, 0]
        x_coords = pixels_array[:, 1]
        
        # Moyenne des valeurs Lab
        L_mean = np.mean(lab_image[y_coords, x_coords, 0])
        a_mean = np.mean(lab_image[y_coords, x_coords, 1])
        b_mean = np.mean(lab_image[y_coords, x_coords, 2])
        
        # Moyenne des positions spatiales
        x_mean = np.mean(x_coords)
        y_mean = np.mean(y_coords)
        
        self.center = np.array([L_mean, a_mean, b_mean, x_mean, y_mean])
    
    def get_size(self):
        """Retourne le nombre de pixels dans le cluster."""
        return len(self.pixels)


class KMeansSLIC:
    """
    K-means adapté pour SLIC.
    Implémente le clustering avec distance SLIC et recherche limitée.
    """
    
    def __init__(self, n_clusters, compactness, S):
        """
        Initialise le clusterer.
        
        Args:
            n_clusters: Nombre de clusters
            compactness: Paramètre m de SLIC
            S: Espacement de la grille
        """
        self.n_clusters = n_clusters
        self.compactness = compactness
        self.S = S
        self.clusters = []
    
    def initialize_clusters(self, centers):
        """
        Initialise les clusters avec les centres donnés.
        
        Args:
            centers: Array (K, 5) avec [L, a, b, x, y]
        """
        self.clusters = []
        for i, center in enumerate(centers):
            self.clusters.append(SLICCluster(center, label=i))
    
    def assign_pixels(self, lab_image, labels, distances):
        """
        Assigne chaque pixel au cluster le plus proche.
        Utilise une recherche limitée à la région 2S×2S autour de chaque centre.
        
        Args:
            lab_image: Image Lab (H, W, 3)
            labels: Array de labels actuel (H, W)
            distances: Array de distances actuelles (H, W)
        
        Returns:
            labels: Labels mis à jour
            distances: Distances mises à jour
        """
        h, w = lab_image.shape[:2]
        
        # Réinitialiser les pixels de chaque cluster
        for cluster in self.clusters:
            cluster.clear_pixels()
        
        # Pour chaque cluster
        for cluster in self.clusters:
            center = cluster.center
            center_lab = center[:3]
            center_x, center_y = int(center[3]), int(center[4])
            
            # Définir la région de recherche 2S×2S
            x_min = max(0, center_x - self.S)
            x_max = min(w, center_x + self.S)
            y_min = max(0, center_y - self.S)
            y_max = min(h, center_y + self.S)
            
            # Extraire les pixels de la région
            region_lab = lab_image[y_min:y_max, x_min:x_max]
            region_h, region_w = region_lab.shape[:2]
            
            # Créer les coordonnées spatiales
            yy, xx = np.meshgrid(range(y_min, y_max), range(x_min, x_max), indexing='ij')
            
            # Reshape pour le calcul vectorisé
            pixels_lab = region_lab.reshape(-1, 3)
            pixels_x = xx.ravel()
            pixels_y = yy.ravel()
            pixels_xy = np.stack([pixels_x, pixels_y], axis=1)
            
            # Calculer les distances
            center_xy = np.array([center_x, center_y])
            dist = compute_distance_vectorized(
                pixels_lab, center_lab, 
                pixels_xy, center_xy,
                self.compactness, self.S
            )
            
            # Reshape les distances
            dist = dist.reshape(region_h, region_w)
            
            # Mettre à jour les labels et distances si plus proche
            region_distances = distances[y_min:y_max, x_min:x_max]
            mask = dist < region_distances
            
            # Mise à jour
            distances[y_min:y_max, x_min:x_max][mask] = dist[mask]
            labels[y_min:y_max, x_min:x_max][mask] = cluster.label
            
            # Ajouter les pixels au cluster
            indices = np.where(mask)
            for i in range(len(indices[0])):
                pixel_y = y_min + indices[0][i]
                pixel_x = x_min + indices[1][i]
                cluster.add_pixel([pixel_y, pixel_x])
        
        return labels, distances
    
    def update_centers(self, lab_image):
        """
        Met à jour les centres de tous les clusters.
        
        Args:
            lab_image: Image Lab (H, W, 3)
        """
        for cluster in self.clusters:
            cluster.update_center(lab_image)
    
    def get_centers(self):
        """
        Retourne les centres de tous les clusters.
        
        Returns:
            centers: Array (K, 5) avec [L, a, b, x, y]
        """
        return np.array([cluster.center for cluster in self.clusters])
    
    def get_cluster_sizes(self):
        """
        Retourne les tailles de tous les clusters.
        
        Returns:
            sizes: Array (K,) avec le nombre de pixels par cluster
        """
        return np.array([cluster.get_size() for cluster in self.clusters])


def initialize_grid_centers(lab_image, S):
    """
    Initialise les centres sur une grille régulière.
    
    Args:
        lab_image: Image Lab (H, W, 3)
        S: Espacement de la grille
    
    Returns:
        centers: Array (K, 5) avec [L, a, b, x, y]
    """
    h, w = lab_image.shape[:2]
    centers = []
    
    # Parcourir la grille
    y_coords = range(S // 2, h - S // 2, S)
    x_coords = range(S // 2, w - S // 2, S)
    
    for y in y_coords:
        for x in x_coords:
            # Extraire les valeurs Lab
            L, a, b = lab_image[y, x]
            centers.append([L, a, b, x, y])
    
    return np.array(centers)


def perturb_centers_gradient(lab_image, centers, window_size=3):
    """
    Déplace les centres vers les positions de gradient minimal.
    Évite de placer les centres sur des contours.
    
    Args:
        lab_image: Image Lab (H, W, 3)
        centers: Array (K, 5) avec [L, a, b, x, y]
        window_size: Taille de la fenêtre de recherche (default: 3)
    
    Returns:
        perturbed_centers: Centres ajustés
    """
    h, w = lab_image.shape[:2]
    L_channel = lab_image[:, :, 0]
    
    # Calculer le gradient (magnitude)
    gradient = compute_gradient_magnitude(L_channel)
    
    perturbed_centers = []
    half_window = window_size // 2
    
    for center in centers:
        x, y = int(center[3]), int(center[4])
        
        # Définir la fenêtre de recherche
        y_min = max(0, y - half_window)
        y_max = min(h, y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(w, x + half_window + 1)
        
        # Trouver le pixel avec le gradient minimal
        window_gradient = gradient[y_min:y_max, x_min:x_max]
        min_idx = np.unravel_index(np.argmin(window_gradient), window_gradient.shape)
        
        # Nouvelles coordonnées
        new_y = y_min + min_idx[0]
        new_x = x_min + min_idx[1]
        
        # Mettre à jour le centre
        new_center = [
            lab_image[new_y, new_x, 0],
            lab_image[new_y, new_x, 1],
            lab_image[new_y, new_x, 2],
            new_x,
            new_y
        ]
        perturbed_centers.append(new_center)
    
    return np.array(perturbed_centers)


def compute_gradient_magnitude(image_channel):
    """
    Calcule la magnitude du gradient d'un canal d'image.
    
    Args:
        image_channel: Canal d'image 2D (H, W)
    
    Returns:
        gradient_magnitude: Magnitude du gradient (H, W)
    """
    # Gradients de Sobel
    grad_x = np.zeros_like(image_channel)
    grad_y = np.zeros_like(image_channel)
    
    # Gradient horizontal (approximation centrée)
    grad_x[:, 1:-1] = (image_channel[:, 2:] - image_channel[:, :-2]) / 2.0
    
    # Gradient vertical (approximation centrée)
    grad_y[1:-1, :] = (image_channel[2:, :] - image_channel[:-2, :]) / 2.0
    
    # Magnitude
    gradient_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    
    return gradient_magnitude


def check_convergence(old_centers, new_centers, tolerance=1e-4):
    """
    Vérifie si l'algorithme a convergé.
    
    Args:
        old_centers: Anciens centres (K, 5)
        new_centers: Nouveaux centres (K, 5)
        tolerance: Seuil de convergence
    
    Returns:
        converged: True si convergé
    """
    if old_centers is None:
        return False
    
    # Calculer la différence maximale
    max_diff = np.max(np.abs(new_centers - old_centers))
    
    return max_diff < tolerance


def compute_cluster_statistics(clusters):
    """
    Calcule des statistiques sur les clusters.
    
    Args:
        clusters: Liste de SLICCluster
    
    Returns:
        stats: Dictionnaire avec les statistiques
    """
    sizes = [cluster.get_size() for cluster in clusters]
    
    stats = {
        'n_clusters': len(clusters),
        'mean_size': np.mean(sizes),
        'std_size': np.std(sizes),
        'min_size': np.min(sizes),
        'max_size': np.max(sizes),
        'median_size': np.median(sizes)
    }
    
    return stats
"""
Implémentation de la méthode SLIC (Simple Linear Iterative Clustering)
Basée sur l'article: Achanta et al. (2012) "SLIC Superpixels Compared to 
State-of-the-art Superpixel Methods" IEEE TPAMI
https://ieeexplore.ieee.org/document/6205760

Algorithme:
1. Initialiser K centres de clusters sur une grille régulière (espacement S)
2. Déplacer les centres vers les positions de gradient minimal (3×3)
3. Pour chaque itération:
   a. Pour chaque centre, assigner les pixels dans une région 2S×2S au cluster le plus proche
   b. Calculer la nouvelle position de chaque centre comme moyenne des pixels assignés
   c. Calculer l'erreur résiduelle E
4. Post-traiter: imposer la connectivité en fusionnant les petites régions isolées
"""
import numpy as np
from src.utils.color_space import rgb_to_lab
from src.utils.distance import compute_distance_vectorized


class SLIC:
    """
    Simple Linear Iterative Clustering (SLIC) pour la segmentation en superpixels.
    
    L'algorithme opère dans un espace 5D combinant couleur Lab et position spatiale.
    La distance SLIC est définie comme:
        D = sqrt(d_lab^2 + (d_xy/S)^2 * m^2)
    où:
        - d_lab: distance euclidienne dans l'espace Lab
        - d_xy: distance euclidienne spatiale
        - S: espacement de la grille (taille approximative des superpixels)
        - m: paramètre de compacité
    
    Paramètres:
        n_segments (int): Nombre approximatif de superpixels désirés (default: 100)
        compactness (float): Paramètre m contrôlant la balance entre cohérence 
                            couleur et spatiale (default: 10.0)
                            - Valeurs faibles (1-5): superpixels irréguliers, 
                              suivent mieux les contours
                            - Valeurs moyennes (10-20): bon compromis
                            - Valeurs élevées (>20): superpixels très compacts et réguliers
        max_iter (int): Nombre maximum d'itérations (default: 10)
        sigma (float): Écart-type pour le flou gaussien préalable (default: 0 = pas de flou)
        enforce_connectivity (bool): Si True, impose la connectivité spatiale (default: True)
        min_size_factor (float): Taille minimale d'un superpixel en fraction de S² (default: 0.25)
    
    Attributs:
        centers: Centres des clusters après convergence (K, 5) [L, a, b, x, y]
        labels: Matrice des labels (H, W)
        distances: Distances minimales de chaque pixel à son centre (H, W)
        S: Espacement de grille calculé
        n_superpixels_: Nombre réel de superpixels après post-traitement
    """
    
    def __init__(self, n_segments=100, compactness=10.0, max_iter=10, 
                 sigma=0, enforce_connectivity=True, min_size_factor=0.25):
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.sigma = sigma
        self.enforce_connectivity = enforce_connectivity
        self.min_size_factor = min_size_factor
        
        # Attributs initialisés lors du fit
        self.centers = None
        self.labels = None
        self.distances = None
        self.S = None
        self.n_superpixels_ = None
    
    def fit(self, image):
        """
        Applique l'algorithme SLIC sur une image.
        
        Args:
            image: Image RGB (H, W, 3) avec valeurs dans [0, 255] ou [0, 1]
        
        Returns:
            labels: Matrice des labels (H, W), chaque valeur représente 
                   l'identifiant du superpixel
        """
        # Normalisation et conversion Lab
        if image.max() > 1.0:
            image = image / 255.0
        lab_image = rgb_to_lab(image)
        
        h, w = lab_image.shape[:2]
        
        # Étape 1: Calculer l'espacement de grille S
        # S = sqrt(N / K) où N = nombre total de pixels, K = nombre de superpixels
        self.S = int(np.sqrt(h * w / self.n_segments))
        
        # Étape 2: Initialiser les centres sur une grille régulière
        self.centers = self._initialize_grid_centers(lab_image)
        
        # Étape 3: Perturber les centres vers les positions de gradient minimal
        self.centers = self._perturb_centers(lab_image)
        
        # Initialiser les labels et distances
        self.labels = -np.ones((h, w), dtype=np.int32)
        self.distances = np.full((h, w), np.inf, dtype=np.float64)
        
        # Étape 4: Itérations K-means
        print(f"SLIC: Démarrage avec {len(self.centers)} centres, S={self.S}")
        
        for iteration in range(self.max_iter):
            # 4a: Assignment - assigner chaque pixel au centre le plus proche
            self._assign_pixels(lab_image)
            
            # 4b: Update - recalculer les centres
            old_centers = self.centers.copy()
            self._update_centers(lab_image)
            
            # 4c: Vérifier la convergence
            max_change = np.max(np.abs(self.centers - old_centers))
            if max_change < 1e-4:
                print(f"  Convergence atteinte à l'itération {iteration + 1}")
                break
        else:
            print(f"  Nombre maximum d'itérations atteint ({self.max_iter})")
        
        # Étape 5: Post-traitement - imposer la connectivité
        if self.enforce_connectivity:
            self.labels = self._enforce_connectivity(lab_image)
        
        # Compter le nombre réel de superpixels
        self.n_superpixels_ = len(np.unique(self.labels))
        print(f"  Nombre final de superpixels: {self.n_superpixels_}")
        
        return self.labels
    
    def _initialize_grid_centers(self, lab_image):
        """
        Initialise les centres sur une grille régulière espacée de S pixels.
        
        Args:
            lab_image: Image Lab (H, W, 3)
        
        Returns:
            centers: Array (K, 5) avec [L, a, b, x, y]
        """
        h, w = lab_image.shape[:2]
        centers = []
        
        # Parcourir la grille avec espacement S
        # Commencer à S/2 pour centrer la grille
        for y in range(self.S // 2, h, self.S):
            for x in range(self.S // 2, w, self.S):
                # Extraire les valeurs Lab du pixel
                L, a, b = lab_image[y, x]
                centers.append([L, a, b, x, y])
        
        return np.array(centers, dtype=np.float64)
    
    def _compute_gradient(self, lab_image):
        """
        Calcule la magnitude du gradient de l'image (canal L uniquement).
        Utilisé pour détecter les contours.
        
        Args:
            lab_image: Image Lab (H, W, 3)
        
        Returns:
            gradient: Magnitude du gradient (H, W)
        """
        L = lab_image[:, :, 0]
        
        # Gradient de Sobel (approximation centrée)
        grad_x = np.zeros_like(L)
        grad_y = np.zeros_like(L)
        
        # Gradient horizontal
        grad_x[:, 1:-1] = (L[:, 2:] - L[:, :-2]) / 2.0
        
        # Gradient vertical
        grad_y[1:-1, :] = (L[2:, :] - L[:-2, :]) / 2.0
        
        # Magnitude du gradient
        gradient = np.sqrt(grad_x ** 2 + grad_y ** 2)
        
        return gradient
    
    def _perturb_centers(self, lab_image, window_size=3):
        """
        Déplace chaque centre vers la position de gradient minimal 
        dans un voisinage 3×3.
        
        Cela évite de placer les centres sur des contours d'objets.
        
        Args:
            lab_image: Image Lab (H, W, 3)
            window_size: Taille de la fenêtre de recherche (default: 3)
        
        Returns:
            perturbed_centers: Centres ajustés (K, 5)
        """
        gradient = self._compute_gradient(lab_image)
        h, w = lab_image.shape[:2]
        
        perturbed_centers = []
        half_window = window_size // 2
        
        for center in self.centers:
            x, y = int(center[3]), int(center[4])
            
            # Définir la fenêtre de recherche
            y_min = max(0, y - half_window)
            y_max = min(h, y + half_window + 1)
            x_min = max(0, x - half_window)
            x_max = min(w, x + half_window + 1)
            
            # Trouver le pixel avec le gradient minimal dans la fenêtre
            window_gradient = gradient[y_min:y_max, x_min:x_max]
            min_idx = np.unravel_index(np.argmin(window_gradient), 
                                       window_gradient.shape)
            
            # Nouvelles coordonnées
            new_y = y_min + min_idx[0]
            new_x = x_min + min_idx[1]
            
            # Créer le nouveau centre avec les valeurs Lab du nouveau pixel
            new_center = [
                lab_image[new_y, new_x, 0],  # L
                lab_image[new_y, new_x, 1],  # a
                lab_image[new_y, new_x, 2],  # b
                new_x,                        # x
                new_y                         # y
            ]
            perturbed_centers.append(new_center)
        
        return np.array(perturbed_centers, dtype=np.float64)
    
    def _assign_pixels(self, lab_image):
        """
        Assigne chaque pixel au centre le plus proche.
        Pour l'efficacité, la recherche est limitée à une région 2S×2S 
        autour de chaque centre.
        
        Args:
            lab_image: Image Lab (H, W, 3)
        """
        h, w = lab_image.shape[:2]
        
        # Pour chaque centre
        for i, center in enumerate(self.centers):
            center_lab = center[:3]
            center_x, center_y = int(center[3]), int(center[4])
            
            # Définir la région de recherche 2S×2S autour du centre
            x_min = max(0, center_x - self.S)
            x_max = min(w, center_x + self.S)
            y_min = max(0, center_y - self.S)
            y_max = min(h, center_y + self.S)
            
            # Extraire les pixels de la région
            region_lab = lab_image[y_min:y_max, x_min:x_max]
            region_h, region_w = region_lab.shape[:2]
            
            # Créer les coordonnées spatiales de la région
            yy, xx = np.meshgrid(range(y_min, y_max), range(x_min, x_max), 
                                indexing='ij')
            
            # Reshape pour le calcul vectorisé
            pixels_lab = region_lab.reshape(-1, 3)
            pixels_x = xx.ravel()
            pixels_y = yy.ravel()
            pixels_xy = np.stack([pixels_x, pixels_y], axis=1)
            
            # Calculer les distances SLIC pour tous les pixels de la région
            center_xy = np.array([center_x, center_y])
            distances = compute_distance_vectorized(
                pixels_lab, center_lab,
                pixels_xy, center_xy,
                self.compactness, self.S
            )
            
            # Reshape les distances
            distances = distances.reshape(region_h, region_w)
            
            # Mettre à jour les labels si la distance est plus petite
            region_distances = self.distances[y_min:y_max, x_min:x_max]
            mask = distances < region_distances
            
            self.distances[y_min:y_max, x_min:x_max][mask] = distances[mask]
            self.labels[y_min:y_max, x_min:x_max][mask] = i
    
    def _update_centers(self, lab_image):
        """
        Recalcule les centres comme la moyenne des pixels assignés à chaque cluster.
        
        Args:
            lab_image: Image Lab (H, W, 3)
        """
        h, w = lab_image.shape[:2]
        
        # Créer une grille de coordonnées
        yy, xx = np.mgrid[0:h, 0:w]
        
        # Pour chaque centre
        for i in range(len(self.centers)):
            # Masque des pixels assignés au centre i
            mask = (self.labels == i)
            
            if np.sum(mask) > 0:
                # Calculer la moyenne des valeurs Lab
                L_mean = np.mean(lab_image[mask, 0])
                a_mean = np.mean(lab_image[mask, 1])
                b_mean = np.mean(lab_image[mask, 2])
                
                # Calculer la moyenne des positions spatiales
                x_mean = np.mean(xx[mask])
                y_mean = np.mean(yy[mask])
                
                # Mettre à jour le centre
                self.centers[i] = [L_mean, a_mean, b_mean, x_mean, y_mean]
    
    def _enforce_connectivity(self, lab_image):
        """
        Post-traitement pour imposer la connectivité spatiale des superpixels.
        
        Les petites régions isolées (< min_size) sont fusionnées avec le superpixel
        voisin le plus proche en couleur Lab.
        
        Algorithme:
        1. Parcourir tous les pixels
        2. Pour chaque nouveau segment connecté trouvé (flood-fill)
        3. Si le segment est trop petit, le fusionner avec son voisin le plus proche
        
        Args:
            lab_image: Image Lab (H, W, 3)
        
        Returns:
            new_labels: Labels avec connectivité imposée (H, W)
        """
        h, w = lab_image.shape[:2]
        min_size = int(self.S * self.S * self.min_size_factor)
        
        # Nouveau tableau de labels
        new_labels = np.full((h, w), -1, dtype=np.int32)
        label_counter = 0
        
        # Parcourir tous les pixels
        for start_y in range(h):
            for start_x in range(w):
                # Si déjà visité, passer
                if new_labels[start_y, start_x] >= 0:
                    continue
                
                # Démarrer un nouveau segment connecté via flood-fill
                original_label = self.labels[start_y, start_x]
                segment_pixels = []
                stack = [(start_y, start_x)]
                new_labels[start_y, start_x] = label_counter
                
                # Flood-fill pour trouver tous les pixels connectés
                while stack:
                    cy, cx = stack.pop()
                    segment_pixels.append((cy, cx))
                    
                    # Vérifier les 4 voisins (connectivité 4)
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        
                        # Vérifier les limites et les conditions
                        if (0 <= ny < h and 0 <= nx < w and
                            new_labels[ny, nx] == -1 and
                            self.labels[ny, nx] == original_label):
                            
                            new_labels[ny, nx] = label_counter
                            stack.append((ny, nx))
                
                # Vérifier si le segment est trop petit
                if len(segment_pixels) < min_size:
                    # Trouver les labels et couleurs des voisins
                    neighbor_labels = []
                    neighbor_colors = []
                    
                    for py, px in segment_pixels:
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = py + dy, px + dx
                            
                            if (0 <= ny < h and 0 <= nx < w and
                                new_labels[ny, nx] >= 0 and
                                new_labels[ny, nx] != label_counter):
                                
                                neighbor_labels.append(new_labels[ny, nx])
                                neighbor_colors.append(lab_image[ny, nx])
                    
                    # Si des voisins existent, fusionner avec le plus proche
                    if neighbor_labels:
                        # Calculer la couleur moyenne du segment actuel
                        segment_colors = np.array([lab_image[py, px] 
                                                  for py, px in segment_pixels])
                        segment_mean_color = np.mean(segment_colors, axis=0)
                        
                        # Calculer les distances Lab aux voisins
                        neighbor_colors = np.array(neighbor_colors)
                        neighbor_labels_arr = np.array(neighbor_labels)
                        
                        distances = np.sqrt(np.sum(
                            (neighbor_colors - segment_mean_color) ** 2, axis=1
                        ))
                        
                        # Fusionner avec le voisin le plus proche en couleur
                        best_neighbor_idx = np.argmin(distances)
                        best_neighbor_label = neighbor_labels_arr[best_neighbor_idx]
                        
                        # Réassigner tous les pixels du petit segment
                        for py, px in segment_pixels:
                            new_labels[py, px] = best_neighbor_label
                    else:
                        # Pas de voisin (cas rare), garder le segment
                        label_counter += 1
                else:
                    # Segment assez grand, incrémenter le compteur
                    label_counter += 1
        
        # Renuméroter les labels de manière contiguë (0, 1, 2, ...)
        unique_labels = np.unique(new_labels)
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        
        final_labels = np.zeros_like(new_labels)
        for old_label, new_label in label_mapping.items():
            final_labels[new_labels == old_label] = new_label
        
        return final_labels
    
    def get_boundaries(self, thickness=1):
        """
        Extrait les contours des superpixels.
        
        Args:
            thickness: Épaisseur des contours en pixels (default: 1)
        
        Returns:
            boundaries: Masque binaire des contours (H, W)
        """
        if self.labels is None:
            raise ValueError("fit() doit être appelé avant get_boundaries()")
        
        h, w = self.labels.shape
        boundaries = np.zeros((h, w), dtype=bool)
        
        # Détecter les transitions de labels entre pixels adjacents
        # Contours horizontaux (différence entre lignes adjacentes)
        boundaries[:-1, :] |= (self.labels[:-1, :] != self.labels[1:, :])
        
        # Contours verticaux (différence entre colonnes adjacentes)
        boundaries[:, :-1] |= (self.labels[:, :-1] != self.labels[:, 1:])
        
        # Optionnel: épaissir les contours par dilatation
        if thickness > 1:
            from scipy import ndimage
            struct = ndimage.generate_binary_structure(2, 1)
            boundaries = ndimage.binary_dilation(
                boundaries, structure=struct, iterations=thickness-1
            )
        
        return boundaries
    
    def get_superpixel_centers(self):
        """
        Retourne les centres des superpixels après convergence.
        
        Returns:
            centers: Array (K, 5) avec [L, a, b, x, y]
        """
        if self.centers is None:
            raise ValueError("fit() doit être appelé avant get_superpixel_centers()")
        
        return self.centers.copy()
    
    def get_superpixel_masks(self):
        """
        Retourne un masque binaire pour chaque superpixel.
        
        Returns:
            masks: Dictionnaire {label: mask} où mask est un array booléen (H, W)
        """
        if self.labels is None:
            raise ValueError("fit() doit être appelé avant get_superpixel_masks()")
        
        masks = {}
        unique_labels = np.unique(self.labels)
        
        for label in unique_labels:
            masks[int(label)] = (self.labels == label)
        
        return masks
    
    def get_statistics(self):
        """
        Retourne des statistiques détaillées sur la segmentation.
        
        Returns:
            stats: Dictionnaire avec diverses statistiques
        """
        if self.labels is None:
            raise ValueError("fit() doit être appelé avant get_statistics()")
        
        unique_labels = np.unique(self.labels)
        sizes = [np.sum(self.labels == label) for label in unique_labels]
        
        stats = {
            'n_superpixels': len(unique_labels),
            'mean_size': float(np.mean(sizes)),
            'std_size': float(np.std(sizes)),
            'min_size': int(np.min(sizes)),
            'max_size': int(np.max(sizes)),
            'median_size': float(np.median(sizes)),
            'grid_spacing_S': int(self.S),
            'compactness_param': float(self.compactness),
            'total_pixels': int(self.labels.size)
        }
        
        return stats
    
    def predict(self, image):
        """
        Alias pour fit() pour compatibilité avec sklearn.
        
        Args:
            image: Image RGB (H, W, 3)
        
        Returns:
            labels: Matrice des labels (H, W)
        """
        return self.fit(image)
"""
SLIC_IPOL: implémentation fidèle à l'article IPOL (Gay et al. 2022) / SLIC original
Conforme au pseudocode IPOL :
 - initialisation sur grille
 - perturbation 3x3 par minimisation du gradient (canal L)
 - assignation dans fenêtre 2S x 2S avec la distance SLIC
 - mise à jour des centres par moyenne
 - enforcement de connectivité EXACT : garder la plus grande composante, réassigner
   les petites composantes en regardant les voisins 4-connectés (premier voisin rencontré)

Dépendances requises (présentes dans votre dépôt):
 - src.utils.color_space.rgb_to_lab
 - src.utils.distance.compute_distance_vectorized

Cette classe évite heuristiques supplémentaires (pas de Dijkstra, pas d'adoption par couleur),
conforme à l'algorithme IPOL tel que requis.

Usage:
    from src.methods.slic.slic_ipol import SLIC_IPOL
    slic = SLIC_IPOL(n_segments=200, compactness=10.0, max_iter=10)
    labels = slic.fit(image)

"""

from typing import Tuple, List
import numpy as np
from collections import deque

from src.utils.color_space import rgb_to_lab
from src.utils.distance import compute_distance_vectorized


class SLIC_IPOL:
    """Classe SLIC conforme à l'implémentation IPOL (pas d'ajouts heuristiques).

    Paramètres principaux:
        n_segments: nombre souhaité de superpixels
        compactness: paramètre m (poids spatiale vs couleur)
        max_iter: itérations k-means
        enforce_connectivity: toujours True pour IPOL
        eps: seuil de convergence des centres
    """

    def __init__(self, n_segments: int = 100, compactness: float = 10.0,
                 max_iter: int = 10, eps: float = 1e-4):
        self.n_segments = int(n_segments)
        self.compactness = float(compactness)
        self.max_iter = int(max_iter)
        self.eps = float(eps)

        # attributs mis à jour par fit
        self.centers = None  # (K,5): [L,a,b,x,y]
        self.labels = None   # (H,W)
        self.distances = None
        self.S = None
        self.n_superpixels_ = None
        self.lab_image = None

    def fit(self, image: np.ndarray) -> np.ndarray:
        """Exécute SLIC/IPOL et retourne les labels (H,W).

        image: RGB image, valeurs dans [0,1] ou [0,255]
        """
        if image.max() > 1.0:
            image = image / 255.0

        self.lab_image = rgb_to_lab(image)
        h, w = self.lab_image.shape[:2]

        N = h * w
        self.S = int(np.sqrt(N / max(1, self.n_segments)))
        if self.S < 1:
            self.S = 1

        # initialiser centres sur grille
        self.centers = self._initialize_grid_centers()

        # perturber centres vers minimum de gradient (3x3)
        self.centers = self._perturb_centers()

        # initialiser labels et distances
        self.labels = -np.ones((h, w), dtype=np.int32)
        self.distances = np.full((h, w), np.inf, dtype=np.float64)

        # itérations k-means
        for it in range(self.max_iter):
            # reset distances pour cette itération
            self.distances.fill(np.inf)
            self._assign_pixels()

            old_centers = self.centers.copy()
            self._update_centers()

            max_move = np.max(np.sqrt(np.sum((self.centers[:, 3:5] - old_centers[:, 3:5]) ** 2, axis=1)))
            if max_move < self.eps:
                break

        # enforcement de connectivité selon IPOL
        self.labels = self._enforce_connectivity_ipol()
        self.n_superpixels_ = len(np.unique(self.labels))

        return self.labels

    def _initialize_grid_centers(self) -> np.ndarray:
        h, w = self.lab_image.shape[:2]
        centers = []
        # démarrer à S/2 pour centrer la grille (comme dans papier)
        start_y = self.S // 2
        start_x = self.S // 2
        for y in range(start_y, h, self.S):
            for x in range(start_x, w, self.S):
                L, a, b = self.lab_image[y, x]
                centers.append([L, a, b, float(x), float(y)])
        return np.array(centers, dtype=np.float64)

    def _compute_gradient(self) -> np.ndarray:
        L = self.lab_image[:, :, 0]
        grad_x = np.zeros_like(L)
        grad_y = np.zeros_like(L)
        grad_x[:, 1:-1] = (L[:, 2:] - L[:, :-2]) / 2.0
        grad_y[1:-1, :] = (L[2:, :] - L[:-2, :]) / 2.0
        return np.sqrt(grad_x ** 2 + grad_y ** 2)

    def _perturb_centers(self) -> np.ndarray:
        """Perturbe chaque centre vers le pixel de gradient minimal dans une fenêtre 3x3.
        Strictement 3x3 (IPOL/achanta).
        """
        gradient = self._compute_gradient()
        h, w = gradient.shape
        perturbed = []
        half = 1  # window 3x3
        for center in self.centers:
            x = int(round(center[3]))
            y = int(round(center[4]))
            y_min = max(0, y - half)
            y_max = min(h, y + half + 1)
            x_min = max(0, x - half)
            x_max = min(w, x + half + 1)
            win = gradient[y_min:y_max, x_min:x_max]
            rel_idx = np.unravel_index(np.argmin(win), win.shape)
            new_y = y_min + rel_idx[0]
            new_x = x_min + rel_idx[1]
            L, a, b = self.lab_image[new_y, new_x]
            perturbed.append([L, a, b, float(new_x), float(new_y)])
        return np.array(perturbed, dtype=np.float64)

    def _assign_pixels(self) -> None:
        """Assigne les pixels aux centres sur une fenêtre 2S x 2S autour de chaque centre.
        Utilise compute_distance_vectorized pour accélérer.
        """
        h, w = self.lab_image.shape[:2]
        yy_all, xx_all = np.mgrid[0:h, 0:w]

        for i, center in enumerate(self.centers):
            cx = int(round(center[3]))
            cy = int(round(center[4]))
            x_min = max(0, cx - self.S)
            x_max = min(w, cx + self.S + 1)
            y_min = max(0, cy - self.S)
            y_max = min(h, cy + self.S + 1)

            region_lab = self.lab_image[y_min:y_max, x_min:x_max]
            region_h, region_w = region_lab.shape[:2]

            # coordonnées spatiales
            yy = yy_all[y_min:y_max, x_min:x_max].ravel()
            xx = xx_all[y_min:y_max, x_min:x_max].ravel()
            pixels_xy = np.stack([xx, yy], axis=1)
            pixels_lab = region_lab.reshape(-1, 3)

            center_lab = center[:3]
            center_xy = np.array([center[3], center[4]])

            distances = compute_distance_vectorized(
                pixels_lab, center_lab, pixels_xy, center_xy,
                self.compactness, max(1, self.S)
            )
            distances = distances.reshape(region_h, region_w)

            # comparer et mettre à jour
            current_region_dist = self.distances[y_min:y_max, x_min:x_max]
            mask = distances < current_region_dist
            if np.any(mask):
                # mettre à jour distances et labels
                self.distances[y_min:y_max, x_min:x_max][mask] = distances[mask]
                self.labels[y_min:y_max, x_min:x_max][mask] = i

    def _update_centers(self) -> None:
        h, w = self.lab_image.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]

        for i in range(len(self.centers)):
            mask = (self.labels == i)
            if np.any(mask):
                L_mean = np.mean(self.lab_image[:, :, 0][mask])
                a_mean = np.mean(self.lab_image[:, :, 1][mask])
                b_mean = np.mean(self.lab_image[:, :, 2][mask])
                x_mean = np.mean(xx[mask])
                y_mean = np.mean(yy[mask])
                self.centers[i] = [L_mean, a_mean, b_mean, float(x_mean), float(y_mean)]

    # ---------- Connectivité IPOL (strict) ----------
    def _enforce_connectivity_ipol(self) -> np.ndarray:
        """Impose la connectivité selon IPOL :
        - pour chaque label original, trouver ses composantes connexes (4-connexité)
        - garder la plus grande composante, réassigner les autres en regardant les voisins
        """
        h, w = self.labels.shape
        visited = np.zeros((h, w), dtype=bool)
        new_labels = -np.ones((h, w), dtype=np.int32)
        current_label = 0

        # map original label -> list of components (each component is list of (y,x))
        components_map = {}

        original_labels = np.unique(self.labels[self.labels >= 0])
        for lab in original_labels:
            components_map[lab] = []

        # parcourir tous les pixels pour extraire composantes
        for y in range(h):
            for x in range(w):
                if visited[y, x]:
                    continue
                orig = self.labels[y, x]
                # BFS flood-fill
                q = deque()
                q.append((y, x))
                comp = []
                visited[y, x] = True
                while q:
                    cy, cx = q.popleft()
                    if self.labels[cy, cx] != orig:
                        continue
                    comp.append((cy, cx))
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                            if self.labels[ny, nx] == orig:
                                visited[ny, nx] = True
                                q.append((ny, nx))
                # stocker composante
                if orig >= 0:
                    components_map[orig].append(comp)

        # pour chaque label original, garder la plus grande composante et assigner un nouveau label
        kept_components = []  # liste de pixels assignés déjà
        orphan_pixels = []

        for orig, comps in components_map.items():
            if len(comps) == 0:
                continue
            # trouver la plus grande
            sizes = [len(c) for c in comps]
            largest_idx = int(np.argmax(sizes))
            for idx, comp in enumerate(comps):
                if idx == largest_idx:
                    # assignation directe à current_label
                    for (py, px) in comp:
                        new_labels[py, px] = current_label
                    current_label += 1
                else:
                    # orphelins: marquer pour réassignation
                    for (py, px) in comp:
                        orphan_pixels.append((py, px))

        # réassigner les orphelins en regardant les voisins 4-connexes (premier voisin rencontré)
        # si aucun voisin labellé immédiat, on effectue une petite recherche BFS jusqu'à trouver
        for (oy, ox) in orphan_pixels:
            assigned = False
            # regarder voisins immédiats
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = oy + dy, ox + dx
                if 0 <= ny < h and 0 <= nx < w and new_labels[ny, nx] >= 0:
                    new_labels[oy, ox] = int(new_labels[ny, nx])
                    assigned = True
                    break
            if assigned:
                continue
            # fallback: BFS croissant pour trouver premier pixel labellé
            q = deque()
            q.append((oy, ox))
            seen = set()
            seen.add((oy, ox))
            found_label = -1
            while q and found_label == -1:
                cy, cx = q.popleft()
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        if new_labels[ny, nx] >= 0:
                            found_label = int(new_labels[ny, nx])
                            break
                        q.append((ny, nx))
            if found_label >= 0:
                new_labels[oy, ox] = found_label
            else:
                # cas extrême: aucun voisin labellé (image très petite ou erreur), assigner label 0
                new_labels[oy, ox] = 0

        # renuméroter contigu
        unique_final = np.unique(new_labels)
        mapping = {old: new for new, old in enumerate(unique_final)}
        final = np.full_like(new_labels, -1)
        for old_label, new_label in mapping.items():
            final[new_labels == old_label] = new_label

        return final

    # utilitaires
    def get_boundaries(self, thickness: int = 1) -> np.ndarray:
        if self.labels is None:
            raise ValueError("fit() must be called before get_boundaries()")
        h, w = self.labels.shape
        boundaries = np.zeros((h, w), dtype=bool)
        boundaries[:-1, :] |= (self.labels[:-1, :] != self.labels[1:, :])
        boundaries[:, :-1] |= (self.labels[:, :-1] != self.labels[:, 1:])
        if thickness > 1:
            from scipy import ndimage
            struct = ndimage.generate_binary_structure(2, 1)
            boundaries = ndimage.binary_dilation(boundaries, structure=struct, iterations=thickness-1)
        return boundaries

    def get_statistics(self) -> dict:
        if self.labels is None:
            raise ValueError("fit() must be called before get_statistics()")
        unique = np.unique(self.labels)
        sizes = [int(np.sum(self.labels == lab)) for lab in unique]
        return {
            'n_superpixels': int(len(unique)),
            'mean_size': float(np.mean(sizes)),
            'std_size': float(np.std(sizes)),
            'min_size': int(np.min(sizes)),
            'max_size': int(np.max(sizes)),
            'grid_spacing_S': int(self.S),
            'compactness_param': float(self.compactness),
            'total_pixels': int(self.labels.size)
        }

    def predict(self, image: np.ndarray) -> np.ndarray:
        return self.fit(image)

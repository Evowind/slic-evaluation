import torch
import numpy as np
import math
import heapq
from skimage.color import rgb2lab

class SITHSS:
    def __init__(self, n_segments=200, t=0.1, tau=2e-7, max_radius=5, device=None):
        self.K = int(n_segments)
        self.t = float(t)
        self.tau = float(tau)
        self.max_radius = int(max_radius)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_features(self, image):
        H, W = image.shape[:2]
        img_lab = rgb2lab(image).astype(np.float32)
        img_lab[:,:,0] /= 100.0 
        img_lab[:,:,1:] /= 128.0
        
        tensor_lab = torch.from_numpy(img_lab).permute(2, 0, 1).to(self.device)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing='ij'
        )
        scale = max(H, W)
        pos_tensor = torch.stack([x_grid / scale, y_grid / scale], dim=0)
        return torch.cat([tensor_lab, pos_tensor], dim=0), H, W

    def _compute_initial_graph(self, features, H, W):
        prev_h1 = 0.0
        best_w, best_shifts = None, None
        
        for r in range(1, self.max_radius + 1):
            # Calcul des shifts pour le rayon r
            shifts = []
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dy == 0 and dx == 0: continue
                    shifts.append((dy, dx))
            
            # Eq 3.3 & 3.4
            feat_c, feat_s = features[:3], features[3:]
            rhos = []
            masks = []
            for dy, dx in shifts:
                c_s = torch.roll(feat_c, (-dy, -dx), (1, 2))
                s_s = torch.roll(feat_s, (-dy, -dx), (1, 2))
                mask = torch.ones((H, W), device=self.device)
                if dy > 0: mask[-dy:, :] = 0
                if dy < 0: mask[:-dy, :] = 0
                if dx > 0: mask[:, -dx:] = 0
                if dx < 0: mask[:, :-dx] = 0
                
                rho = torch.sum((feat_c - c_s)**2, 0) * torch.sqrt(torch.sum((feat_s - s_s)**2, 0))
                rhos.append(rho)
                masks.append(mask)
            
            rhos = torch.stack(rhos)
            masks = torch.stack(masks)
            mean_rho = rhos[masks > 0].mean()
            weights = torch.exp(-rhos / (self.t * mean_rho + 1e-12)) * masks
            
            # 1D SE (Eq 3.5)
            deg = weights.sum(0)
            vg = deg.sum()
            h1 = -torch.sum((deg/vg) * torch.log(deg/vg + 1e-30)).item()
            
            if r > 1 and (h1 - prev_h1) < self.tau: break
            prev_h1, best_w, best_shifts = h1, weights, shifts
            
        return best_w, best_shifts, vg.item()

    def _calculate_delta_h(self, i, j, w_ij, vol, cut, V_G):
        vi, vj = vol[i], vol[j]
        gi, gj = cut[i], cut[j]
        v_new = vi + vj
        g_new = gi + gj - 2 * w_ij
        eps = 1e-30
        
        # Eq 3.8
        term_old = (vi - gi) * math.log(vi + eps) + (vj - gj) * math.log(vj + eps)
        term_new = (v_new - g_new) * math.log(v_new + eps)
        term_glob = (2 * w_ij) * math.log(V_G + eps)
        return (term_old - term_new + term_glob) / V_G

    def fit(self, image):
        features, H, W = self._get_features(image)
        weights, shifts, V_G = self._compute_initial_graph(features, H, W)
        
        n_pixels = H * W
        adj = [{} for _ in range(n_pixels)]
        weights_np = weights.cpu().numpy()
        
        for k, (dy, dx) in enumerate(shifts):
            w_map = weights_np[k]
            ys, xs = np.where(w_map > 1e-8) # Seuil légèrement plus bas
            for y, x in zip(ys, xs):
                u = int(y * W + x)
                v = int((y + dy) * W + (x + dx))
                if 0 <= v < n_pixels:
                    val = float(w_map[y, x])
                    adj[u][v] = val
                    adj[v][u] = val # Assurer la symétrie initiale

        vol = np.array([sum(adj[i].values()) for i in range(n_pixels)], dtype=np.float64)
        # Éviter les volumes nuls qui font planter le log
        vol = np.clip(vol, 1e-12, None)
        cut = vol.copy()
        
        self.parent = np.arange(n_pixels, dtype=np.int64)
        active_regions = set(range(n_pixels))
        
        iteration = 0
        while len(active_regions) > self.K:
            iteration += 1
            best_targets = {} 
            
            # --- ÉTAPE 1: Recherche des meilleurs candidats ---
            for u in active_regions:
                if not adj[u]: continue
                
                best_v = -1
                max_d = -float('inf')
                
                for v, w in adj[u].items():
                    d = self._calculate_delta_h(u, v, w, vol, cut, V_G)
                    if d > max_d:
                        max_d = d
                        best_v = v
                
                if best_v != -1:
                    best_targets[u] = (int(best_v), max_d)

            # --- ÉTAPE 2: Sélection des fusions ---
            to_merge = []
            processed_in_layer = set()
            
            # Tri par delta décroissant
            sorted_candidates = sorted(best_targets.items(), key=lambda x: x[1][1], reverse=True)
            
            # 1. Priorité aux fusions mutuelles (MNN)
            for u, (v, d) in sorted_candidates:
                if u in processed_in_layer or v in processed_in_layer: continue
                if best_targets.get(v, (None, None))[0] == u:
                    to_merge.append((u, v))
                    processed_in_layer.add(u)
                    processed_in_layer.add(v)

            # 2. IMPORTANT : Si stagnation, on force les meilleures fusions non-mutuelles
            if not to_merge and sorted_candidates:
                for u, (v, d) in sorted_candidates:
                    if u in processed_in_layer or v in processed_in_layer: continue
                    to_merge.append((u, v))
                    processed_in_layer.add(u)
                    processed_in_layer.add(v)
                    if len(to_merge) > 100: break # On avance par petits bonds

            if not to_merge:
                print("Warning: No more possible merges found.")
                break

            # --- ÉTAPE 3: Application des fusions ---
            for u, v in to_merge:
                if len(active_regions) <= self.K: break
                if v not in active_regions or u not in active_regions: continue

                w_uv = adj[u].get(v, 0.0)
                vol[u] += vol[v]
                cut[u] = cut[u] + cut[v] - 2 * w_uv
                
                # Mise à jour du graphe
                neighbors_v = list(adj[v].items())
                for neighbor, w in neighbors_v:
                    if neighbor == u: continue
                    
                    # Mise à jour symétrique u <-> neighbor
                    new_w = adj[u].get(neighbor, 0.0) + w
                    adj[u][neighbor] = new_w
                    adj[neighbor][u] = new_w
                    
                    # Suppression de v chez le voisin
                    if v in adj[neighbor]:
                        del adj[neighbor][v]
                
                adj[v] = {} # Vider v
                adj[u].pop(v, None) # Enlever v de u
                active_regions.remove(v)
                self.parent[v] = u 
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: {len(active_regions)} regions remaining")

        return self._generate_final_labels(H, W, n_pixels)

    def _generate_final_labels(self, H, W, n_pixels):
        final_labels = np.zeros(n_pixels, dtype=np.int32)
        # Remonter à la racine pour chaque pixel (Path Compression)
        for i in range(n_pixels):
            root = i
            while self.parent[root] != root:
                root = self.parent[root]
            # Compression pour accélérer les futurs accès
            curr = i
            while self.parent[curr] != root:
                nxt = self.parent[curr]
                self.parent[curr] = root
                curr = nxt
            final_labels[i] = root
            
        # Mapper les identifiants de racine vers 0...K-1
        unique_roots = np.unique(final_labels)
        root_to_id = {root: i for i, root in enumerate(unique_roots)}
        
        # Application du mapping
        remapped_labels = np.array([root_to_id[r] for r in final_labels], dtype=np.int32)
        return remapped_labels.reshape(H, W)
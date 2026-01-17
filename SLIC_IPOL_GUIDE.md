# Guide SLIC IPOL - Version Améliorée

## Vue d'ensemble

SLIC IPOL est une version améliorée de l'algorithme SLIC basée sur l'article :

> Gay et al. (2022) "Bilateral K-Means for Superpixel Computation (the SLIC Method)"  
> Image Processing On Line, https://www.ipol.im/pub/art/2022/373/

## Améliorations par rapport à SLIC Original

### 1. **Meilleure Application de la Connectivité**

- Utilise un algorithme de propagation guidée par distance (Dijkstra)
- Adoption intelligente des pixels orphelins basée sur la similarité de couleur
- Garantit que tous les superpixels sont des régions connexes

### 2. **Gestion Optimisée des Orphelins**

- Identifie les composantes déconnectées pour chaque superpixel
- Conserve la plus grande composante
- Les petites composantes sont marquées comme "orphelins"
- Adoption des orphelins via propagation par distance Manhattan
- Choix du label voisin le plus proche en couleur Lab

### 3. **Gradient Preconditioning Flexible**

- Fenêtres de taille variable (0, 3, 5, 7, etc.)
- `gradient_window=0` : pas de preconditioning
- `gradient_window=3` : comportement SLIC original (3×3)
- `gradient_window>3` : preconditioning plus agressif

## Utilisation

### Utilisation Simple

```python
from src.methods.slic.slic_original import SLIC

# Avec SLIC IPOL activé
slic = SLIC(
    n_segments=200,
    compactness=10,
    use_ipol=True  # ← Active la version IPOL
)

labels = slic.fit(image)
```

### Comparaison avec SLIC Original

```python
# SLIC Original
slic_original = SLIC(n_segments=200, compactness=10, use_ipol=False)
labels_original = slic_original.fit(image)

# SLIC IPOL
slic_ipol = SLIC(n_segments=200, compactness=10, use_ipol=True)
labels_ipol = slic_ipol.fit(image)
```

### Paramètres Spécifiques IPOL

```python
slic = SLIC(
    n_segments=200,
    compactness=10,
    use_ipol=True,
    gradient_window=5,      # Fenêtre de gradient (0, 3, 5, 7...)
    min_size_factor=0.25    # Taille minimale des superpixels
)
```

## Scripts de Comparaison

### 1. Comparaison Directe

```bash
python3 experiments/compare_slic_ipol.py     --image data/BSDS500/data/images/test/107045.jpg --g data --n_segments 200     --compactness 10
```

**Sortie :**

- Visualisation côte-à-côte
- Métriques comparatives
- Temps d'exécution
- Graphiques de métriques

## Différences Algorithmiques Clés

### Connectivité (Phase de Post-traitement)

#### SLIC Original

```
1. Flood-fill pour trouver composantes
2. Fusionner petites composantes avec voisin le plus proche
3. Critère : distance euclidienne Lab
```

#### SLIC IPOL

```
1. Flood-fill pour trouver composantes
2. Marquer orphelins (petites + déconnectées)
3. Propagation guidée par distance (Dijkstra)
4. Critère : distance Manhattan spatiale + similarité Lab
5. Choix du voisin le plus proche en COULEUR
```

### Adoption des Orphelins

**SLIC Original :**

- Trouve le voisin spatial le plus proche
- Fusionne immédiatement

**SLIC IPOL :**

- File de priorité basée sur distance Manhattan
- Traite les orphelins par ordre de proximité aux régions labellisées
- Si plusieurs voisins : choisit le plus proche en couleur Lab
- Propagation progressive depuis les bords

## Avantages de SLIC IPOL

### ✅ Connectivité Garantie

- Tous les superpixels sont des régions 4-connexes
- Pas de composantes isolées

### ✅ Meilleure Adhérence aux Contours

- Gradient preconditioning flexible
- Orphelins adoptés selon similarité de couleur
- Superpixels plus cohérents visuellement

### ✅ Robustesse

- Gestion intelligente des cas difficiles
- Algorithme de propagation par distance
- Évite les artefacts de fragmentation

## Cas d'Usage Recommandés

### Quand utiliser SLIC IPOL (`use_ipol=True`) ?

1. **Applications nécessitant une connectivité stricte**

   - Graphes de superpixels
   - Extraction de régions
   - Segmentation hiérarchique

2. **Images avec contours complexes**

   - Textures fines
   - Frontières irrégulières
   - Structures enchevêtrées

3. **Post-traitement critique**
   - Quand les orphelins causent des problèmes
   - Besoin de superpixels homogènes
   - Visualisation de haute qualité

### Quand utiliser SLIC Original (`use_ipol=False`) ?

1. **Performance maximale**

   - Post-traitement plus rapide
   - Applications temps-réel

2. **Compatibilité**

   - Reproduction d'expériences existantes
   - Benchmarks standards

3. **Simplicité**
   - Moins de paramètres à ajuster
   - Comportement prévisible

## Métriques de Comparaison

```python
from src.evaluation.metrics import compute_all_metrics

# Calculer métriques
metrics_original = compute_all_metrics(labels_original)
metrics_ipol = compute_all_metrics(labels_ipol)

# Comparer
print(f"Compacité:")
print(f"  Original: {metrics_original['compactness']:.4f}")
print(f"  IPOL:     {metrics_ipol['compactness']:.4f}")

print(f"\nRégularité:")
print(f"  Original: {metrics_original['regularity']:.4f}")
print(f"  IPOL:     {metrics_ipol['regularity']:.4f}")
```

## Paramètres Recommandés

### Configuration Standard

```python
slic = SLIC(
    n_segments=200,
    compactness=10,
    max_iter=10,
    use_ipol=True,
    gradient_window=3  # SLIC original
)
```

### Configuration Haute Qualité

```python
slic = SLIC(
    n_segments=400,
    compactness=15,
    max_iter=15,
    use_ipol=True,
    gradient_window=5,      # Preconditioning plus agressif
    min_size_factor=0.15    # Superpixels plus petits acceptés
)
```

### Configuration Rapide

```python
slic = SLIC(
    n_segments=100,
    compactness=20,
    max_iter=5,
    use_ipol=True,
    gradient_window=0,      # Pas de preconditioning
    min_size_factor=0.4     # Fusion plus agressive
)
```

## Complexité

### Temps d'exécution

- **SLIC Original** : O(N × max_iter)
- **SLIC IPOL** : O(N × max_iter) + O(N log N) pour Dijkstra
- Surcoût typique : +10-30% du temps total

### Mémoire

- Identique à SLIC original
- Structures supplémentaires temporaires pour Dijkstra

## Exemple Complet

```python
import numpy as np
from PIL import Image
from src.methods.slic.slic_original import SLIC
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.visualize import visualize_segmentation

# Charger image
image = np.array(Image.open('image.jpg'))

# SLIC IPOL
slic = SLIC(
    n_segments=200,
    compactness=10,
    use_ipol=True,
    gradient_window=3
)

labels = slic.fit(image)

# Métriques
metrics = compute_all_metrics(labels)
print(f"Superpixels: {metrics['n_superpixels']}")
print(f"Compacité: {metrics['compactness']:.4f}")
print(f"Régularité: {metrics['regularity']:.4f}")

# Visualiser
fig = visualize_segmentation(image, labels, title="SLIC IPOL")
fig.savefig('result.png')
```

## Références

1. **Article IPOL Original**

   - Gay, R., Lecoutre, J., Menouret, N., Morillon, A., & Monasse, P. (2022)
   - "Bilateral K-Means for Superpixel Computation (the SLIC Method)"
   - Image Processing On Line, 12, 72-91
   - https://www.ipol.im/pub/art/2022/373/

2. **SLIC Original**
   - Achanta, R., et al. (2012)
   - "SLIC Superpixels Compared to State-of-the-art Superpixel Methods"
   - IEEE TPAMI

## Support

Pour toute question ou problème :

- Consultez les notebooks dans `notebooks/`
- Exécutez les tests : `python tests/test_slic.py`
- Voir les exemples : `python quick_start.py`

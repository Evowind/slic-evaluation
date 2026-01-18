import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Lire le CSV
# -----------------------------
df = pd.read_csv("benchmark_results.csv")
print(df.head())

# -----------------------------
# 2. Définir les métriques principales
# -----------------------------
metrics = [
    "global_regularity", "boundary_recall", "under_segmentation_error",
    "asa", "precision", "contour_density", "explained_variation"
]

# -----------------------------
# 3. Fonction pour retirer les outliers par méthode
# -----------------------------
def remove_outliers_by_method(df, columns, method_col='method'):
    df_filtered = pd.DataFrame()
    for method in df[method_col].unique():
        df_method = df[df[method_col] == method].copy()
        for col in columns:
            Q1 = df_method[col].quantile(0.25)
            Q3 = df_method[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_method = df_method[(df_method[col] >= lower) & (df_method[col] <= upper)]
        df_filtered = pd.concat([df_filtered, df_method], ignore_index=True)
    return df_filtered

df_no_outliers = remove_outliers_by_method(df, metrics)

# Vérifier les méthodes restantes
print(df_no_outliers['method'].value_counts())

# -----------------------------
# 4. Configuration générale Seaborn
# -----------------------------
sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# -----------------------------
# 5. Histogrammes des métriques principales
# -----------------------------
plt.figure(figsize=(16, 10))
for i, metric in enumerate(metrics, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df_no_outliers[metric], kde=True, bins=20)
    plt.title(metric)
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Courbes des métriques par n_superpixels
# -----------------------------
plt.figure(figsize=(16, 10))
for i, metric in enumerate(metrics, 1):
    plt.subplot(3, 3, i)
    sns.lineplot(data=df_no_outliers, x="n_superpixels", y=metric, hue="method", marker="o")
    plt.title(metric)
    plt.xlabel("n_superpixels")
    plt.ylabel(metric)
    if i != 1:
        plt.legend([], [], frameon=False)
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Heatmap de corrélation
# -----------------------------
corr = df_no_outliers[metrics].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corrélation entre métriques")
plt.show()

# -----------------------------
# 8. Boxplots par méthode
# -----------------------------
plt.figure(figsize=(16, 8))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 4, i)
    sns.boxplot(x="method", y=metric, data=df_no_outliers)
    plt.title(metric)
plt.tight_layout()
plt.show()

# -----------------------------
# 9. Violin plots par méthode
# -----------------------------
plt.figure(figsize=(16, 8))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 4, i)
    sns.violinplot(x="method", y=metric, data=df_no_outliers)
    plt.title(metric)
plt.tight_layout()
plt.show()

# -----------------------------
# 10. Scatter plot entre deux métriques
# -----------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x="boundary_recall", y="under_segmentation_error", data=df_no_outliers, hue="method")
plt.title("Scatter plot entre boundary_recall et under_segmentation_error")
plt.xlabel("boundary_recall")
plt.ylabel("under_segmentation_error")
plt.show()

# -----------------------------
# 11. Pairplot pour toutes les métriques
# -----------------------------
sns.pairplot(df_no_outliers[metrics + ["method"]], hue="method")
plt.show()

# -----------------------------
# 12. Distribution KDE des métriques par méthode
# -----------------------------
plt.figure(figsize=(16, 8))
for i, metric in enumerate(metrics, 1):
    plt.subplot(2, 4, i)
    sns.kdeplot(data=df_no_outliers, x=metric, hue="method", fill=True)
    plt.title(f"Distribution de {metric} par méthode")
plt.tight_layout()
plt.show()

# -----------------------------
# 13. Ligne de tendance par méthode
# -----------------------------
plt.figure(figsize=(16, 10))
for i, metric in enumerate(metrics, 1):
    plt.subplot(3, 3, i)
    sns.lineplot(data=df_no_outliers, x="n_superpixels", y=metric, hue="method", ci=None, marker="o")
    plt.title(f"Tendance de {metric} par méthode")
    plt.xlabel("n_superpixels")
    plt.ylabel(metric)
    if i != 1:
        plt.legend([], [], frameon=False)
plt.tight_layout()
plt.show()

# -----------------------------
# 14. FacetGrid par méthode
# -----------------------------
g = sns.FacetGrid(df_no_outliers, col="method", col_wrap=3, height=4)
for metric in metrics:
    g.map(sns.histplot, metric, kde=True)
g.add_legend()
plt.show()

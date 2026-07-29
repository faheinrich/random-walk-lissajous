import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def direct_multivariate_rw_pca(n_dims=40, n_steps=5000, pc_indices=(0, 1, 2), seed=42):
    """
    Generiert einen echten multivariaten Random Walk in D Dimensionen
    und führt eine direkte PCA auf der resultierenden Trajektorienmatrix durch.

    Parameters:
    -----------
    n_dims : int
        Anzahl der Dimensionen des Random Walks (D).
    n_steps : int
        Anzahl der Zeitschritte (N).
    pc_indices : tuple von 3 int (z.B. (0, 1, 2))
        Die 0-basierten Indizes der zu plottenden Hauptkomponenten (PC1, PC2, PC3).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # --- 1. Echten multivariaten Random Walk generieren ---
    # Erstellt eine Matrix der Form (N_steps, N_dims).
    # Jede der D Dimensionen führt unabhängig voneinander einen Random Walk aus.
    steps = np.random.randn(n_steps, n_dims)
    rw_trajectory = np.cumsum(steps, axis=0)  # Kumulative Summe entlang der Zeitachse (Rows)
    
    # --- 2. Direkte PCA auf der Trajektorienmatrix durchführen ---
    # Wir projizieren die N_steps x N_dims Matrix auf die wichtigsten Varianzachsen.
    pca = PCA(n_components=max(pc_indices) + 1)
    rw_pca = pca.fit_transform(rw_trajectory)
    
    # Erklärte Varianz in Prozent berechnen
    explained_var = pca.explained_variance_ratio_ * 100
    
    # --- 3. 3D Visualisierung ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Indizes mappieren (0-basiert auf 1-basierte Labels)
    idx_x, idx_y, idx_z = pc_indices
    
    t = np.arange(n_steps)
    # Punkte plotten, farbcodiert nach Zeitverlauf
    sc = ax.scatter(rw_pca[:, idx_x], rw_pca[:, idx_y], rw_pca[:, idx_z],
                    c=t, cmap='plasma', s=2, alpha=0.6)
    
    # Verbindungslinie zeichnen (Trajektorie)
    ax.plot(rw_pca[:, idx_x], rw_pca[:, idx_y], rw_pca[:, idx_z],
            color='gray', alpha=0.3, linewidth=0.5)
    
    # Achsenbeschriftungen
    ax.set_xlabel(f'PC {idx_x + 1} ({explained_var[idx_x]:.1f}%)')
    ax.set_ylabel(f'PC {idx_y + 1} ({explained_var[idx_y]:.1f}%)')
    ax.set_zlabel(f'PC {idx_z + 1} ({explained_var[idx_z]:.1f}%)')
    
    title = f'Direkte PCA eines echten {n_dims}-D Random Walks\n(Keine Phasenraum-Einbettung)'
    ax.set_title(title)
    
    # Farbleiste für die Zeit
    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label('Zeitschritt $t$')
    
    plt.tight_layout()
    plt.show()


# --- Aufruf ---
# Wir nutzen 40 Dimensionen und plottet PC1, PC2 und PC3.
direct_multivariate_rw_pca(n_dims=2000, n_steps=10000, pc_indices=(3,4,5))
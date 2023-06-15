from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm 


CACHE_FEAT_FEATURES = Path("./cache/features.txt")
CACHE_CLUS_SILHOUETTE = Path("./cache/silhouette.png")



class MyUtils:

    @staticmethod
    def plot_to_png(fname, fig):
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        cv2.imwrite(fname.as_posix(), data)

    @staticmethod
    def silhouette_plot_on_ax(ax, sample_silhouette_values, n_clusters, n_samples):
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, n_samples + (n_clusters + 1) * 10])

        y_lower = 10
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                facecolor=color, edgecolor=color, alpha=0.7,
            )

            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel(f"{n_clusters}\nCluster label")
        ax.axvline(x=silhouette_avg, color="red", linewidth=3.0)
        ax.set_yticks([]) 
        ax.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        return ax



if __name__ == "__main__":

    try:
        X = np.loadtxt(CACHE_FEAT_FEATURES, dtype=object)
    except Exception as e:
        print(f"No features found at {CACHE_FEAT_FEATURES}, run main.py first.")
        quit()    
    metrics = []

    fig, axes = plt.subplots(len(X) - 2, 1, sharex=True)
    fig.set_size_inches(5, len(axes)*3)
    for ax, n_clusters in zip(axes, tqdm(range(2, len(X)))):
        
        # Do clustering for current n_clusters
        clusterer = KMeans(n_clusters=n_clusters, n_init=3, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # Compute silhouette metrics and draw them 
        silhouette_avg = silhouette_score(X, cluster_labels)
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
        ax = MyUtils.silhouette_plot_on_ax(ax, sample_silhouette_values, n_clusters, len(X))
        metrics.append([n_clusters, silhouette_avg])

    # Save overview figure as image 
    MyUtils.plot_to_png(CACHE_CLUS_SILHOUETTE, fig)
    print("Image saved to", CACHE_CLUS_SILHOUETTE)
    
    # Result printout
    metrics = np.array(metrics) 
    best_n, best_sil = metrics[metrics[:,1].argmax(), :]
    print(f"Best n_clusters is {int(best_n)}, with sil {best_sil}")
    print("n_clusters\tsilhouette score")
    print(metrics)

    

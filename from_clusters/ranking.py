import numpy as np

def compute_thresholds(embeddings_simple):
    coords = embeddings_simple[["umap_1", "umap_2"]].values
    dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
    median_dist = np.median(dists[dists > 0])

    return {
        "close": median_dist * 0.25,
        "moderate": median_dist * 0.75,
        "far": median_dist * 1.5
    }

def label_dist(dist, thresholds):
    if dist < thresholds["close"]:
        return "close"
    elif dist < thresholds["moderate"]:
        return "moderate"
    else:
        return "far"

def rank_clusters_helper(thresholds, ovlp):
    for cluster in ovlp["failure"]:
        cluster["label"] = label_dist(cluster["min_dist"], thresholds)
    for cluster in ovlp["success"]:
        cluster["label"] = label_dist(cluster["min_dist"], thresholds)
    return ovlp

def rank_clusters(ovlp, embeddings_simple):
    thresholds = compute_thresholds(embeddings_simple)
    ovlp = rank_clusters_helper(thresholds, ovlp)
    return ovlp
def compute_centroids(cluster_features, embeddings):
    centroids = []
    cluster_ids = sorted(embeddings["cluster"].unique())
    for i, cid in enumerate(cluster_ids):
        mask = embeddings[embeddings["cluster"] == cid]
        c = mask[["umap_1", "umap_2"]].mean().values
        ids = mask["id"].values.tolist()
        cent = (c, cluster_features[i], ids)
        centroids.append(cent)
    return centroids

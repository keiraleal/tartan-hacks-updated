import hdbscan

def run_hdbscan(embed, min_clus_size=3, min_samp=2):

    clustering_model = hdbscan.HDBSCAN(
        min_cluster_size=min_clus_size,
        metric='euclidean',
        cluster_selection_method='eom',
        min_samples=min_samp
    )

    labels = clustering_model.fit_predict(embed)
    return labels
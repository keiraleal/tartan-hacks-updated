import pandas as pd
from cluster_naming.llm import semantic_cluster_name
from stats.risk import uniqueness, norm_latency, scale_risk, risk_score, risk_level
from policy_engine.risk_engine import engine_results
from stats.stats import basic_stats, cluster_stats
from stats.insights import basic_insights
from to_clusters.embeddings import get_embeddings
from to_clusters.umap import run_umap
from to_clusters.clustering import run_hdbscan
from to_clusters.vectorizing import retrieve_cluster_features
from from_clusters.centroids import compute_centroids
from from_clusters.overlap import overlaps
from from_clusters.ranking import rank_clusters
from ui.graphs import failures_by_domain_chart, latency_distribution_chart, embeddings_scatterplot

def run_pipeline(df):
    df = df.reset_index(drop=True)

    embeddings = get_embeddings(df["input_text"].tolist())

    embeddings = pd.DataFrame(embeddings)
    embeddings["id"] = df["id"].values
    embeddings = embeddings.merge(df[["id", "success"]], on="id")

    embeddings_simple = run_umap(embeddings)

    failure_embeddings = embeddings_simple[embeddings_simple["success"] == 0]
    success_embeddings = embeddings_simple[embeddings_simple["success"] == 1]

    failure_clusters = run_hdbscan(failure_embeddings[["umap_1", "umap_2"]])
    failure_embeddings["cluster"] = failure_clusters

    success_clusters = run_hdbscan(success_embeddings[["umap_1", "umap_2"]])
    success_embeddings["cluster"] = success_clusters

    failure_features = retrieve_cluster_features(failure_embeddings,df)
    success_features = retrieve_cluster_features(success_embeddings,df)

    failure_centroids = compute_centroids(failure_features,failure_embeddings)
    success_centroids = compute_centroids(success_features,success_embeddings)

    ovlps = overlaps(failure_centroids,success_centroids)
    ovlps = rank_clusters(ovlps, embeddings_simple)

    ovlps["failure"] = cluster_stats(ovlps["failure"], df)
    ovlps["success"] = cluster_stats(ovlps["success"], df)

    ovlps = semantic_cluster_name(ovlps)

    ovlps = uniqueness(ovlps)
    ovlps = norm_latency(ovlps)
    ovlps = scale_risk(ovlps)
    ovlps = risk_score(ovlps)
    ovlps = risk_level(ovlps)
    ovlps = engine_results(ovlps)

    stats_result = basic_stats(df)
    insights_result = basic_insights(stats_result)
    failures_chart = failures_by_domain_chart(stats_result["domain_failures"])
    latency_chart = latency_distribution_chart(df["latency_ms"])
    embeddings_chart = embeddings_scatterplot(embeddings_simple)

    return {
        "basic_stats": stats_result,
        "basic_insights": insights_result,
        "failures_chart": failures_chart,
        "latency_chart": latency_chart,
        "embeddings": embeddings,
        "embeddings_simple": embeddings_simple,
        "embeddings_chart": embeddings_chart,
        "failure_clusters": failure_clusters,
        "failure_embeddings": failure_embeddings,
        "success_clusters": success_clusters,
        "success_embeddings": success_embeddings,
    #    "failure_features": failure_features,
    #    "success_features": success_features,
        "failure_centroids": failure_centroids,
        "success_centroids": success_centroids,
        "overlaps": ovlps
    }
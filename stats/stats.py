def basic_stats(df):

    num_failures = (df["success"] == 0).sum()
    failure_rate = num_failures / len(df)

    avg_latency = df["latency_ms"].mean()

    domain_failures = (
        df[df["success"] == 0]
        .groupby("domain")
        .size()
        .to_dict()
    )

    domain_failure_rate = df.groupby("domain")["success"].apply(lambda x: (x == 0).sum() / len(x)).to_dict()

    return {
        "num_failures": num_failures,
        "failure_rate": failure_rate,
        "avg_latency": avg_latency,
        "domain_failures": domain_failures,
        "domain_failure_rate": domain_failure_rate
    }

def clusterwise_stats(ids, df):
    subset = df[df["id"].isin(ids)]
    return {
        "cluster_size": len(subset),
        "failure_rate": (subset["success"] == 0).sum() / len(subset),
        "avg_latency": subset["latency_ms"].mean(),
        "avg_tokens": subset["tokens"].mean(),
        "domain_distribution": subset["domain"].value_counts().to_dict()
    }

def cluster_stats(clusters, df):
    results = []
    for cluster in clusters:
        stats = clusterwise_stats(cluster["ids"], df)
        results.append({**cluster, **stats})
    return results

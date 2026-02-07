import matplotlib.pyplot as plt

def failures_by_domain_chart(domain_failures):
    domains = list(domain_failures.keys())
    counts = list(domain_failures.values())
    
    fig, ax = plt.subplots()
    ax.bar(domains, counts)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Failures")
    ax.set_title("Failures by Domain")
    return fig

def latency_distribution_chart(latency_series):
    fig, ax = plt.subplots()
    ax.hist(latency_series, bins=20)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title("Latency Distribution")
    return fig

def embeddings_scatterplot(embeddings_simple):
    fig, ax = plt.subplots()
    colors = ["red" if s == 0 else "green" for s in embeddings_simple["success"]]
    ax.scatter(embeddings_simple["umap_1"], embeddings_simple["umap_2"], c=colors, alpha=0.6)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Embeddings (red=failure, green=success)")
    return fig

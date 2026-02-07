import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#ccc",
    "axes.labelcolor": "#333",
    "text.color": "#333",
    "xtick.color": "#555",
    "ytick.color": "#555",
    "grid.color": "#eee",
    "font.size": 11,
})

FAIL_COLOR = "#FF4B4B"
SUCCESS_COLOR = "#21C354"
ACCENT = "#636EFA"
HIST_COLOR = "#636EFA"

def failures_by_domain_chart(domain_failures):
    domains = list(domain_failures.keys())
    counts = list(domain_failures.values())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(domains, counts, color=FAIL_COLOR, edgecolor="none", height=0.6)
    ax.set_xlabel("Failures")
    ax.set_title("Failures by Domain", fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10, color="#333")
    fig.tight_layout()
    return fig

def latency_distribution_chart(latency_series):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(latency_series, bins=20, color=HIST_COLOR, edgecolor="white", alpha=0.85)
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title("Latency Distribution", fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

def embeddings_scatterplot(embeddings_simple):
    fig, ax = plt.subplots(figsize=(6, 5))
    fail = embeddings_simple[embeddings_simple["success"] == 0]
    success = embeddings_simple[embeddings_simple["success"] == 1]
    ax.scatter(fail["umap_1"], fail["umap_2"], c=FAIL_COLOR, alpha=0.6, s=30, label="Failure", edgecolors="none")
    ax.scatter(success["umap_1"], success["umap_2"], c=SUCCESS_COLOR, alpha=0.6, s=30, label="Success", edgecolors="none")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Embedding Space", fontweight="bold", pad=12)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

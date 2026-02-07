from to_clusters.embeddings import get_embeddings
import numpy as np

def route_input(user_input, results_a, results_b):
    print("[router] embedding user input...")
    embedding = get_embeddings([user_input])[0]

    print("[router] finding closest cluster for A...")
    closest_a = find_closest_cluster(embedding, results_a)
    print("[router] finding closest cluster for B...")
    closest_b = find_closest_cluster(embedding, results_b)

    print("[router] comparing models...")
    recommendation = compare_models(closest_a, closest_b)
    print("[router] done")
    return recommendation

def find_closest_cluster(embedding, results):
    # compare in high-dimensional space using original embeddings
    emb_df = results["embeddings"]
    coords = emb_df.drop(columns=["id", "success"]).values

    # distance from new point to every training point
    dists = np.linalg.norm(coords - embedding, axis=1)

    # for each cluster, compute mean distance
    all_clusters = results["overlaps"]["failure"] + results["overlaps"]["success"]
    best = None
    best_dist = float("inf")
    for c in all_clusters:
        ids = c["ids"]
        mask = emb_df["id"].isin(ids)
        mean_dist = dists[mask].mean()
        if mean_dist < best_dist:
            best_dist = mean_dist
            best = c
    return {"cluster": best, "distance": best_dist}

def score_line(label, cluster):
    if cluster["success"]:
        return f"{label}: confidence {cluster.get('confidence_score', 0):.2f}"
    else:
        return f"{label}: risk {cluster.get('risk_score', 1):.2f}"

def compare_models(closest_a, closest_b):
    a_success = closest_a["cluster"]["success"]
    b_success = closest_b["cluster"]["success"]

    score_a = score_line("Model A", closest_a["cluster"])
    score_b = score_line("Model B", closest_b["cluster"])

    if a_success and not b_success:
        msg = "Model A — this prompt falls in a success cluster for A but a failure cluster for B."
    elif b_success and not a_success:
        msg = "Model B — this prompt falls in a success cluster for B but a failure cluster for A."
    elif a_success and b_success:
        a_conf = closest_a["cluster"].get("confidence_score", 0)
        b_conf = closest_b["cluster"].get("confidence_score", 0)
        if a_conf >= b_conf:
            msg = "Model A — both succeed, but A has higher confidence."
        else:
            msg = "Model B — both succeed, but B has higher confidence."
    else:
        a_risk = closest_a["cluster"].get("risk_score", 1)
        b_risk = closest_b["cluster"].get("risk_score", 1)
        if a_risk <= b_risk:
            msg = "⚠️ WARNING: Both models fail for this prompt. Model A has lower risk."
        else:
            msg = "⚠️ WARNING: Both models fail for this prompt. Model B has lower risk."

    return msg, score_a, score_b

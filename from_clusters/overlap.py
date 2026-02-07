from numpy.linalg import norm

def euclidean_dist(a, b):
    return norm(a - b)

def overlap_scores(failure_centroids, success_centroids):
    scores = []
    for fail_cluster in failure_centroids:
        row = []
        for success_cluster in success_centroids:
            row.append(euclidean_dist(fail_cluster[0], success_cluster[0]))
        scores.append(row)
    return scores

def min_scores(scores):
    success_dist = []
    failure_dist = []

    for row in scores:
        failure_dist.append(min(row))

    for j in range(len(scores[0])):
        success_dist.append(min(scores[i][j] for i in range(len(scores))))

    return {
        "success_dist": success_dist,
        "failure_dist": failure_dist
        }

def overlaps(failure_centroids, success_centroids):
    scores = overlap_scores(failure_centroids, success_centroids)
    min_overlaps = min_scores(scores)

    failure_results = []
    for i, fc in enumerate(failure_centroids):
        failure_results.append({"centroid": fc[0], "features": fc[1], "ids": fc[2], "min_dist": min_overlaps["failure_dist"][i]})

    success_results = []
    for j, sc in enumerate(success_centroids):
        success_results.append({"centroid": sc[0], "features": sc[1], "ids": sc[2], "min_dist": min_overlaps["success_dist"][j]})

    return {"failure": failure_results, "success": success_results}
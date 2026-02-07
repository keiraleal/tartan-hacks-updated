import math

def uniqueness(ovlp):
    all_clusters = ovlp["failure"] + ovlp["success"]
    dists = [c["min_dist"] for c in all_clusters]
    min_d, max_d = min(dists), max(dists)

    for c in all_clusters:
        if max_d == min_d:
            c["uniqueness"] = 0.0
        else:
            c["uniqueness"] = (c["min_dist"] - min_d) / (max_d - min_d)
    return ovlp

def norm_latency(ovlp):
    all_clusters = ovlp["failure"] + ovlp["success"]
    lats = [c["avg_latency"] for c in all_clusters]
    min_l, max_l = min(lats), max(lats)

    for c in all_clusters:
        if max_l == min_l:
            c["operational_risk"] = 0.0
        else:
            c["operational_risk"] = (c["avg_latency"] - min_l) / (max_l - min_l)
    return ovlp

def scale_risk(ovlp):
    all_clusters = ovlp["failure"] + ovlp["success"]
    raw = [math.log(c["cluster_size"]) if c["cluster_size"] > 1 else 0.0 for c in all_clusters]
    min_s, max_s = min(raw), max(raw)

    for c, r in zip(all_clusters, raw):
        if max_s == min_s:
            c["scale"] = 0.0
        else:
            c["scale"] = (r - min_s) / (max_s - min_s)
    return ovlp

def risk_score(ovlp):
    for c in ovlp["failure"]:
        c["risk_score"] = 0.5 * c["uniqueness"] + 0.2 * c["operational_risk"] + 0.3 * c["scale"]

    for c in ovlp["success"]:
        c["confidence_score"] = 0.5 * c["uniqueness"] + 0.2 * (1 - c["operational_risk"]) + 0.3 * c["scale"]

    return ovlp

def risk_level(ovlp):
    for c in ovlp["failure"]:
        c["risk_level"] = "HIGH" if c["risk_score"] > 0.7 else "MODERATE" if c["risk_score"] > 0.3 else "LOW"
    return ovlp
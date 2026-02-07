def deployment_action(cluster):
    why = []
    domain = cluster.get("domain", [])
    act = -1
    if cluster["risk_score"] > 0.85:
        act = 3
        why.append("very low overlap with successful usage, nearly always unsucessful")
    elif cluster["risk_score"] > 0.4 and ("security" in domain or "emotional" in domain):
        act = 3
        why.append("moderate overlap with successful usage, often unsucessful")
        why.append("sensitive domain")
    elif cluster["risk_level"] == "HIGH":
        act = 2
        why.append("low overlap with successful usage, almost always unsucessful")
    elif cluster["risk_level"] == "MODERATE":
        act = 1
        why.append("moderate overlap with successful usage, often unsucessful")
    else:
        act = 0
        why.append("risk is low, high overlap with successful usage")
    if "security" in domain or "emotional" in domain:
        if(act < 3):
            act += 1
            why.append("sensitive domain")
    return {"act": act, "why": why}

def act_lvl_to_word(act):
    if act == 3:
        return "BLOCK DEPLOYMENT"
    if act == 2:
        return "RESTRICTED DEPLOYMENT"
    if act == 1:
        return "LIMITED DEPLOYMENT"
    return "FULL DEPLOYMENT"

def deployment_controls(act):
    if act == 2:
        return ["human review", "monitoring"]
    if act == 1:
        return ["logging", "guardrails"]
    return []

def operational_adj_rec(cluster):
    why = []
    recs = []
    if cluster["avg_latency"] > 1800:
        recs.append("performance monitoring for speed")
        why.append("high average latency")
    if cluster["avg_tokens"] > 600:
        recs.append("add output length cap")
        why.append("high average tokens")
    return {"recs": recs, "why": why}

def confidence(cluster):
    if cluster["cluster_size"] < 5:
        return "Lower confidence due to small cluster."
    return None
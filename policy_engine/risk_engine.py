from policy_engine.risk_to_action import deployment_action, act_lvl_to_word, deployment_controls, operational_adj_rec, confidence

def top_domains(cluster):
    domain_dist = cluster.get("domain_distribution", {})
    if not domain_dist:
        return []
    max_count = max(domain_dist.values())
    return [d for d, count in domain_dist.items() if count == max_count]

def engine_results(ovlp):
    for c in ovlp["failure"]:
        c["success"] = False
        c["name"] = c.get("name", "Unnamed")
        c["domain"] = top_domains(c)
        result = deployment_action(c)
        act = result["act"]
        c["action"] = act_lvl_to_word(act)
        c["action_why"] = result["why"]
        c["controls"] = deployment_controls(act)
        op = operational_adj_rec(c)
        c["operational_adj"] = op["recs"]
        c["operational_why"] = op["why"]
        c["confidence_adj"] = confidence(c)

    for c in ovlp["success"]:
        c["success"] = True
        c["name"] = c.get("name", "Unnamed")
        c["domain"] = top_domains(c)

    return ovlp

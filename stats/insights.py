unstable_threshold = 0.4

def basic_insights(data):
    insight = list()
    if (data["failure_rate"] > unstable_threshold):
        insight.append("Model shows unstable deployment risk")
    top_2_domains = sorted(data["domain_failure_rate"].items(), key=lambda x: x[1], reverse=True)[:2]
    insight.append(f"Highest failure rate in {top_2_domains[0]}. Second highest failure rate in {top_2_domains[1]}")
    return insight
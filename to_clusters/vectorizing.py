from sklearn.feature_extraction.text import TfidfVectorizer

def run_tfidf(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer.get_feature_names_out()

def group_texts_by_cluster(emb, df):
    cluster_texts = []
    for c in sorted(emb["cluster"].unique()):
        ids = emb[emb["cluster"] == c]["id"].values
        texts = df[df["id"].isin(ids)]["input_text"].tolist()
        cluster_texts.append(texts)
    return cluster_texts

def cluster_desc(texts):
    cluster_features = []
    for cluster in texts:
        feature = run_tfidf(cluster)
        cluster_features.append(feature)
    return cluster_features

def retrieve_cluster_features(emb, df):
    cluster_texts = group_texts_by_cluster(emb,df)
    features = cluster_desc(cluster_texts)
    return features
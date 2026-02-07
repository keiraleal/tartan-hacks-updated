import umap
import pandas as pd

def run_umap(data, n_neighbors=15, min_dist=0.05, random_state=0):
    """
    Fits UMAP on data and returns the embedding.
    """
    # Create the model
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        random_state=random_state
    )
    
    # Fit and transform the data
    embedding_sim = reducer.fit_transform(data.drop(columns = ["id","success"]))
    embedding_sim = pd.DataFrame(embedding_sim, columns=["umap_1", "umap_2"])
    embedding_sim["id"] = data["id"].values
    embedding_sim["success"] = data["success"].values
    return embedding_sim, reducer
# It propopagates userinteraction values calculated in play_clusterSongs.py
# to top 10 songs of each clusters.
# This help in creating better input feature for our PPO model.
# And partially solves Cold start problem.

import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

df = pd.read_csv("vae_latent_vectors.csv")
with open("user_song_interactions.json", "r") as f:
    interactions = json.load(f)

latent_cols = [f'latent_{i}' for i in range(5)]
latent_matrix = df[latent_cols].values

optimal_k = 20
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(latent_matrix)
df['latent_vector'] = df[latent_cols].values.tolist()

final_records = []

for interaction in interactions:
    cluster_id = interaction['cluster']
    percentage_listened = interaction['percentage_listened']
    liked = interaction['liked']
    centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
    cluster_songs = df[df['cluster'] == cluster_id].copy()
    cluster_vectors = np.vstack(cluster_songs['latent_vector'].to_numpy())
    distances = euclidean_distances(cluster_vectors, centroid).reshape(-1)
    cluster_songs['distance_to_center'] = distances
    top_songs = cluster_songs.sort_values(by='distance_to_center').head(10)
    for _, row in top_songs.iterrows():
        final_records.append({
            "track_id": row['id'],
            "artist": row['artist'],
            "cluster": int(cluster_id),
            "latent_vector": row['latent_vector'],
            "percentage_listened": percentage_listened,
            "liked": liked
        })
with open("propagated_feedback.json", "w") as f:
    json.dump(final_records, f, indent=2)


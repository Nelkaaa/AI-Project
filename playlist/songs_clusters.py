# As we don't have user-song interaction first we will run cluster algorithm.
# After running cluster algorithm we fetch top 1 songs from each cluster.
# These recommended songs will acts an input for play_clusterSong.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('vae_latent_vectors.csv')

latent_features = df[[col for col in df.columns if 'latent' in col]].values


inertia = []
k_range = range(1, min(100, len(df)))

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(latent_features)
    inertia.append(kmeans.inertia_)

# Plot elbow curve to find optimal k value
plt.figure(figsize=(8,5))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

optimal_k = 20



kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(latent_features)
df['cluster'] = cluster_labels

print("Clustered Songs:")
print(df[['id', 'artist', 'cluster']])

recommendations = []

for cluster_id in range(optimal_k):
    cluster_songs = df[df['cluster'] == cluster_id]
    top_songs = cluster_songs.head(1)

    for _, row in top_songs.iterrows():
        recommendations.append({
            'song_id': row['id'],
            'artist': row['artist'],
            'cluster': cluster_id,
            'latent_0': row['latent_0'],
            'latent_1': row['latent_1'],
            'latent_2': row['latent_2'],
            'latent_3': row['latent_3'],
            'latent_4': row['latent_4'],
        })

recommended_songs = pd.DataFrame(recommendations)

print("\nFinal List of Songs to Recommend:")
print(recommended_songs)

recommended_songs.to_csv('recommended_songs.csv', index=False)
print("\nRecommendations saved to /content/recommended_songs.csv")
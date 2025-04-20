# AI-Project
Adaptive Spotify Music Recommendation using Reinforcement Learning

**State Space Representation**

The state space is represented by a latent vector derived from audio features and metadata of individual tracks. These latent vectors are learned using a Variational Autoencoder (VAE), which compresses high-dimensional audio feature data into a lower-dimensional embedding space. This latent representation captures the essential characteristics of each song, enabling compact and meaningful state descriptions. The state space **𝑆** thus consists of all such latent vectors corresponding to the available tracks, where each vector serves as a unique, continuous representation of the musical content and style of a track. This formulation allows the reinforcement learning agent to generalize across similar tracks and effectively learn user preferences, even in cold start (no user history) scenarios. Since we have a cold-start problem, we cannot use a user-track interaction matrix for state space representation and must instead rely on latent vector generation.

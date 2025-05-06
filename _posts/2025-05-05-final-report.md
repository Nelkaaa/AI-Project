---
layout: post
title: "Adaptive Spotify Music Recommendation using Reinforcement Learning"
date: 2025-05-05
author: Kaustik Ranaware, Noha El Kachach
---
Problem Statement & Motivation
In the context of music recommendation systems, the system is faced with millions of songs to choose from. This creates a vast action space, where each song represents a possible recommendation the system could make (state).
Additionally, user behavior is highly uncertain. Since musical preferences are personal, dynamic, and often unpredictable, there is no fixed formula to determine whether a user will like or dislike a particular song.
Moreover, the system must learn from the user’s listening history to improve future recommendations. This transforms the task from a simple, one-step prediction into a sequential decision-making problem that requires ongoing learning and adaptation.
In such an environment, the agent must learn through interaction by recommending songs, observing user feedback, and refining its strategy over time. However, due to the huge size of the action space, it is infeasible to explore or store information about every possible state.
To address these challenges, we apply deep reinforcement learning, where neural networks enable the agent to generalize from limited experience and make effective predictions without needing to visit or memorize every individual state or action.
Related Solutions
Traditional recommendation methods, such as collaborative filtering, have been widely used to model user-item interactions. But these methods usually treat each recommendation as a single guess, without thinking about the order of what the user listened to before.
Recent works have introduced reinforcement learning (RL) techniques to address these limitations:
1.	Deep Q-Network (DQN) with Simulated Training
One study proposes the use of a Deep Q-Network (DQN) trained in a simulated playlist-generation environment. This approach allows the system to handle the large action space by learning from trial-and-error interactions in a safe, offline setting. 
(Tomasi et al., 2023)
2.	List-wise Recommendations via MDP and Online Simulation
Another work models the recommendation process as a Markov Decision Process (MDP), where:
•	Each moment of interaction is a state,
•	Recommending something is an action,
•	And the user’s reaction (e.g: click, skip) is the reward.
Similar to the DQN-based method, this approach uses an online environment simulator to pre-train and evaluate the model.
(Zhao et al., 2018)
3.	Continuous Action Space with DDPG (Deep Deterministic Policy Gradient)
A third approach leverages DDPG, a type of reinforcement learning designed for continuous action spaces. Rather than selecting songs by Id, this method represents each song using continuous features such as tempo, energy, or mood. This allows the system to handle a much larger number of song options while still providing accurate and varied recommendations.
(Qian, Zhao, & Wang, 2019)
Solution Method:
Our solution leverages Deep Reinforcement Learning using Proximal Policy Optimization (PPO), which is an algorithm that trains both a policy network and a value network at the same time, improving them together throughout the learning process.
We adopt an Actor-Critic architecture, where:
•	The actor represents the policy (i.e., how the system chooses songs),
•	The critic estimates the value of the current state (i.e., how promising the situation is based on the user's history).
•	State Space:
Each state includes the currently playing song along with its features (such as tempo, energy, or genre), all represented as a vector.
•	Action Space:
The action corresponds to selecting the next song from the available pool.
•	Reward Function:
The system receives a positive reward when a user likes or listens to the song fully, and a negative reward if the user skips it, especially if they skip it immediately.
Solution Implementation:
1. State Representation
•	Each song is originally represented in an 11-dimensional feature space (e.g., tempo, energy, etc.).
•	To make the model more efficient, we use a Variational Autoencoder (VAE) to reduce this to a 5-dimensional latent space.
•	These compressed vectors become our state representations.
•	All feature values are normalized to the range [0, 1].
2.Value Network
•	Input: The 5D latent vector representing the current song.
•	Output: A single value estimating the expected future reward (i.e., how good the current situation is).
•	Training: The value network is trained to minimize the Mean Squared Error (MSE) between its prediction and the actual return.
3. Policy Network
•	Input: Same 5D latent vector.
•	Output: A probability distribution over the top 4 recommended songs, filtered using k-Nearest Neighbors (kNN).
•	Training: Trained using PPO loss, based on advantage estimates:
 
•	The model either samples from or selects the most likely action (i.e., next song) to maximize user satisfaction.
4.Reward

It is calculated by: 
 
Such that: 
•	percentage_listened: how much of the song was played (e.g., 0.7 if 70% was played).
•	liked: a binary signal (1 if the user liked the song, 0 otherwise).
•	λ: a weight that emphasizes the importance of “liking” (e.g., 10 or 100).
5. Offline Training
We use offline reinforcement learning, training both:
•	A policy network  to decide which song to recommend
•	A value network  to estimate how good a state is.
This is all done using pre-collected logs (records of past user interactions that have already been saved).
6. Inference Phase
Once trained, the model is used online as follows:
1.	Getting the Current Song's Vector (State)
→ The 5D latent representation of the current song.
2.	Using kNN to Filter Action Space
→ Find the top 4 nearest songs as potential recommendations.
3.	Running the Value & Policy Network
→ Generate values & probabilities over those top 4 candidates.
4.	Choosing the Next Song
→ Either pick the most probable one or sample based on the policy.
If real-time feedback is available (liked/skipped), the system can log new interactions and periodically fine-tune the models (online learning).
7. Reward Propagation with Clustering
1. Clustering the Songs
•	We grouped songs into clusters based on how similar they are by using K-Means on their 2D latent vectors.
2. Collecting Real Feedback for a Few Songs in Each Cluster
•	From each cluster, we picked around 4 songs.
•	For each of those songs, we record data:
o	How much of the song was played (percentage_listened, e.g., 0.64)
o	Whether the user liked it (liked, e.g., 0.9)
o	And we compute the reward.
3. Estimate Rewards for the Other Songs in the Cluster
For each untested song within a cluster, we:
•	Calculated its distance to the cluster centroid using Euclidean distance:
d=∥latent_vector−centroid∥ 
This distance indicates how similar the song is to the center of the cluster (and by extension, to the top-4 songs used for evaluation).
•	Normalized the distances so that the maximum distance within the cluster is scaled to 1.
•	Applied a decay function based on the normalized distance. Songs closer to the centroid receive rewards similar to the top-4 songs, while those farther away receive proportionally lower rewards. This allows the estimated reward to decrease smoothly with increasing distance.
References:
Nick Qian - Sophie Zhao - Yizhou Wang. (n.d.). Spotify Reinforcement Learning Recommendation System. Large-Scale Distributed Sentiment Analysis with RNN. https://sophieyanzhao.github.io/AC297r_2019_SpotifyRL/2019-12-14-Spotify-Reinforcement-Learning-Recommendation-System/ 
Tomasi, F., Cauteruccio, J., Kanoria, S., Ciosek, K., Rinaldi, M., & Dai, Z. (2023, October 13). Automatic Music Playlist Generation via simulation-based reinforcement learning. arXiv.org. https://arxiv.org/abs/2310.09123 
Zhao, X., Xia, L., Zhang, L., Ding, Z., Yin, D., & Tang, J. (2018). Deep reinforcement learning for page-wise recommendations. Proceedings of the 12th ACM Conference on Recommender Systems, 95–103. https://doi.org/10.1145/3240323.3240374 








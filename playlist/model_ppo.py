# Actor Critic based PPO algorithm setup.
# It produces given the current state(latent vectors) and action (songs) and produces new state.

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from sklearn.preprocessing import MinMaxScaler

with open('propagated_feedback.json') as f:
    user_data = json.load(f)

song_df = pd.read_csv('vae_latent_vectors.csv')
latent_cols = [col for col in song_df.columns if 'latent' in col]
scaler = MinMaxScaler()
song_df[latent_cols] = scaler.fit_transform(song_df[latent_cols])
track_to_latent = {
    row['id']: np.array([row[col] for col in latent_cols], dtype=np.float32)
    for _, row in song_df.iterrows()
}
data = []
for entry in user_data:
    state = np.array(entry['latent_vector'], dtype=np.float32)
    action = entry['track_id']
    reward = 0.5 * entry['percentage_listened'] + 1.0 * entry['liked']
    next_state = track_to_latent[action]
    data.append((state, action, reward, next_state))

all_track_ids = song_df['id'].tolist()
track_id_to_idx = {tid: i for i, tid in enumerate(all_track_ids)}

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.out(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

state_dim = 5
action_dim = len(all_track_ids)

# Init model
actor = Actor(state_dim, action_dim)
critic = Critic(state_dim)

# Hyperparameter
gamma = 0.99
clip_eps = 0.2
actor_lr = 3e-4
critic_lr = 1e-4
max_grad_norm = 0.5
epochs = 100
ppo_epochs = 4

# Optimizer
actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

for epoch in range(epochs):
    states, actions, rewards, next_states = [], [], [], []
    for state, action_id, reward, next_state in data:
        action_idx = track_id_to_idx[action_id]
        states.append(state)
        actions.append(action_idx)
        rewards.append(reward)
        next_states.append(next_state)

    states = torch.tensor(np.array(states), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
    rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)
    with torch.no_grad():
        old_probs = actor(states)
        old_dist = Categorical(old_probs)
        old_log_probs = old_dist.log_prob(actions)
        values = critic(states).squeeze()
        next_values = critic(next_states).squeeze()
        advantages = rewards + gamma * next_values - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(ppo_epochs):
        critic_optim.zero_grad()
        current_values = critic(states).squeeze()
        critic_loss = F.mse_loss(current_values, rewards + gamma * next_values.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
        critic_optim.step()

        actor_optim.zero_grad()
        new_probs = actor(states)
        new_dist = Categorical(new_probs)
        new_log_probs = new_dist.log_prob(actions)

        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
        actor_optim.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Actor Loss={actor_loss.item():.4f}, Critic Loss={critic_loss.item():.4f}")

def recommend(state_vector, top_k=10):
    state_tensor = torch.tensor(state_vector, dtype=torch.float32)
    with torch.no_grad():
        probs = actor(state_tensor).numpy()
    top_indices = np.argsort(probs)[-top_k:][::-1]
    return [all_track_ids[i] for i in top_indices]

def track_recommendation_trace(initial_state, steps=3, top_k=20, save_path="final_recommendations.json"):
    current_state = initial_state.copy()
    final_recommendations = []

    print("\nRecommendation Trace:")
    for step in range(steps):
        recommendations = recommend(current_state, top_k)
        print(f"\nStep {step + 1}:")
        print(f"Current State: {current_state.round(4)}")
        print(f"Top Recommendations: {recommendations}")
        if step == steps - 1:
            final_recommendations = recommendations
        selected_track = recommendations[0]
        selected_latent = track_to_latent[selected_track]
        current_state = 0.7 * current_state + 0.3 * selected_latent
    with open(save_path, 'w') as f:
        json.dump(final_recommendations, f)

    print(f"\nSaving final recommendations to {save_path}")
    return current_state
if __name__ == "__main__":
    print("\nRecommendations:")
    initial_state = np.random.rand(state_dim)
    final_state = track_recommendation_trace(initial_state)
    print(f"\nNew User State: {final_state.round(4)}")



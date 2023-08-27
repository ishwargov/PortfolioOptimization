import gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common import set_global_seeds
from stable_baselines3.common.utils import conv, linear
from stable_baselines3 import A2C
import torch
from torch import nn

class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.mu = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Tanh(),
        )
        self.std = nn.Sequential(
            nn.Linear(state_dim, n_actions),
            nn.Softplus(),
        )
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, x):
        mu = self.mu(x)
        log_std = self.std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, torch.exp(log_std)

    def get_dist(self, x):
        mean, std = self.forward(x)
        dist = torch.distributions.normal.Normal(mean, std)
        return dist

    def get_action(self, x):
        dist = self.get_dist(x)
        act = dist.rsample()
        return torch.tanh(act), dist.log_prob(act).sum(axis=1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim+action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, s, a):
        inp = torch.cat((s, a), dim=1).to(device)
        return self.q(inp)

# Risk Sensitive Actor Critic


class RiskAC(object):
    def __init__(self, state_dim, action_dim, max_action, ref_state, ref_action, discount=0.99, tau=0.005, lr=1e-4, alpha=1.0):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.ref_state = torch.tensor(ref_state, device=device).reshape(1, -1)
        self.ref_action = torch.tensor(
            ref_action, device=device).reshape(1, -1)
        self.discount = discount
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.alpha = alpha
        self.gamma = discount

    # Training
    def train(self, replay_buffer, batch_size=256):
        transitions = replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))
        state = torch.cat(batch.state).to(device).reshape(-1, self.state_dim)
        action = torch.cat(batch.action).to(
            device).reshape(-1, self.action_dim)
        next_state = torch.cat(batch.next_state).to(
            device).reshape(-1, self.state_dim)
        reward = torch.cat(batch.reward).to(device).reshape(-1, 1)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.get_action(next_state)
            target_q = self.critic_target(next_state, next_action)
            cur_q = self.critic_target(state, action)
            ref_q = self.critic_target(self.ref_state, self.ref_action)
            target_q = torch.div(target_q, (ref_q+1e-9))
            target_q = torch.exp(-reward)*target_q
            target_q = (target_q - self.alpha * next_log_prob)

        current_q = self.critic(state, action)
        q_loss = F.mse_loss(target_q, current_q)
        self.critic_opt.zero_grad()
        q_loss.backward()
        self.critic_opt.step()

        for params in self.critic.parameters():
            params.requires_grad = False

        a, log_pi_a = self.actor(state)
        q = self.critic(state, a)

        a_loss = (self.alpha * log_pi_a - q).mean()
        self.actor_opt.zero_grad()
        a_loss.backward()
        self.actor_opt.step()

        for params in self.critic.parameters():
            params.requires_grad = True

        # Soft Update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, checkpoint_path):
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))

        state = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'ref_state': self.ref_state,
            'ref_action': self.ref_action,
            'discount': self.discount,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'tau': self.tau,
            'alpha': self.alpha,
            'gamma': self.gamma
        }

        torch.save(state, checkpoint_path)

    def load(self, checkpoint_path):
        state = torch.load(checkpoint_path)

        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.actor_opt.load_state_dict(state['actor_opt'])
        self.critic_opt.load_state_dict(state['critic_opt'])
        self.ref_state = state['ref_state']
        self.ref_action = state['ref_action']
        self.discount = state['discount']
        self.state_dim = state['state_dim']
        self.action_dim = state['action_dim']
        self.tau = state['tau']
        self.alpha = state['alpha']
        self.gamma = state['gamma']


def select_action(actor, state):
    with torch.no_grad():
        state = state.reshape(1, -1).clone().detach().to(device)
        a, _ = actor.get_action(state)
    return a.cpu().numpy().flatten().reshape(1, -1)

# train


memory = ReplayMemory(50000)
episodes = []
episodes_assets = []
BATCH_SIZE = 1024
ref_state = env.observation_space.sample()
ref_action = env.action_space.sample()
RAC = RiskAC(state_dim=env.observation_space.shape[0],
             action_dim=env.action_space.shape[0],
             max_action=1,
             ref_state=ref_state,
             ref_action=ref_action
             )
for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32,
                         device=device).unsqueeze(0)
    last_asset = []
    last_act = []
    last_state = []
    for t in count():
        action = select_action(RAC.actor, state)
        observation, reward, terminated, _ = env.step(action.reshape(-1))
        reward = torch.tensor(np.array([reward]), device=device)
        done = terminated
        if terminated:
            next_state = None
            episodes.append(last_act)
            episodes_assets.append(last_asset)
        else:
            next_state = torch.tensor(
                observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state.float(), torch.tensor(
                np.array([action]), device=device).float(), next_state.float(), reward.float())

            # Move to the next state
            state = next_state
            last_act = env.actions_memory
            last_asset = env.asset_memory
            last_state = env.state_memory
            #last_episode, last_asset  = env.env_method(method_name="get_action_memory")[0]

        # Perform one step of the optimization (on the policy network)
        if len(memory) >= BATCH_SIZE:
            RAC.train(memory)

        if(done):
            break

print('Complete')

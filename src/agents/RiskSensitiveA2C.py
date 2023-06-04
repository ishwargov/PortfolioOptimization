import gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common import set_global_seeds
from stable_baselines3.common.utils import conv, linear
from stable_baselines3 import A2C

class RiskSensitiveA2C(A2C):
    def __init__(self, policy, env, gamma=0.99, n_steps=5, vf_coef=0.5,
                 ent_coef=0.01, max_grad_norm=0.5, learning_rate=7e-4,
                 alpha=1, eps=1e-8, comment='', **kwargs):
        super(RiskSensitiveA2C, self).__init__(policy, env, gamma=gamma,
                                               n_steps=n_steps, vf_coef=vf_coef,
                                               ent_coef=ent_coef, max_grad_norm=max_grad_norm,
                                               learning_rate=learning_rate, **kwargs)
        self.alpha = alpha
        self.eps = eps
        self.comment = comment

    def risk_sensitive_cost(self, log_prob, actions, returns, values):
        # Compute risk-sensitive cost using the sensitivity formula from Borkar's paper
        # log_prob: log-probability of the actions taken by the agent
        # actions: actions taken by the agent
        # returns: discounted rewards obtained by the agent
        # values: estimated values of the states visited by the agent

        n_steps = len(actions)
        R = np.zeros((n_steps, n_steps))
        for i in range(n_steps):
            for j in range(i, n_steps):
                if i == j:
                    R[i][j] = np.exp(self.alpha * returns[i] - values[i])
                else:
                    R[i][j] = R[i][j - 1] * np.exp(self.alpha * returns[j] - values[j])

        C = np.zeros((n_steps,))
        for t in range(n_steps):
            C[t] = np.sum(R[t, t:n_steps] * log_prob[t:n_steps])

        cost = np.mean(C)
        return cost

    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="A2C",
              reset_num_timesteps=True):

        with self.graph.as_default():
            self._setup_learn()

        runner = A2CRunner(env=self.env, model=self, n_steps=self.n_steps)
        epinfobuf = deque(maxlen=100)
        tstart = time.time()

        # Start training
        for update in range(1, total_timesteps // self.batch_size + 1):
            obs, states, rewards, masks, actions, values, log_probs = runner.run()
            returns = self._get_returns(rewards, masks, values)

            # Compute risk-sensitive cost using the sensitivity formula
            cost = self.risk_sensitive_cost(log_probs, actions, returns, values)

            # Compute gradients and update parameters
            self._train_step(obs, returns, masks, actions, values, log_probs, cost)

            # Log training statistics
            epinfos = runner.get_episode_rewards()
            if epinfos:
                epinfobuf.extend(epinfos)
            if callback is not None:
                callback(locals(), globals())

            if (update * self.batch_size) % log_interval == 0:
                fps = int((update * self.batch_size) / (time.time() - tstart))
                logger.record_tabular("steps", update * self.batch_size)
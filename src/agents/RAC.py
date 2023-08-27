from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import Schedule
import torch as th

# Risk Sensitive Soft Actor Critic
class RAC(SAC):
    def __init__(
        self,
        risk_lambda: float = 0.5,
        risk_param_schedule: Schedule = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.risk_lambda = risk_lambda
        self.risk_param_schedule = risk_param_schedule
    
    def train(self, gradient_steps: int, batch_size: int = 100, replay_buffer=None) -> None:
        # Compute the risk-sensitive reward weights
        risk_weights = self.compute_risk_weights()

        # Original SAC training code here
        super().train(gradient_steps=gradient_steps, batch_size=batch_size, replay_buffer=replay_buffer)
        
        # Update the critic with the risk-sensitive reward weights
        for _ in range(gradient_steps):
            self._update_critic_with_risk_weights(risk_weights, batch_size, replay_buffer)

    def compute_risk_weights(self):
        # Compute the risk-sensitive reward weights using the sensitivity formula
        with th.no_grad():
            q1_values = self.critic.q1_forward(self._last_obs).squeeze(-1)
            q2_values = self.critic.q2_forward(self._last_obs).squeeze(-1)
            min_q_values = th.min(q1_values, q2_values)
            risk_weights = th.exp(-self.risk_lambda * min_q_values)
        return risk_weights
    
    def _update_critic_with_risk_weights(self, risk_weights, batch_size, replay_buffer):
        # Update the critic with the risk-sensitive reward weights
        # Sample a batch of transitions from the replay buffer
        obs, action, reward, next_obs, done = replay_buffer.sample(batch_size)
        
        # Compute the risk-sensitive rewards using the sensitivity formula
        with th.no_grad():
            next_action, next_log_prob = self.actor.forward(next_obs)
            q1_next_target, q2_next_target = self.critic_target.forward(next_obs, next_action)
            min_q_next_target = th.min(q1_next_target, q2_next_target)
            rs_rewards = -self.risk_lambda * th.exp(-self.risk_lambda * (reward + self.gamma * (1 - done) * min_q_next_target))
        
        # Compute the Q targets with the risk-sensitive rewards
        q1_target = rs_rewards + self.gamma * (1 - done) * self.critic_target.q1_forward(next_obs, next_action).squeeze(-1)
        q2_target = rs_rewards + self.gamma * (1 - done) * self.critic_target.q2_forward(next_obs, next_action).squeeze(-1)

        # Compute the Q loss and update the critic
        q1_pred = self.critic.q1_forward(obs, action).squeeze(-1)
        q2_pred = self.critic.q2_forward(obs, action).squeeze(-1)
        q1_loss = th.nn.functional.mse_loss(q1_pred, q1_target, reduction='none')
        q2_loss = th.nn.functional.mse_loss(q2_pred, q2_target, reduction='none')
        q_loss = (risk_weights * q1_loss + risk_weights * q2_loss).mean()
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()